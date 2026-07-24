// ============================================================
// fused_moe_flagship_op.cpp — PyTorch custom op for FusedMoE
//
// Registers torch.ops.fl_custom.fused_moe_flagship so the
// vLLM FL plugin can invoke the AscendC kernel directly.
//
// Architecture:
//   1. Receive PyTorch tensors (NPU resident)
//   2. Generate tiling data (using MultiCoreMatmulTiling)
//   3. RTC compile kernel source → load via ACL runtime (PDF §2.3.1.5 + §2.4.2)
//   4. Launch kernel, sync, return output tensor
//
// Kernel launch API (CANN 8.5.0):
//   aclrtcCreateProg → aclrtcCompileProg → aclrtcGetBinData →
//   aclrtBinaryLoadFromData → aclrtBinaryGetFunction →
//   aclrtKernelArgsInit → aclrtKernelArgsAppend × N →
//   aclrtKernelArgsFinalize → aclrtLaunchKernelWithConfig →
//   aclrtSynchronizeStream
//
//   Reference: dev_guide §2.4.2 "Kernel加载与执行（加载二进制）"
//
// Build: part of libfused_moe_flagship.so (torch_op CMake target)
// Platform: Ascend 910B, CANN 8.5.0
// ============================================================

#include <torch/extension.h>

// CANN ACL Runtime API (kernel loading & launch)
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_rt_compile.h"   // RTC: aclrtcCreateProg, aclrtcCompileProg (PDF §2.3.1.5)

// Tiling generation API
#include "tiling/platform/platform_ascendc.h"
#include "adv_api/matmul/bmm_tiling.h"

// Shared tiling data structure
#include "fused_moe_tiling.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <new>
#include <fstream>

// ============================================================
// Kernel binary loading: try pre-compiled .o first, fallback to RTC
// ============================================================
// Build-time bisheng produces op_kernel/fused_moe_flagship.o (CMakeLists.txt §2.3.1.3).
// RTC (§2.3.1.5) used as fallback if .o not found.
extern const char* g_kernelSource;

// ============================================================
// Tiling constants (matches op_host)
// ============================================================
namespace {
    constexpr int32_t DEFAULT_TILE_M = 32;
    constexpr int32_t DEFAULT_TILE_K = 64;
    constexpr int32_t DEFAULT_TILE_N = 128;
    constexpr int32_t MAX_EXPERTS = 64;
}

// ============================================================
// Standalone tiling generation (no TilingContext dependency)
//
// Reuses the same MultiCoreMatmulTiling API as the host-side
// op_host/fused_moe_flagship.cpp, but without the GE framework
// TilingContext dependency.
//
// PDF 5.2.2 MultiCoreMatmulTiling, 6.2.1 PlatformAscendC
// ============================================================
static bool GenerateStandaloneMatmulTiling(
    AscendC::tiling::TCubeTiling& outCubeTiling,
    int32_t m, int32_t n, int32_t k,
    bool transB)
{
    // Kernel Launch (直调) 模式: 使用 PlatformAscendCManager::GetInstance()
    // 接口参考 6.2.2: 自动检测当前 NPU 硬件, 无需传入 soc version
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();

    matmul_tiling::MultiCoreMatmulTiling tilingObj(*platform);
    tilingObj.SetDim(1);

    tilingObj.SetAType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, false);
    tilingObj.SetBType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16, transB);
    tilingObj.SetCType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT16);

    tilingObj.SetShape(m, n, k);
    tilingObj.SetOrgShape(m, n, k);
    tilingObj.SetSingleShape(m, n, k);
    tilingObj.SetMatmulConfigParams(0);

    return (tilingObj.GetTiling(outCubeTiling) != -1);
}

// ============================================================
// FusedMoEFlagship implementation (host-side orchestration)
// ============================================================
static torch::Tensor FusedMoEFlagshipForward(
    torch::Tensor hidden_states,       // FP16 [num_tokens, hidden_dim]
    torch::Tensor w1,                  // FP16 [num_experts, inter_dim, hidden_dim]
    torch::Tensor w2,                  // FP16 [num_experts, hidden_dim, inter_dim/2]
    torch::Tensor temp_buffer,         // FP16 [num_cores * tile_m * inter_dim] (preallocated)
    torch::Tensor sorted_token_ids,    // INT32 [num_tokens_post_padded]
    torch::Tensor topk_weights,        // FP32 [num_tokens, top_k]
    torch::Tensor sorted_weights,      // FP32 [num_tokens_post_padded]
    torch::Tensor output,              // FP16 [num_tokens_post_padded, hidden_dim] (OUT)
    std::vector<int64_t> tokens_per_expert,  // [num_experts] aligned counts
    std::vector<int64_t> token_offsets,       // [num_experts] start offsets
    std::string activation,                  // "silu"
    int64_t num_tokens_post_padded,
    int64_t num_cores)
{
    // --- Validate ---
    TORCH_CHECK(activation == "silu", "Only silu activation is supported");
    TORCH_CHECK(hidden_states.dtype() == torch::kFloat16, "hidden_states must be FP16");
    TORCH_CHECK(w1.dtype() == torch::kFloat16, "w1 must be FP16");
    TORCH_CHECK(w2.dtype() == torch::kFloat16, "w2 must be FP16");

    int32_t num_tokens   = static_cast<int32_t>(hidden_states.size(0));
    int32_t hidden_dim   = static_cast<int32_t>(hidden_states.size(1));
    int32_t num_experts  = static_cast<int32_t>(w1.size(0));
    int32_t inter_dim    = static_cast<int32_t>(w1.size(1));
    int32_t top_k        = static_cast<int32_t>(topk_weights.size(1));
    int32_t tileM        = std::min(DEFAULT_TILE_M, num_tokens);
    int32_t tileK        = std::min(DEFAULT_TILE_K, hidden_dim);
    int32_t tileN        = std::min(DEFAULT_TILE_N, inter_dim / 2);
    int32_t block_dim    = static_cast<int32_t>(num_cores);

    // --- Create NPU stream ---
    // CANN 8.5.0 dev_guide §2.4.2: use aclrtCreateStream, NOT aclrtGetStream
    aclrtStream stream = nullptr;
    aclError ret = aclrtCreateStream(&stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtCreateStream failed: ", ret);

    // --- Get NPU device pointers ---
    // Torch NPU tensors reside in device memory;
    // data_ptr() returns the NPU GM address directly.
    void* hs_gm  = reinterpret_cast<void*>(hidden_states.data_ptr());
    void* w1_gm  = reinterpret_cast<void*>(w1.data_ptr());
    void* w2_gm  = reinterpret_cast<void*>(w2.data_ptr());
    void* tmp_gm = reinterpret_cast<void*>(temp_buffer.data_ptr());
    void* sid_gm = reinterpret_cast<void*>(sorted_token_ids.data_ptr());
    void* tw_gm  = reinterpret_cast<void*>(topk_weights.data_ptr());
    void* sw_gm  = reinterpret_cast<void*>(sorted_weights.data_ptr());
    void* out_gm = reinterpret_cast<void*>(output.data_ptr());

    // --- Allocate GM for system workspace + tiling data ---
    // workspace 布局: [sysWorkspace][padding][tilingData]
    //   sysWorkspace: Matmul 高阶API 需要的系统 workspace (PDF §4.4.2.1.2)
    //   padding: 原 per-core temp buffer 大小 (保持偏移对齐, 当前 kernel 使用 tempBuffer arg 3)
    //   tilingData: FusedMoeTilingData (由 kernel 通过 __builtin_memcpy 读取)
    size_t sysWsSize = 0;
    {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        sysWsSize = platform->GetLibApiWorkSpaceSize();
    }

    size_t ws_size = static_cast<size_t>(num_cores) *
                     static_cast<size_t>(tileM) *
                     static_cast<size_t>(inter_dim) *
                     sizeof(uint16_t);

    void* ws_gm = nullptr;
    ret = aclrtMalloc(&ws_gm, sysWsSize + ws_size + sizeof(FusedMoeTilingData),
                      ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: ", ret);

    // tiling 数据在系统 workspace + padding 之后
    void* tiling_gm = static_cast<uint8_t*>(ws_gm) + sysWsSize + ws_size;

    // --- Generate tiling data ---
    FusedMoeTilingData tiling = {};
    tiling.numTokens            = static_cast<uint32_t>(num_tokens);
    tiling.hiddenDim            = static_cast<uint32_t>(hidden_dim);
    tiling.intermediateDim      = static_cast<uint32_t>(inter_dim);
    tiling.numExperts           = static_cast<uint32_t>(num_experts);
    tiling.topK                 = static_cast<uint32_t>(top_k);
    tiling.numTokensPostPadded  = static_cast<uint32_t>(num_tokens_post_padded);
    tiling.tileM                = static_cast<uint32_t>(tileM);
    tiling.tileK                = static_cast<uint32_t>(tileK);
    tiling.tileN                = static_cast<uint32_t>(tileN);
    tiling.numCores             = static_cast<uint32_t>(block_dim);

    int32_t expCount = std::min(num_experts, MAX_EXPERTS);
    for (int32_t i = 0; i < expCount; i++) {
        tiling.tokensPerExpert[i] = static_cast<uint32_t>(
            i < static_cast<int32_t>(tokens_per_expert.size())
                ? tokens_per_expert[i] : 0);
        tiling.tokenOffsets[i] = static_cast<uint32_t>(
            i < static_cast<int32_t>(token_offsets.size())
                ? token_offsets[i] : 0);
    }

    // --- Generate Cube Tiling for MM1 & MM2 ---
    // MM1: hidden_states × w1^T → gate_up [tileM, interDim]
    bool ok = GenerateStandaloneMatmulTiling(
        tiling.cubeTilingMM1, tileM, inter_dim, tileK, true);
    TORCH_CHECK(ok, "Failed to generate cube tiling for MM1");

    // MM2: activated × w2^T → output [tileM, hiddenDim]
    ok = GenerateStandaloneMatmulTiling(
        tiling.cubeTilingMM2, tileM, hidden_dim, tileN, true);
    TORCH_CHECK(ok, "Failed to generate cube tiling for MM2");

    // --- Copy tiling data to GM ---
    ret = aclrtMemcpy(tiling_gm, sizeof(FusedMoeTilingData),
                      &tiling, sizeof(FusedMoeTilingData),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpy tiling data failed: ", ret);

    // ============================================================
    // Kernel loading: pre-compiled .o (bisheng §2.3.1.3) → RTC fallback (§2.3.1.5)
    // ============================================================
    // Prefer bisheng-compiled kernel .o over RTC. The .o is produced by
    // CMake at build/op_kernel/fused_moe_flagship.o with -x asc flag.
    // RTC compilation observed to produce IFU/CCU errors (invalid instructions)
    // leading to 507014 aicore timeout; pre-compiled .o avoids this.
    std::vector<char> binData;
    {
        // Search for kernel .o relative to the shared library or CWD
        std::string oPaths[] = {
            "op_kernel/fused_moe_flagship.o",
            "./op_kernel/fused_moe_flagship.o",
            "../op_kernel/fused_moe_flagship.o",
        };
        bool loaded = false;
        for (const auto& path : oPaths) {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) continue;
            size_t sz = static_cast<size_t>(f.tellg());
            f.seekg(0);
            binData.resize(sz);
            f.read(binData.data(), sz);
            loaded = true;
            std::cerr << "[Kernel] Loaded pre-compiled: " << path << " (" << sz << " bytes)" << std::endl;
            break;
        }

        if (!loaded) {
            // Fallback to RTC
            std::cerr << "[Kernel] .o not found, falling back to RTC compile" << std::endl;
            aclrtcProg rtcProg = nullptr;
            ret = aclrtcCreateProg(&rtcProg, g_kernelSource,
                                   "fused_moe_flagship", 0, nullptr, nullptr);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtcCreateProg failed: ", ret);

            const char* compileOpts[] = {"--npu-arch=dav-2201", "-O2"};
            ret = aclrtcCompileProg(rtcProg, 2, compileOpts);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtcCompileProg failed: ", ret);

            size_t binSize = 0;
            ret = aclrtcGetBinDataSize(rtcProg, &binSize);
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtcGetBinDataSize failed: ", ret);

            binData.resize(binSize);
            ret = aclrtcGetBinData(rtcProg, binData.data());
            TORCH_CHECK(ret == ACL_SUCCESS, "aclrtcGetBinData failed: ", ret);

            size_t logSize = 0;
            if (aclrtcGetCompileLogSize(rtcProg, &logSize) == ACL_SUCCESS && logSize > 0) {
                std::vector<char> compileLog(logSize);
                if (aclrtcGetCompileLog(rtcProg, compileLog.data()) == ACL_SUCCESS) {
                    std::string logStr(compileLog.data(), logSize);
                    if (!logStr.empty())
                        std::cerr << "[RTC Compile Log]: " << logStr << std::endl;
                }
            }
            aclrtcDestroyProg(&rtcProg);
        }
    }

    // Load binary to device (PDF §2.4.2)
    // Magic value: AI CORE (Cube+Vector), not Vector-only default.
    // Without explicit magic, runtime loads as Vector Core → Cube instructions
    // hang → FixPipe timeout → 507014.
    aclrtBinHandle binHandle = nullptr;
    aclrtBinaryLoadOption loadOpt;
    loadOpt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    loadOpt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions loadOpts;
    loadOpts.numOpt = 1;
    loadOpts.options = &loadOpt;
    ret = aclrtBinaryLoadFromData(
        reinterpret_cast<const void*>(binData.data()),
        binData.size(), &loadOpts, &binHandle);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtBinaryLoadFromData failed: ", ret);

    // --- Step 3: Get function handle ---
    aclrtFuncHandle funcHandle = nullptr;
    ret = aclrtBinaryGetFunction(binHandle, "fused_moe_flagship", &funcHandle);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtBinaryGetFunction failed: ", ret);

    // --- Step 4: Build kernel argument list ---
    //   aclrtKernelArgsInit → aclrtKernelArgsAppend × N → aclrtKernelArgsFinalize
    //   PDF §2.4.2 Step 2: 获取核函数句柄并根据核函数句柄操作其参数列表
    aclrtArgsHandle argsHandle = nullptr;
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtKernelArgsInit failed: ", ret);

    aclrtParamHandle paramHandle = nullptr;

    // Kernel expects 10 GM_ADDR arguments:
    //   0: hiddenStates   1: w1         2: w2        3: tempBuffer
    //   4: sortedTokenIds 5: topkWeights 6: sortedWeights 7: output
    //   8: workspace      9: tiling
    auto appendArg = [&](void* ptr) -> void {
        aclrtKernelArgsAppend(argsHandle, (void**)&ptr,
                              sizeof(uintptr_t), &paramHandle);
    };
    appendArg(hs_gm);    // 0: hiddenStates
    appendArg(w1_gm);    // 1: w1
    appendArg(w2_gm);    // 2: w2
    appendArg(tmp_gm);   // 3: tempBuffer
    appendArg(sid_gm);   // 4: sortedTokenIds
    appendArg(tw_gm);    // 5: topkWeights
    appendArg(sw_gm);    // 6: sortedWeights
    appendArg(out_gm);   // 7: output
    appendArg(ws_gm);    // 8: workspace
    appendArg(tiling_gm);// 9: tiling

    ret = aclrtKernelArgsFinalize(argsHandle);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtKernelArgsFinalize failed: ", ret);

    // --- Step 5: Launch kernel ---
    //   aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, prop, args, reserved)
    //   PDF §2.4.2 Step 3: 调用aclrtLaunchKernelWithConfig启动算子计算任务
    ret = aclrtLaunchKernelWithConfig(
        funcHandle, static_cast<uint32_t>(block_dim),
        stream, nullptr, argsHandle, nullptr);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtLaunchKernelWithConfig failed: ", ret);

    // --- Step 6: Wait for completion ---
    ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed: ", ret);

    // --- Cleanup ---
    aclrtDestroyStream(stream);
    aclrtBinaryUnLoad(binHandle);
    aclrtFree(ws_gm);

    return output;
}

// ============================================================
// TORCH_LIBRARY registration
//
// Matches vllm_fl call site:
//   torch.ops.fl_custom.fused_moe_flagship(
//       permuted_hidden, w1, w2, temp_buffer,
//       sorted_token_ids, topk_weights, sorted_weights,
//       permuted_output,
//       tokens_per_expert.tolist(), token_offsets.tolist(),
//       activation, num_tokens_post_padded, num_cores)
// ============================================================
TORCH_LIBRARY(fl_custom, m) {
    m.def("fused_moe_flagship("
          "Tensor hidden_states, "
          "Tensor w1, "
          "Tensor w2, "
          "Tensor temp_buffer, "
          "Tensor sorted_token_ids, "
          "Tensor topk_weights, "
          "Tensor sorted_weights, "
          "Tensor output, "
          "int[] tokens_per_expert, "
          "int[] token_offsets, "
          "str activation, "
          "int num_tokens_post_padded, "
          "int num_cores) -> Tensor");
}

// PrivateUse1 = Ascend NPU device type in PyTorch
TORCH_LIBRARY_IMPL(fl_custom, PrivateUse1, m) {
    m.impl("fused_moe_flagship", &FusedMoEFlagshipForward);
}
