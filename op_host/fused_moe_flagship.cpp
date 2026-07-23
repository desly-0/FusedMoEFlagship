// ============================================================
// fused_moe_flagship.cpp — Host 侧算子注册 + Tiling 实现
//
// 注册 FusedMoEFlagship 自定义算子:
// - 定义输入/输出/属性
// - 实现 Tiling 函数 (包含 MultiCoreMatmulTiling 生成 CubeTiling)
// - 实现 Shape 推导
//
// 平台: Ascend 910B, CANN 8.5.0
// ============================================================

#include <cstdint>
#include <algorithm>
#include "register/ops.h"
#include "tiling/platform/platform_ascendc.h"
// Matmul Tiling API (MultiCoreMatmulTiling + optiling::TCubeTiling)
#include "lib/matmul/matmul_tiling.h"
#include "lib/matmul/bmm_tiling.h"
#include "fused_moe_tiling.h"

using namespace AscendC;

// ========== 注册标准 C++ TilingData 结构体 ==========
REGISTER_TILING_DEFAULT(FusedMoeTilingData);

// ========== 分块常量 (基于 910B L1/UB 大小) ==========
namespace {
    constexpr int32_t DEFAULT_TILE_M = 32;
    constexpr int32_t DEFAULT_TILE_K = 64;
    constexpr int32_t DEFAULT_TILE_N = 128;
    constexpr int32_t MAX_EXPERTS = 64;
}

namespace optiling {

// ---------- 辅助: 生成单个 MatMul 的 Cube Tiling ----------
static bool GenerateMatmulTiling(
    matmul_tiling::MultiCoreMatmulTiling& tilingObj,
    optiling::TCubeTiling& outCubeTiling,
    int32_t m, int32_t n, int32_t k,
    bool transB)
{
    // 单核执行每路 MatMul (AIC 负责 Cube)
    tilingObj.SetDim(1);

    // 配置矩阵类型
    // MM A: hidden_states / activated, GM, ND, FP16, 不转置
    // MM B: w1 / w2, GM, ND, FP16, 需要转置 (transB=true)
    // MM C: 输出, GM, ND, FP16
    tilingObj.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingObj.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, transB);
    tilingObj.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);

    // 配置形状
    tilingObj.SetShape(m, n, k);
    tilingObj.SetOrgShape(m, n, k);
    tilingObj.SetSingleShape(m, n, k);

    // 使用 Norm 模板 (默认即可)
    tilingObj.SetMatmulConfigParams(0);  // 0 = Norm template

    // 生成 Tiling 参数
    int64_t ret = tilingObj.GetTiling(outCubeTiling);
    return (ret != -1);
}

// ---------- Tiling 主函数 ----------
void FusedMoEFlagshipTiling(AscendC::TilingContext* context)
{
    // ---- 1. 读取输入形状 ----
    const auto* hiddenShape = context->GetInputShape(0);
    const auto* w1Shape = context->GetInputShape(1);
    // const auto* w2Shape = context->GetInputShape(2);

    int32_t numTokens   = hiddenShape->GetDim(0);
    int32_t hiddenDim   = hiddenShape->GetDim(1);
    int32_t interDim    = w1Shape->GetDim(1);       // w1: [E, N, D]
    int32_t numExperts  = w1Shape->GetDim(0);
    int32_t topK        = context->GetInputShape(4)->GetDim(1);

    // ---- 2. 计算 Tile 大小 ----
    int32_t tileM = std::min(DEFAULT_TILE_M, numTokens);
    int32_t tileK = std::min(DEFAULT_TILE_K, hiddenDim);
    int32_t tileN = std::min(DEFAULT_TILE_N, interDim / 2);

    // ---- 2b. P0: 获取可用 AIC 核心数, 按 expert 数裁剪 ----
    int32_t maxCores = context->GetBlockNum();
    // Python 侧分配的 tempBuffer 限制了最大 core 数
    int32_t numCoresFromAttr = static_cast<int32_t>(
        context->GetAttr("numCores")->GetInt(0));
    int32_t numCores = std::min({maxCores, numExperts, numCoresFromAttr});
    numCores = std::max(numCores, 1);  // 至少 1 核

    // ---- 3. 填充 TilingData 结构体 ----
    FusedMoeTilingData tilingData;
    tilingData.numTokens       = static_cast<uint32_t>(numTokens);
    tilingData.hiddenDim       = static_cast<uint32_t>(hiddenDim);
    tilingData.intermediateDim = static_cast<uint32_t>(interDim);
    tilingData.numExperts      = static_cast<uint32_t>(numExperts);
    tilingData.topK            = static_cast<uint32_t>(topK);
    tilingData.tileM           = static_cast<uint32_t>(tileM);
    tilingData.tileK           = static_cast<uint32_t>(tileK);
    tilingData.tileN           = static_cast<uint32_t>(tileN);
    tilingData.numCores        = static_cast<uint32_t>(numCores);

    // 读取 Expert 分配信息 (由 Python 侧预处理后通过属性传入)
    auto tokensPerExpertAttr = context->GetAttr("tokensPerExpert");
    auto tokenOffsetsAttr    = context->GetAttr("tokenOffsets");
    int32_t expertCount = std::min(numExperts, MAX_EXPERTS);
    for (int32_t i = 0; i < expertCount; i++) {
        tilingData.tokensPerExpert[i] =
            static_cast<uint32_t>(tokensPerExpertAttr->GetInt(i));
        tilingData.tokenOffsets[i] =
            static_cast<uint32_t>(tokenOffsetsAttr->GetInt(i));
    }

    // 读取 numTokensPostPadded (Python 侧 moe_align_block_size 计算)
    tilingData.numTokensPostPadded =
        static_cast<uint32_t>(context->GetAttr("numTokensPostPadded")->GetInt(0));

    // ---- 4. 生成 Cube Tiling 参数 ----
    // 使用平台信息初始化 Tiling API
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // MM1: hidden_states × w1_gate_up^T → gate_up
    //   A: [tileM, tileK]  B: [interDim, tileK](transposed=true)
    //   C: [tileM, interDim]
    {
        matmul_tiling::MultiCoreMatmulTiling mm1Tiling(ascendcPlatform);
        bool ok = GenerateMatmulTiling(
            mm1Tiling,
            reinterpret_cast<optiling::TCubeTiling&>(tilingData.cubeTilingMM1),
            tileM, interDim, tileK,
            true);  // B 矩阵转置
        if (!ok) {
            context->SetTilingKey(-1);  // 标记失败
            return;
        }
    }

    // MM2: activated × w2^T → output
    //   A: [tileM, tileN]  B: [hiddenDim, tileN](transposed=true)
    //   C: [tileM, hiddenDim]
    {
        matmul_tiling::MultiCoreMatmulTiling mm2Tiling(ascendcPlatform);
        bool ok = GenerateMatmulTiling(
            mm2Tiling,
            reinterpret_cast<optiling::TCubeTiling&>(tilingData.cubeTilingMM2),
            tileM, hiddenDim, tileN,
            true);  // B 矩阵转置
        if (!ok) {
            context->SetTilingKey(-1);
            return;
        }
    }

    // ---- 5. 设置 Temp Buffer (Workspace) 大小 ----
    // tempBuffer 存放 gate_up 中间结果: [tileM, interDim] × sizeof(half)
    // P0: 每个 core 使用独立的 tempGM 区域, 总大小 × numCores
    int32_t workspaceSize = numCores * tileM * interDim *
                            static_cast<int32_t>(sizeof(uint16_t));

    // ---- 6. 输出 Tiling 信息 ----
    context->SetTilingData(
        reinterpret_cast<uint8_t*>(&tilingData),
        sizeof(FusedMoeTilingData));
    context->SetBlockDim(static_cast<uint32_t>(numCores));  // P0: 多核分发
    context->SetWorkspaceSize(workspaceSize);
    context->SetTilingKey(1);      // TilingKey = 1 表示正常模式
}

}  // namespace optiling

// ========== Shape 推导函数 ==========
namespace ge {

static void InferShape(const OpDesc& opDesc, const std::vector<DataPtr>& inputs,
                       std::vector<DataPtr>& outputs)
{
    // output shape = hidden_states shape
    const auto& hiddenShape = inputs[0].GetShape();
    outputs[0].SetShape(hiddenShape);
}

static void InferDataType(const OpDesc& opDesc,
                          const std::vector<DataPtr>& inputs,
                          std::vector<DataPtr>& outputs)
{
    // output dtype = hidden_states dtype = FLOAT16
    outputs[0].SetDataType(inputs[0].GetDataType());
}

}  // namespace ge

// ========== 算子原型定义 ==========
namespace ops {

class FusedMoEFlagship : public OpDef {
public:
    explicit FusedMoEFlagship(const char* name) : OpDef(name)
    {
        // ---- 输入 ----
        // 0: hidden_states  [num_tokens, hidden_dim]  FP16
        this->Input("hiddenStates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 1: w1  [num_experts, intermediate_dim, hidden_dim]  FP16
        this->Input("w1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 2: w2  [num_experts, hidden_dim, intermediate_dim/2]  FP16
        this->Input("w2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 3: tempBuffer  [tileM, intermediate_dim]  FP16   (workspace-like)
        this->Input("tempBuffer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // 4: sortedTokenIds  [num_tokens, top_k]  INT32
        this->Input("sortedTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // 5: topkWeights  [num_tokens, top_k]  FP32
        this->Input("topkWeights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // 6: sortedWeights  [numTokensPostPadded]  FP32  (按 expert 排序后的权重)
        this->Input("sortedWeights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // ---- 输出 ----
        // 0: output  [num_tokens, hidden_dim]  FP16
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // ---- 属性 ----
        // Expert 分配信息 (Python 侧 moe_align_block_size 预处理结果)
        this->Attr("tokensPerExpert")
            .AttrType(REQUIRED)
            .ListInt();

        this->Attr("tokenOffsets")
            .AttrType(REQUIRED)
            .ListInt();

        this->Attr("activation")
            .AttrType(OPTIONAL)
            .String("silu");

        // moe_align_block_size padding 后的总 sorted 长度
        this->Attr("numTokensPostPadded")
            .AttrType(REQUIRED)
            .Int();

        // P0: 多核分发使用的 AIC 核心数 (Host 侧基于平台信息设置)
        this->Attr("numCores")
            .AttrType(REQUIRED)
            .Int();

        // ---- 注册推导函数 ----
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        // ---- 注册 Tiling ----
        this->AICore()
            .SetTiling(optiling::FusedMoEFlagshipTiling)
            .AddConfig("ascend910b");
    }
};

}  // namespace ops

// ---- 注册算子 ----
OP_ADD(ops::FusedMoEFlagship);
