// ============================================================
// fused_moe_flagship.cpp — AscendC Kernel 侧核函数实现
//
// 融合 MoE FFN 计算链路 (Slice + MatMul ×2 + Activation + Accumulate)
// 到一个算子，消除所有中间 Slice 调用（用指针偏移替代数据拷贝）。
//
// 计算策略:
//   REGIST_MATMUL_OBJ × 2 个 MatMul 对象
//   MatMul 1: hidden_states × w1_gate_up^T → gate_up (中间 GM Buffer)
//   Vector:   silu(gate) × up → activated (Local Memory)
//   MatMul 2: activated × w2^T → output (累加到 GM)
//
// 平台: Ascend 910B, CANN 8.5.0
// ============================================================

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_moe_tiling.h"

using namespace AscendC;

// ---------- Kernel 模板类 ----------
template <typename T>
class FusedMoEFlagshipKernel {
public:
    __aicore__ inline FusedMoEFlagshipKernel() {}

    __aicore__ inline void Init(
        GM_ADDR hiddenStates,
        GM_ADDR w1,
        GM_ADDR w2,
        GM_ADDR tempBuffer,       // 临时 GM Buffer (存放 gate_up 中间结果)
        GM_ADDR sortedTokenIds,
        GM_ADDR topkWeights,
        GM_ADDR sortedWeights,    // 按 expert 排序后的 weights [numTokensPostPadded]
        GM_ADDR output,
        const FusedMoeTilingData& tiling)
    {
        tiling_ = tiling;

        uint32_t numTokens = tiling.numTokens;
        uint32_t hiddenDim = tiling.hiddenDim;
        uint32_t interDim  = tiling.intermediateDim;  // N = gate+up 合并

        hiddenGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(hiddenStates),
            numTokens * hiddenDim);

        w1GM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(w1),
            tiling.numExperts * interDim * hiddenDim);

        w2GM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(w2),
            tiling.numExperts * hiddenDim * (interDim / 2));

        // 存储 tempBuffer GM 基地址, Process 中加上 per-core 偏移
        tempBufferBase_ = reinterpret_cast<__gm__ T*>(tempBuffer);
        tempGM_.SetGlobalBuffer(tempBufferBase_,
                                tiling.tileM * interDim);

        sortedIdsGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(sortedTokenIds),
            numTokens * tiling.topK);

        weightsGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(topkWeights),
            numTokens * tiling.topK);

        // sortedWeights: 每个 sorted 位置对应的 topk_weight
        // 长度 = numTokensPostPadded (由 moe_align_block_size 决定)
        // 从 tiling 中读取 numTokensPostPadded
        sortedWeightsGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(sortedWeights),
            tiling_.numTokensPostPadded);

        outputGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output),
            numTokens * hiddenDim);

        // ==== 注册 Matmul 对象 (必须在 pipe.InitBuffer 之前) ====
        // API 约束: 分离模式中 REGIST_MATMUL_OBJ 必须在 InitBuffer 前调用
        // 详见: 5.2.1.14 REGIST_MATMUL_OBJ, 5.2.1.15 Init
        TCubeTiling* cubeTilingMM1 =
            reinterpret_cast<TCubeTiling*>(&(tiling_.cubeTilingMM1));
        TCubeTiling* cubeTilingMM2 =
            reinterpret_cast<TCubeTiling*>(&(tiling_.cubeTilingMM2));
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(),
                          mm1_, cubeTilingMM1,
                          mm2_, cubeTilingMM2);
        // 不再需要 mm.Init() — REGIST_MATMUL_OBJ 已内部调用

        // 分配 Local Memory (UB) 缓冲区 (P0+P3+P4 优化后布局)
        //   mergeBuf:      tileM × interDim × sizeof(T)   (合并 DataCopy: gate_up 整体 → 视图切片)
        //   weightLocal:   tileM × sizeof(float)           (F32 权重加载)
        //   weightTmp:     tileM × sizeof(T)               (F32→T 转换缓冲)
        //   weightBCast:   tileM × tileN × sizeof(T)       (Broadcast 展开, 用于批量 Mul)
        // 总 UB 使用: 32×256×2 + 32×4 + 32×2 + 32×128×2 = 24.2KB < 256KB ✓
        uint32_t tileM = tiling.tileM;
        uint32_t tileN = tiling.tileN;
        uint32_t interDim = tiling.intermediateDim;

        pipe.InitBuffer(mergeBuf_, tileM * interDim * sizeof(T));
        pipe.InitBuffer(weightLocalBuf_, tileM * sizeof(float));
        pipe.InitBuffer(weightTmpBuf_, tileM * sizeof(T));
        pipe.InitBuffer(weightBCastBuf_, tileM * tileN * sizeof(T));
    }

    __aicore__ inline void Process() {
        uint32_t coreId = GetBlockIdx();
        uint32_t numCores = tiling_.numCores;
        uint32_t numExperts = tiling_.numExperts;
        uint32_t tileM = tiling_.tileM;
        uint32_t tileK = tiling_.tileK;
        uint32_t tileN = tiling_.tileN;
        uint32_t hiddenDim = tiling_.hiddenDim;
        uint32_t interDim  = tiling_.intermediateDim;  // = 2 * tileN

        // P0: per-core tempBuffer 偏移 (各 core 使用独立的 tempGM 区域)
        tempGM_.SetGlobalBuffer(
            tempBufferBase_ + coreId * tileM * interDim,
            tileM * interDim);

        // ---- 外层循环 (P0: Round-Robin 多核分发) ----
        // 每个 core 处理 expIdx ≡ coreId (mod numCores) 的 expert
        for (uint32_t expIdx = coreId; expIdx < numExperts; expIdx += numCores) {
            uint32_t numExpTokens = tiling_.tokensPerExpert[expIdx];
            if (numExpTokens == 0) continue;

            uint32_t tokenOffset = tiling_.tokenOffsets[expIdx];

            // Expert 权重在 GM 上的偏移 (元素单位)
            uint32_t w1Offset = expIdx * interDim * hiddenDim;
            uint32_t w2Offset = expIdx * hiddenDim * (interDim / 2);

            // ---- 中层循环: 按 tileM 切分该 Expert 的 Token ----
            for (uint32_t t = 0; t < numExpTokens; t += tileM) {
                uint32_t curTileM =
                    (t + tileM <= numExpTokens) ? tileM : (numExpTokens - t);

                // ==========================================================
                // Phase 1: MatMul — Gate+Up 合并投影
                //   hidden_states[tileStart, :tileK]  ×
                //   w1[expIdx, :interDim, :tileK]^T
                //   → gate_up[curTileM, interDim] → tempGM
                // ==========================================================
                mm1_.SetTensorA(
                    hiddenGM_[tokenOffset * hiddenDim + t * tileK], false);
                mm1_.SetTensorB(w1GM_[w1Offset], true);
                mm1_.IterateAll(tempGM_);

                // ==========================================================
                // Phase 2 (P4: 单次 DataCopy + GetWithOffset 视图)
                //   gate_up  = tempGM[curTileM, interDim]
                //   前半段 (0..curTileM×tileN-1): gate 部分
                //   后半段 (curTileM×tileN .. curTileM×2×tileN-1): up 部分
                //   activated = silu(gate) × up  [curTileM, tileN]
                //
                // P4 优化: 单次 DataCopy 加载整个 gate_up,
                //          GetWithOffset 视图切片, 无需 2 次独立 DataCopy。
                // ==========================================================
                LocalTensor<T> mergeBuf = mergeBuf_.Get<T>(curTileM * interDim);
                DataCopy(mergeBuf, tempGM_, curTileM * interDim);

                LocalTensor<T> gateView = mergeBuf_.GetWithOffset<T>(
                    curTileM * tileN, 0);
                LocalTensor<T> upView = mergeBuf_.GetWithOffset<T>(
                    curTileM * tileN, curTileM * tileN * sizeof(T));

                Silu(gateView, gateView, curTileM * tileN);
                Mul(gateView, gateView, upView, curTileM * tileN);

                // ==========================================================
                // Phase 2.5 (P3: Broadcast + Mul 批量权重缩放)
                //   activated[i, :] *= sortedWeights[tokenOffset + t + i]
                //
                // P3 优化: 用 Broadcast 将权重向量展开为矩阵,
                //         再用单次 Mul 替代 N 次 Muls 行循环。
                // ==========================================================
                {
                    LocalTensor<float> weightF32 =
                        weightLocalBuf_.Get<float>(curTileM);
                    DataCopy(weightF32,
                             sortedWeightsGM_[tokenOffset + t], curTileM);

                    // F32 → T 转换后用于 Broadcast
                    LocalTensor<T> weightT = weightTmpBuf_.Get<T>(curTileM);
                    for (uint32_t i = 0; i < curTileM; i++) {
                        weightT(i) = static_cast<T>(weightF32(i));
                    }

                    // Broadcast: [curTileM, 1] → [curTileM, tileN]
                    // 沿 axis=1 扩展, 每个 weight[i] 重复 tileN 次
                    // 官方 API (5.10.9): Broadcast<T, dim, axis>(dst, src, dstShape, srcShape)
                    LocalTensor<T> weightBCast =
                        weightBCastBuf_.Get<T>(curTileM * tileN);
                    uint32_t bDstShape[2] = {curTileM, tileN};
                    uint32_t bSrcShape[2] = {curTileM, 1};
                    Broadcast<half, 2, 1>(weightBCast, weightT,
                                          bDstShape, bSrcShape);

                    Mul(gateView, gateView, weightBCast, curTileM * tileN);
                }

                // ==========================================================
                // Phase 3: MatMul — Down 投影
                //   activated [curTileM, tileN] × w2[expIdx] [tileK, tileN]^T
                //   → output [curTileM, tileK]
                //
                //   gateView 中的数据是 activated 结果 →
                //   DataCopy to tempGM (MM2 的 GM 输入)
                // ==========================================================
                DataCopy(tempGM_, gateView, curTileM * tileN);

                mm2_.SetTensorA(tempGM_, false);
                mm2_.SetTensorB(w2GM_[w2Offset], true);
                mm2_.IterateAll(outputGM_[(tokenOffset + t) * hiddenDim]);
            }
        }

        mm1_.End();
        mm2_.End();
    }

private:
    GlobalTensor<T>        hiddenGM_;
    GlobalTensor<T>        w1GM_;
    GlobalTensor<T>        w2GM_;
    GlobalTensor<T>        tempGM_;
    GlobalTensor<int32_t>  sortedIdsGM_;
    GlobalTensor<float>    weightsGM_;
    GlobalTensor<float>    sortedWeightsGM_;  // [numTokensPostPadded], 按 expert 排序后的权重
    GlobalTensor<T>        outputGM_;

    FusedMoeTilingData tiling_;

    __gm__ T* tempBufferBase_;   // tempBuffer GM 基地址 (用于 Process 中计算 per-core 偏移)

    TPipe pipe;
    TBuf<TPosition::VECCALC> mergeBuf_;             // gate_up [tileM, interDim] (P4: 单次 DataCopy + 视图)
    TBuf<TPosition::VECCALC> weightLocalBuf_;       // weight [tileM] (F32 per-token weight, 加载用)
    TBuf<TPosition::VECCALC> weightTmpBuf_;         // weight [tileM] (T 类型, 转换后)
    TBuf<TPosition::VECCALC> weightBCastBuf_;       // weight [tileM, tileN] (T 类型, Broadcast 展开)

    // MatMul 对象:
    //   A: GM, ND, half  | B: GM, ND, half (transposed) | C: GM, ND, half
    //   transposed 通过 SetTensorB(bool) 参数控制, 不在 MatmulType 中
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, T>,
           MatmulType<TPosition::GM, CubeFormat::ND, T>,
           MatmulType<TPosition::GM, CubeFormat::ND, T>> mm1_;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, T>,
           MatmulType<TPosition::GM, CubeFormat::ND, T>,
           MatmulType<TPosition::GM, CubeFormat::ND, T>> mm2_;
};

// ---------- 核函数入口 ----------
extern "C" __global__ __aicore__ void fused_moe_flagship(
    GM_ADDR hiddenStates,
    GM_ADDR w1,
    GM_ADDR w2,
    GM_ADDR tempBuffer,
    GM_ADDR sortedTokenIds,
    GM_ADDR topkWeights,
    GM_ADDR sortedWeights,       // [numTokensPostPadded] F32 权重
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // MIX 模式: AIC 处理 MatMul, AIV 处理 Vector 运算
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    FusedMoEFlagshipKernel<half> kernel;
    kernel.Init(hiddenStates, w1, w2, tempBuffer,
                sortedTokenIds, topkWeights, sortedWeights, output, tilingData);
    kernel.Process();
}
