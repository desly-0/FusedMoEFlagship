// ============================================================
// fused_moe_tiling.h — Tiling 结构体定义 (Kernel + Host 共享)
//
// 使用标准 C++ POD 语法定义 Tiling 结构体 (PDF 2.9.2.5.4)
// - TCubeTiling 来自 kernel_tiling/kernel_tiling.h（Kernel 侧命名空间）
// - Host 侧通过 MultiCoreMatmulTiling::GetTiling() 直接填充 TCubeTiling 成员
// - Kernel 侧通过 REGISTER_TILING_DEFAULT + GET_TILING_DATA 注册并解析
//
// 平台: Ascend 910B, CANN 8.5.0
// ============================================================
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct FusedMoeTilingData {
    // ----- 全局形状参数 -----
    uint32_t numTokens;             // 总 token 数
    uint32_t hiddenDim;             // hidden_states 隐层维度 (D)
    uint32_t intermediateDim;       // intermediate 维度 (gate+up 合并后的 N)
    uint32_t numExperts;            // 本地 expert 数 (E)
    uint32_t topK;                  // 每个 token 选中的 expert 数
    uint32_t numTokensPostPadded;   // moe_align_block_size padding 后的总 sorted 长度

    // ----- 分块参数 (Tile 大小) -----
    uint32_t tileM;                 // token 维度分块 (M 轴), 默认 32
    uint32_t tileK;                 // hidden_dim 分块 (K 轴), 默认 64
    uint32_t tileN;                 // intermediate 分块 (N 轴), 默认 128
    uint32_t numCores;              // 多核分发的核心数 (Host 侧读取平台信息后填充)

    // ----- Expert 路由信息 (Python 侧预处理填入) -----
    // tokensPerExpert[E]: 每个 expert 分到的 token 数
    // tokenOffsets[E]:    每个 expert 在 sortedTokenIds 中的起始偏移
    uint32_t tokensPerExpert[64];
    uint32_t tokenOffsets[64];

    // ----- MatMul Tiling 数据 (PDF 2.9.2.5.4) -----
    // Host 侧通过 MultiCoreMatmulTiling::GetTiling() 直接填充此成员。
    // Kernel 侧通过 REGIST_MATMUL_OBJ(&pipe, ws, mmObj, &tiling.cubeTilingMMx) 注册。
    AscendC::tiling::TCubeTiling cubeTilingMM1;
    AscendC::tiling::TCubeTiling cubeTilingMM2;
};

// 编译期检查: 确保 TilingData 结构体大小在合理范围内
static_assert(sizeof(FusedMoeTilingData) <= 4096,
    "FusedMoeTilingData size exceeds 4KB limit");
