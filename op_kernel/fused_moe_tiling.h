// ============================================================
// fused_moe_tiling.h — Tiling 结构体定义 (Kernel + Host 共享)
//
// Host 侧填充所有参数, 通过 tiling_data GM 地址传给核函数。
// cubeTilingMM1/2 由 Host 侧 MultiCoreMatmulTiling::GetTiling()
// 填充为 TCubeTiling 二进制数据，Kernel 侧通过
// reinterpret_cast<TCubeTiling*> 访问。
//
// 平台: Ascend 910B, CANN 8.5.0
// ============================================================
#pragma once

#include <cstdint>

// Host 侧 Tiling 函数填充此结构体，通过 REGISTER_TILING_DEFAULT 注册。
// Kernel 侧通过 GET_TILING_DATA 或 GetTilingData<T>() 获取。
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

    // ----- MatMul Tiling 数据 (TCubeTiling 二进制) -----
    // Host 侧通过 MultiCoreMatmulTiling::GetTiling() 填充。
    // Kernel 侧通过 REGIST_MATMUL_OBJ 注册，转换回 TCubeTiling*。
    // 512 字节足够容纳 TCubeTiling (实际约 200-300 字节)。
    uint8_t cubeTilingMM1[512];
    uint8_t cubeTilingMM2[512];
};

// 编译期检查: 确保 TilingData 结构体大小在合理范围内
static_assert(sizeof(FusedMoeTilingData) <= 4096,
    "FusedMoeTilingData size exceeds 4KB limit");
