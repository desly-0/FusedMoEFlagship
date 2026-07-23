# ============================================================
# fused_moe_ascendc.py — FusedMoEFlagship Python 封装
#
# 功能:
#   1. 对 topk_ids 按 expert 分组排序 (moe_align_block_size)
#   2. 组装 Tiling 参数 (不含 CubeTiling, 由 Host 侧填充)
#   3. 分配临时 GM Buffer
#   4. 调用自定义 AscendC 算子
#
# 使用方式 (直接 kernel 调用):
#   output = fused_moe_ascendc(hidden_states, w1, w2,
#                               topk_weights, topk_ids)
#
# 平台: Ascend 910B, CANN 8.5.0
# ============================================================

import math
from typing import Optional, Tuple

import torch


# ========== Tiling 结构体布局 (与 C++ 侧对齐) ==========
# struct FusedMoeTilingData {
#     uint32_t numTokens;             // 0
#     uint32_t hiddenDim;             // 4
#     uint32_t intermediateDim;       // 8
#     uint32_t numExperts;            // 12
#     uint32_t topK;                  // 16
#     uint32_t numTokensPostPadded;   // 20
#     uint32_t tileM;                 // 24
#     uint32_t tileK;                 // 28
#     uint32_t tileN;                 // 32
#     uint32_t numCores;              // 36  (P0: 多核分发)
#     uint32_t tokensPerExpert[64];   // 40
#     uint32_t tokenOffsets[64];      // 40 + 256 = 296
#     uint8_t  cubeTilingMM1[512];   // 296 + 256 = 552
#     uint8_t  cubeTilingMM2[512];   // 552 + 512 = 1064
# };  // total = 1064 + 512 = 1576 bytes
# ========================================================

TILING_DATA_NUM_TOKENS_OFFSET = 0
TILING_DATA_HIDDEN_DIM_OFFSET = 4
TILING_DATA_INTERMEDIATE_DIM_OFFSET = 8
TILING_DATA_NUM_EXPERTS_OFFSET = 12
TILING_DATA_TOPK_OFFSET = 16
TILING_DATA_NUM_TOKENS_POST_PADDED_OFFSET = 20
TILING_DATA_TILE_M_OFFSET = 24
TILING_DATA_TILE_K_OFFSET = 28
TILING_DATA_TILE_N_OFFSET = 32
TILING_DATA_NUM_CORES_OFFSET = 36               # ← NEW
TILING_DATA_TOKENS_PER_EXPERT_OFFSET = 40       # was 36
TILING_DATA_TOKEN_OFFSETS_OFFSET = 296          # was 292
TILING_DATA_CUBE_TILING_MM1_OFFSET = 552        # was 548
TILING_DATA_CUBE_TILING_MM2_OFFSET = 1064       # was 1060
TILING_DATA_TOTAL_SIZE = 1576                   # was 1572

# 分块默认值
DEFAULT_TILE_M = 32
DEFAULT_TILE_K = 64
DEFAULT_TILE_N = 128


# ========== 辅助函数 ==========

def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    topk_weights: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对 topk_ids 按 expert 分组排序, 生成 sorted_token_ids,
    tokens_per_expert, token_offsets 及对应的 sorted_weights.

    参考 vLLM 的 moe_align_block_size 实现。

    Args:
        topk_ids: [num_tokens, top_k] 每个 token 选中的 expert id
        block_size: token 分块大小 (通常 = tileM)
        num_experts: 本地 expert 数
        topk_weights: [num_tokens, top_k] FP32 权重, 用于生成 sorted_weights
        expert_map: [global_num_experts] 全局→本地 expert 映射 (可选)

    Returns:
        sorted_token_ids: [num_tokens * top_k] 按 expert 排序的 token ids
        tokens_per_expert: [num_experts] 每个 expert 的 token 数
        token_offsets: [num_experts] 每个 expert 的起始偏移
        num_tokens_post_padded: padding 后的总 token 数
        expert_ids: [num_sorted_tokens] 每个位置对应的 expert id
        sorted_weights: [num_tokens_post_padded] 按 expert 排序后的权重 (FP32)
    """
    num_tokens, top_k = topk_ids.shape
    total_tokens = num_tokens * top_k

    # 展平 topk_ids
    flat_ids = topk_ids.reshape(-1)  # [num_tokens * top_k]

    # 计数每个 expert 的 token 数
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device='cpu')
    for i in range(total_tokens):
        exp_id = flat_ids[i].item()
        if expert_map is not None:
            exp_id = expert_map[exp_id].item()
        if 0 <= exp_id < num_experts:
            tokens_per_expert[exp_id] += 1

    # 计算偏移，同时将 tokens_per_expert 更新为 block_size 对齐的值
    # Kernel 需要 aligned count 确保循环中每 tile 都是完整的 tileM
    token_offsets = torch.zeros(num_experts, dtype=torch.int32, device='cpu')
    offset = 0
    for i in range(num_experts):
        token_offsets[i] = offset
        # 按 block_size 对齐
        aligned = ((tokens_per_expert[i] + block_size - 1) // block_size) * block_size
        tokens_per_expert[i] = aligned  # ← 更新为 aligned count
        offset += aligned

    num_tokens_post_padded = offset

    # 填充 sorted_token_ids 和 sorted_weights
    sorted_token_ids = torch.full(
        (num_tokens_post_padded,), -1, dtype=torch.int32, device='cpu'
    )
    sorted_weights = torch.zeros(
        num_tokens_post_padded, dtype=torch.float32, device='cpu'
    )
    expert_ids = torch.full(
        (num_tokens_post_padded,), -1, dtype=torch.int32, device='cpu'
    )

    # 为每个 expert 填充 token id 和对应的 weight
    cursor = token_offsets.clone()
    for token_idx in range(num_tokens):
        for k in range(top_k):
            exp_id = flat_ids[token_idx * top_k + k].item()
            if expert_map is not None:
                exp_id = expert_map[exp_id].item()
            if 0 <= exp_id < num_experts:
                pos = cursor[exp_id].item()
                sorted_token_ids[pos] = token_idx
                sorted_weights[pos] = topk_weights[token_idx, k].item()
                expert_ids[pos] = exp_id
                cursor[exp_id] += 1

    return sorted_token_ids, tokens_per_expert, token_offsets, \
        num_tokens_post_padded, expert_ids, sorted_weights


def _pack_tiling_data(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    num_tokens_post_padded: int,
    tokens_per_expert: torch.Tensor,
    token_offsets: torch.Tensor,
    cube_tiling_mm1: Optional[bytes] = None,
    cube_tiling_mm2: Optional[bytes] = None,
    num_cores: int = 1,
) -> bytes:
    """
    将 Tiling 参数打包为 FusedMoeTilingData 二进制结构体。

    cube_tiling_mm1/2 由 Host 侧 Tiling 函数填充。
    num_cores 由 Host 侧 Tiling 函数覆盖 (基于平台信息)。
    Python 侧预设默认值确保布局正确。
    """
    buf = bytearray(TILING_DATA_TOTAL_SIZE)

    def pack_u32(offset: int, value: int):
        buf[offset:offset + 4] = value.to_bytes(4, 'little', signed=False)

    def pack_u32_array(offset: int, arr: torch.Tensor, count: int):
        for i in range(count):
            buf[offset + i * 4: offset + i * 4 + 4] = \
                int(arr[i].item()).to_bytes(4, 'little', signed=False)

    pack_u32(TILING_DATA_NUM_TOKENS_OFFSET, num_tokens)
    pack_u32(TILING_DATA_HIDDEN_DIM_OFFSET, hidden_dim)
    pack_u32(TILING_DATA_INTERMEDIATE_DIM_OFFSET, intermediate_dim)
    pack_u32(TILING_DATA_NUM_EXPERTS_OFFSET, num_experts)
    pack_u32(TILING_DATA_TOPK_OFFSET, top_k)
    pack_u32(TILING_DATA_NUM_TOKENS_POST_PADDED_OFFSET, num_tokens_post_padded)
    pack_u32(TILING_DATA_TILE_M_OFFSET,
             min(DEFAULT_TILE_M, num_tokens))
    pack_u32(TILING_DATA_TILE_K_OFFSET,
             min(DEFAULT_TILE_K, hidden_dim))
    pack_u32(TILING_DATA_TILE_N_OFFSET,
             min(DEFAULT_TILE_N, intermediate_dim // 2))
    pack_u32(TILING_DATA_NUM_CORES_OFFSET, num_cores)      # P0: 多核 (Host 覆盖)

    pack_u32_array(TILING_DATA_TOKENS_PER_EXPERT_OFFSET,
                   tokens_per_expert, num_experts)
    pack_u32_array(TILING_DATA_TOKEN_OFFSETS_OFFSET,
                   token_offsets, num_experts)

    if cube_tiling_mm1 is not None:
        buf[TILING_DATA_CUBE_TILING_MM1_OFFSET:
            TILING_DATA_CUBE_TILING_MM1_OFFSET + 512] = cube_tiling_mm1
    if cube_tiling_mm2 is not None:
        buf[TILING_DATA_CUBE_TILING_MM2_OFFSET:
            TILING_DATA_CUBE_TILING_MM2_OFFSET + 512] = cube_tiling_mm2

    return bytes(buf)


# ========== 主入口函数 ==========

def fused_moe_ascendc(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    expert_map: Optional[torch.Tensor] = None,
    use_custom_op: bool = False,
    num_cores: int = 4,
) -> torch.Tensor:
    """
    使用 FusedMoEFlagship 自定义 AscendC 算子计算 MoE FFN。

    替代原来的逐 expert 循环 + Slice 实现。

    Args:
        hidden_states: [num_tokens, hidden_dim] FP16
        w1: [num_experts, intermediate_dim, hidden_dim] FP16
        w2: [num_experts, hidden_dim, intermediate_dim // 2] FP16
        topk_weights: [num_tokens, top_k] FP32
        topk_ids: [num_tokens, top_k] INT32
        activation: 激活函数类型 (目前仅支持 "silu")
        expert_map: [global_num_experts] INT32 全局→本地映射
        use_custom_op: True=通过 torch.ops 调用, False=返回占位
        num_cores: 多核分发使用的 AIC 核心数 (默认 4, 对应 910B4)

    Returns:
        output: [num_tokens, hidden_dim] FP16
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, intermediate_dim, _ = w1.shape
    top_k = topk_ids.shape[1]

    # Step 1: 对 topk_ids 按 expert 分组排序 (同时生成 sorted_weights)
    sorted_token_ids, tokens_per_expert, token_offsets, \
        num_tokens_post_padded, expert_ids, sorted_weights = _moe_align_block_size(
            topk_ids, DEFAULT_TILE_M, num_experts, topk_weights, expert_map
        )

    # Step 2: 预排列 — 将 hidden_states 按 sorted_token_ids 重排成 permuted_hidden
    #   kernel 内部使用 tokenOffset 作为直接索引, 因此输入需要预先排列为 sorted 顺序
    safe_ids = sorted_token_ids.clamp(min=0).to(
        device=hidden_states.device, non_blocking=True
    )
    permuted_hidden = hidden_states[safe_ids]  # [num_tokens_post_padded, hidden_dim]

    # Step 3: 分配临时 GM Buffer (存放 gate_up 中间结果)
    #   P0: 每个 core 使用独立的 tempGM 区域, 总大小 × num_cores
    tile_m = min(DEFAULT_TILE_M, num_tokens)
    tile_n = min(DEFAULT_TILE_N, intermediate_dim // 2)

    temp_buffer = torch.empty(
        num_cores * tile_m * intermediate_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )

    # Step 4: 分配 permuted 输出 (kernel 写 permuted 顺序)
    permuted_output = torch.zeros(
        num_tokens_post_padded, hidden_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )

    # Step 5: 将 sorted 数据搬到 NPU
    sorted_token_ids_dev = sorted_token_ids.to(
        device=hidden_states.device, non_blocking=True
    )
    sorted_weights_dev = sorted_weights.to(
        device=hidden_states.device, non_blocking=True
    )

    if use_custom_op and hasattr(torch.ops, 'fl_custom'):
        # ---- 通过 torch.ops 调用已注册的自定义算子 ----
        # 需要先编译算子并加载 .so
        # torch.ops.load_library('path/to/libfused_moe_flagship.so')

        # 调用算子 (permuted_hidden/permuted_output 均在 sorted 顺序上操作)
        torch.ops.fl_custom.fused_moe_flagship(
            permuted_hidden,    # input 0: 预排列后的 hidden_states
            w1, w2,
            temp_buffer,
            sorted_token_ids_dev,
            topk_weights,
            sorted_weights_dev,
            permuted_output,    # output: permuted 顺序 (kernel 写)
            tokens_per_expert.tolist(),
            token_offsets.tolist(),
            activation,
            num_tokens_post_padded,
            num_cores,          # P0: 多核核心数
        )
    else:
        # ---- 占位: permuted_output 保持为零 ----
        import warnings
        warnings.warn(
            "FusedMoEFlagship custom op not available. "
            "Use fused_moe_torch_fallback() instead."
        )

    # ---- Step 6: 后散射 — permuted_output → 原始 token 顺序 ----
    #   只筛选有效 (非 padding) 位置, 使用 index_add_ 累加
    mask = sorted_token_ids != -1
    valid_sorted_indices = torch.where(mask)[0]  # CPU int64
    valid_original_tokens = sorted_token_ids[valid_sorted_indices]  # CPU int32
    valid_output = permuted_output[valid_sorted_indices.to(
        device=hidden_states.device)]

    output = torch.zeros(
        num_tokens, hidden_dim,
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    output.index_add_(0,
        valid_original_tokens.to(device=hidden_states.device),
        valid_output)

    return output


def fused_moe_torch_fallback(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """
    PyTorch fallback 实现 (原始的逐 expert 循环)。

    用于精度对比和自定义算子不可用时的回退。
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, intermediate_dim, _ = w1.shape
    top_k = topk_ids.shape[1]

    output = torch.zeros_like(hidden_states)

    for expert_idx in range(num_experts):
        # 找出路由到该 expert 的所有 (token, k) 对
        mask = (topk_ids == expert_idx)  # [num_tokens, top_k]
        token_indices, k_indices = torch.where(mask)
        if token_indices.numel() == 0:
            continue

        # 收集 hidden states
        expert_hidden = hidden_states[token_indices]

        # Gate+Up 投影
        gate_up = torch.mm(expert_hidden, w1[expert_idx].t())
        if activation == "silu":
            gate = torch.nn.functional.silu(gate_up[..., :intermediate_dim // 2])
        else:
            gate = gate_up[..., :intermediate_dim // 2]
        up = gate_up[..., intermediate_dim // 2:]
        activated = gate * up

        # Down 投影
        partial_out = torch.mm(activated, w2[expert_idx].t())

        # 应用 per-token weight (与 kernel Phase 2.5 对齐)
        # 注意: 将 weight 转为 partial_out 的 dtype, 避免 FP16×FP32 提升到 FP32
        weights = topk_weights[token_indices, k_indices].to(dtype=partial_out.dtype)
        partial_out = partial_out * weights.unsqueeze(-1)

        # 累加回 output (同一 token 可能出现在多个 k 中, index_add_ 正确累加)
        output.index_add_(0, token_indices, partial_out)

    return output


def fused_moe_ascendc_with_verify(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_custom_op: bool = False,
) -> torch.Tensor:
    """
    验证模式: 同时运行自定义算子和 fallback, 对比精度。

    Args:
        use_custom_op: True=调用 AscendC 自定义算子, False=仅走 Python 占位路径
    """
    custom_out = fused_moe_ascendc(
        hidden_states, w1, w2, topk_weights, topk_ids,
        use_custom_op=use_custom_op,
    )
    ref_out = fused_moe_torch_fallback(
        hidden_states, w1, w2, topk_weights, topk_ids
    )

    diff = (custom_out - ref_out).abs().max().item()
    rel_diff = (diff / ref_out.abs().max().item())

    print(f"[Verify] max_abs_diff={diff:.6f}, max_rel_diff={rel_diff:.6f}")
    if rel_diff < 0.01:
        print("[Verify] PASS (rel_diff < 1%)")
    else:
        print("[Verify] FAIL (rel_diff >= 1%)")

    return custom_out
