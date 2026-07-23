# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu
from flag_gems.runtime.backend._ascend import fused

# ========== FusedMoEFlagship Custom Operator Support ==========
# The FusedMoEFlagship is a fused AscendC custom operator that replaces the
# per-expert Slice+MatMul×2+Activation+Accumulate loop with a single kernel.
#
# Pre-requisites:
#   1. Compile the custom operator (see FusedMoEFlagship project)
#   2. Set FUSED_MOE_FLAGSHIP_LIB env var to the .so path, or
#      place it at the default search path
#
# Build & registration:
#   cd FusedMoEFlagship && cmake -B build && cmake --build build
#   # Produces: build/libfused_moe_flagship.so + kernel_object.o
#   # Copy .so to a known location, e.g. /usr/local/lib/
#
# The operator is registered via CANN's OP_ADD mechanism (C++ side)
# and loaded at runtime via torch.ops.load_library.
# ================================================================

_FLAGSHIP_AVAILABLE = False
_FLAGSHIP_LIB_LOADED = False
_FLAGSHIP_LIB_PATH = os.environ.get(
    "FUSED_MOE_FLAGSHIP_LIB",
    "/usr/local/lib/libfused_moe_flagship.so",
)


def _try_load_flagship_lib():
    global _FLAGSHIP_AVAILABLE, _FLAGSHIP_LIB_LOADED
    if _FLAGSHIP_LIB_LOADED:
        return
    _FLAGSHIP_LIB_LOADED = True
    if not os.path.exists(_FLAGSHIP_LIB_PATH):
        return
    try:
        torch.ops.load_library(_FLAGSHIP_LIB_PATH)
        _FLAGSHIP_AVAILABLE = True
    except Exception as e:
        warnings.warn(
            f"Failed to load FusedMoEFlagship lib ({e}), "
            "falling back to PyTorch implementation."
        )


# ---------- Default tile sizes (910B UB constraint: 256KB) ----------
DEFAULT_TILE_M = 32
DEFAULT_TILE_K = 64
DEFAULT_TILE_N = 128

# ---------- Tiling struct layout (synchronized with C++ FusedMoeTilingData) ----------
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
#     uint32_t numCores;              // 36
#     uint32_t tokensPerExpert[64];   // 40
#     uint32_t tokenOffsets[64];      // 296
#     uint8_t  cubeTilingMM1[512];   // 552
#     uint8_t  cubeTilingMM2[512];   // 1064
# };  // total = 1576 bytes
TILING_DATA_TOTAL_SIZE = 1576
TILING_DATA_TOKENS_PER_EXPERT_OFFSET = 40
TILING_DATA_TOKEN_OFFSETS_OFFSET = 296
TILING_DATA_CUBE_TILING_MM1_OFFSET = 552
TILING_DATA_CUBE_TILING_MM2_OFFSET = 1064


# ========== Internal helpers ==========

def _get_default_num_cores() -> int:
    """Detect available AIC core count on Ascend NPU."""
    try:
        return torch.npu.get_device_property(
            torch.npu.current_device(), "aicCoreCount"
        )
    except Exception:
        return 4  # default for 910B4


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    topk_weights: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Sort topk_ids by expert, align token counts to block_size.

    Returns:
        sorted_token_ids: [num_tokens_post_padded] INT32, sorted by expert
        tokens_per_expert: [num_experts] INT32, aligned to block_size
        token_offsets: [num_experts] INT32, start offset per expert
        num_tokens_post_padded: total length after padding
        expert_ids: [num_tokens_post_padded] INT32, expert per position
        sorted_weights: [num_tokens_post_padded] FP32, weights in sorted order
    """
    num_tokens, top_k = topk_ids.shape
    total_tokens = num_tokens * top_k

    flat_ids = topk_ids.reshape(-1)

    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device="cpu")
    for i in range(total_tokens):
        exp_id = flat_ids[i].item()
        if expert_map is not None:
            exp_id = expert_map[exp_id].item()
        if 0 <= exp_id < num_experts:
            tokens_per_expert[exp_id] += 1

    token_offsets = torch.zeros(num_experts, dtype=torch.int32, device="cpu")
    offset = 0
    for i in range(num_experts):
        token_offsets[i] = offset
        aligned = ((tokens_per_expert[i] + block_size - 1) // block_size) * block_size
        tokens_per_expert[i] = aligned
        offset += aligned

    num_tokens_post_padded = offset

    sorted_token_ids = torch.full(
        (num_tokens_post_padded,), -1, dtype=torch.int32, device="cpu"
    )
    sorted_weights = torch.zeros(
        num_tokens_post_padded, dtype=torch.float32, device="cpu"
    )
    expert_ids = torch.full(
        (num_tokens_post_padded,), -1, dtype=torch.int32, device="cpu"
    )

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

    return (
        sorted_token_ids,
        tokens_per_expert,
        token_offsets,
        num_tokens_post_padded,
        expert_ids,
        sorted_weights,
    )


def _pack_tiling_data(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    num_tokens_post_padded: int,
    tokens_per_expert: torch.Tensor,
    token_offsets: torch.Tensor,
    num_cores: int = 1,
) -> bytes:
    """Pack tiling data into FusedMoeTilingData binary struct (Host side fills cube tiling)."""
    buf = bytearray(TILING_DATA_TOTAL_SIZE)

    def pack_u32(offset: int, value: int):
        buf[offset : offset + 4] = value.to_bytes(4, "little", signed=False)

    def pack_u32_array(offset: int, arr: torch.Tensor, count: int):
        for i in range(count):
            buf[offset + i * 4 : offset + i * 4 + 4] = int(arr[i].item()).to_bytes(
                4, "little", signed=False
            )

    pack_u32(0, num_tokens)
    pack_u32(4, hidden_dim)
    pack_u32(8, intermediate_dim)
    pack_u32(12, num_experts)
    pack_u32(16, top_k)
    pack_u32(20, num_tokens_post_padded)
    pack_u32(24, min(DEFAULT_TILE_M, num_tokens))
    pack_u32(28, min(DEFAULT_TILE_K, hidden_dim))
    pack_u32(32, min(DEFAULT_TILE_N, intermediate_dim // 2))
    pack_u32(36, num_cores)

    pack_u32_array(TILING_DATA_TOKENS_PER_EXPERT_OFFSET, tokens_per_expert, num_experts)
    pack_u32_array(TILING_DATA_TOKEN_OFFSETS_OFFSET, token_offsets, num_experts)

    return bytes(buf)


def _fused_moe_flagship_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    num_cores: Optional[int] = None,
) -> torch.Tensor:
    """FusedMoEFlagship custom AscendC operator path.

    Pre-permute hidden_states by sorted_token_ids → call custom op →
    post-scatter via index_add_ with -1 sentinel masking.

    Falls back to _torch_fused_experts_impl if the operator library
    is not available.
    """
    _try_load_flagship_lib()

    if not _FLAGSHIP_AVAILABLE:
        return _torch_fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    num_tokens, hidden_dim = hidden_states.shape
    num_experts, intermediate_dim, _ = w1.shape
    top_k = topk_ids.shape[1]
    num_cores = num_cores or _get_default_num_cores()
    tile_m = min(DEFAULT_TILE_M, num_tokens)

    # Step 1: Sort tokens by expert
    sorted_token_ids, tokens_per_expert, token_offsets, \
        num_tokens_post_padded, _expert_ids, sorted_weights = _moe_align_block_size(
            topk_ids, DEFAULT_TILE_M, num_experts, topk_weights, expert_map
        )

    # Step 2: Pre-permute hidden_states into sorted order
    safe_ids = sorted_token_ids.clamp(min=0).to(
        device=hidden_states.device, non_blocking=True
    )
    permuted_hidden = hidden_states[safe_ids]

    # Step 3: Allocate temp buffer (gate_up intermediate) + permuted output
    temp_buffer = torch.empty(
        num_cores * tile_m * intermediate_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    permuted_output = torch.zeros(
        num_tokens_post_padded, hidden_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # Step 4: Move metadata to NPU
    sorted_token_ids_dev = sorted_token_ids.to(
        device=hidden_states.device, non_blocking=True
    )
    sorted_weights_dev = sorted_weights.to(
        device=hidden_states.device, non_blocking=True
    )

    # Step 5: Call the custom operator
    torch.ops.fl_custom.fused_moe_flagship(
        permuted_hidden,
        w1, w2,
        temp_buffer,
        sorted_token_ids_dev,
        topk_weights,
        sorted_weights_dev,
        permuted_output,
        tokens_per_expert.tolist(),
        token_offsets.tolist(),
        activation,
        num_tokens_post_padded,
        num_cores,
    )

    # Step 6: Post-scatter — index_add_ back to original token order
    mask = sorted_token_ids != -1
    valid_sorted_indices = torch.where(mask)[0]
    valid_original_tokens = sorted_token_ids[valid_sorted_indices]
    valid_output = permuted_output[valid_sorted_indices.to(
        device=hidden_states.device
    )]

    output = torch.zeros(
        num_tokens, hidden_dim,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    output.index_add_(
        0,
        valid_original_tokens.to(device=hidden_states.device),
        valid_output,
    )

    return output


# ========== Existing PyTorch fallback implementation ==========


def _torch_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch implementation of fused MoE experts for NPU.

    This avoids the Triton fused_moe_kernel which has compatibility issues
    on Ascend NPU hardware.
    """
    num_tokens, hidden_dim = hidden_states.size()
    E, N, _ = w1.size()  # w1: [E, N, K_in]
    K = w2.size(1)        # w2: [E, K_out, N//2]
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.zeros_like(hidden_states)

    # Map global expert ids to local expert ids
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids.long()]
    else:
        local_topk_ids = topk_ids.long()

    # Process each expert
    for expert_idx in range(E):
        # Find which (token, k) pairs are assigned to this expert
        mask = (local_topk_ids == expert_idx)  # [num_tokens, top_k]
        if not mask.any():
            continue

        # Get token indices and their k-slot indices
        token_indices, k_indices = torch.where(mask)

        # Gather the hidden states for these tokens
        expert_input = hidden_states[token_indices]  # [n, hidden_dim]

        # Apply router weight on input if needed
        if apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_input = expert_input * weights.to(expert_input.dtype)

        # First matmul: expert_input @ w1[expert_idx].T
        # w1[expert_idx] shape: [N, hidden_dim], result: [n, N]
        gate_up = torch.mm(expert_input, w1[expert_idx].t())

        # Activation (pure PyTorch to avoid Triton kernel issues on NPU)
        if activation == "silu":
            d = gate_up.shape[-1] // 2
            gate_up = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        elif activation == "gelu":
            gate_up = torch_npu.npu_gelu_mul(gate_up)
        elif activation == "silu_no_mul":
            gate_up = F.silu(gate_up)
        elif activation == "gelu_no_mul":
            gate_up = torch_npu.npu_gelu(gate_up)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

        # Second matmul: activated @ w2[expert_idx].T
        # w2[expert_idx] shape: [K_out, N//2], result: [n, K_out]
        expert_output = torch.mm(gate_up, w2[expert_idx].t())

        # Apply router weight on output if not applied on input
        if not apply_router_weight_on_input:
            weights = topk_weights[token_indices, k_indices].unsqueeze(-1)
            expert_output = expert_output * weights.to(expert_output.dtype)

        # Accumulate results
        out_hidden_states.index_add_(0, token_indices, expert_output)

    return out_hidden_states


# ========== Entry point ==========


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # Try FusedMoEFlagship custom op for the standard MoE case:
    #   FP16 weights, silu activation, apply_router_weight_on_input=False,
    #   no quantization, not inplace.
    can_use_flagship = (
        not use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and not apply_router_weight_on_input
        and not inplace
        and activation == "silu"
        and hidden_states.dtype == torch.float16
        and w1.dtype == torch.float16
        and w2.dtype == torch.float16
    )

    if can_use_flagship:
        return _fused_moe_flagship_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    # Fallback: pure PyTorch per-expert loop
    return _torch_fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
