# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend implementation.

This backend provides operator implementations for Huawei Ascend NPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    This backend uses Ascend CANN libraries to provide high-performance
    operator implementations for Huawei Ascend NPUs.
    """

    _available: Optional[bool] = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> Optional[str]:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            # Check if NPU device is available
            if torch.npu.is_available() and torch.npu.device_count() > 0:
                AscendBackend._available = True
            else:
                AscendBackend._available = False
        return AscendBackend._available

    # ==================== Operator Implementations ====================
    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_ascend

        return silu_and_mul_ascend(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_ascend

        return rms_norm_ascend(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_ascend

        return rotary_embedding_ascend(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def fused_experts(
        self,
        obj,
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
        """
        Fused MoE expert computation.

        Uses FusedMoEFlagship AscendC custom operator when available (FP16,
        silu, non-quantized), falling back to pure PyTorch per-expert loop.

        Args:
            obj: The calling obj (for interface consistency)
            hidden_states: [num_tokens, hidden_dim] input tensor
            w1: [num_experts, intermediate_dim, hidden_dim] gate+up weights
            w2: [num_experts, hidden_dim, intermediate_dim//2] down weights
            topk_weights: [num_tokens, top_k] routing weights
            topk_ids: [num_tokens, top_k] expert indices

        Returns:
            Output tensor of shape [num_tokens, hidden_dim]
        """
        from .impl.fused_moe import fused_experts_impl

        return fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Ascend NPU.

        This method returns the native Ascend attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Ascend implementation which directly calls
        torch_npu operators without depending on vllm-ascend package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            if use_sparse:
                raise NotImplementedError("MLA with sparse attention is not implemented for Ascend yet.")
            return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendMLABackend"
        return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"
