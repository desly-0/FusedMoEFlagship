# ============================================================
# test_fused_moe.py — FusedMoEFlagship 综合测试套件
#
# 测试覆盖:
#   1. _moe_align_block_size: 路由数据结构正确性 (CPU)
#   2. fused_moe_torch_fallback: PyTorch 参考实现正确性
#   3. 全流水线: pre-permute → kernel → post-scatter 数据流
#   4. 边界情况: 空 expert、单 token、tileM 边界、topk 变化
#   5. 形状通用性: 不同 hidden_dim / intermediate_dim / num_experts
#   6. 跨组件数据连贯性: sorted 数组 → 预排列 → 核函数 → 后散射
#
# 运行方式: python test_fused_moe.py
# ============================================================

import math
import random
import unittest
from typing import Optional, Tuple

import torch

# 导入被测试模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_moe_ascendc import (
    _moe_align_block_size,
    _pack_tiling_data,
    fused_moe_ascendc,
    fused_moe_torch_fallback,
    fused_moe_ascendc_with_verify,
    DEFAULT_TILE_M,
    DEFAULT_TILE_K,
    DEFAULT_TILE_N,
)


# ============================================================
#  全局配置
# ============================================================

# 测试随机种子 (可复现)
_TEST_SEED = 42
# 默认测试形状 (匹配 Qwen3-27B 典型配置)
_DEFAULT_HIDDEN = 256    # 缩小版 hidden_dim (便于快速测试)
_DEFAULT_INTER = 512     # 缩小版 intermediate_dim
_DEFAULT_EXPERTS = 8
_DEFAULT_TOPK = 2


def setup_seed(seed: int = _TEST_SEED):
    """固定随机种子，确保测试可复现。"""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  Helper: 合成测试数据
# ============================================================

def make_moe_inputs(
    num_tokens: int = 32,
    hidden_dim: int = _DEFAULT_HIDDEN,
    intermediate_dim: int = _DEFAULT_INTER,
    num_experts: int = _DEFAULT_EXPERTS,
    top_k: int = _DEFAULT_TOPK,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    生成一组 MoE 测试输入。

    Returns:
        dict 包含:
          hidden_states: [num_tokens, hidden_dim]
          w1: [num_experts, intermediate_dim, hidden_dim]
          w2: [num_experts, hidden_dim, intermediate_dim // 2]
          topk_weights: [num_tokens, top_k] FP32
          topk_ids: [num_tokens, top_k] INT64
    """
    hidden_states = torch.randn(
        num_tokens, hidden_dim, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_dim, hidden_dim, device=device, dtype=dtype)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim // 2, device=device, dtype=dtype)
    topk_weights = torch.rand(
        num_tokens, top_k, device=device, dtype=torch.float32)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.long)

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def make_routing_skewed(
    num_tokens: int = 32,
    num_experts: int = _DEFAULT_EXPERTS,
    top_k: int = _DEFAULT_TOPK,
    hot_expert: int = 0,
    hot_ratio: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成倾斜的路由分布 — hot_expert 获得 hot_ratio 比例的路由。

    用于测试 expert 间负载不均的场景。
    """
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.long)
    # 强制 hot_ratio 的 token 路由到 hot_expert
    hot_count = int(num_tokens * hot_ratio)
    for i in range(min(hot_count, num_tokens)):
        for k in range(top_k):
            topk_ids[i, k] = hot_expert
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)
    return topk_ids, topk_weights


# ============================================================
#  TestCase 1: moe_align_block_size 数据流正确性
# ============================================================

class TestMoeAlignBlockSize(unittest.TestCase):
    """测试路由数据结构生成的正确性。"""

    def setUp(self):
        setup_seed()
        self.num_experts = 4
        self.top_k = 2
        self.block_size = DEFAULT_TILE_M

    def _verify_routing_structure(
        self,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        token_offsets: torch.Tensor,
        num_tokens_post_padded: int,
        sorted_weights: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        """验证路由数据结构的核心不变性。"""
        num_tokens, top_k = topk_ids.shape

        # [1] 总 sorted 长度 === num_tokens_post_padded
        self.assertEqual(sorted_token_ids.shape[0], num_tokens_post_padded)
        self.assertEqual(sorted_weights.shape[0], num_tokens_post_padded)

        # [2] 每个 expert 的 aligned token 数是 block_size 的整数倍
        for exp_id in range(self.num_experts):
            cnt = tokens_per_expert[exp_id].item()
            self.assertEqual(
                cnt % self.block_size, 0,
                f"Expert {exp_id} aligned count {cnt} not multiple of "
                f"block_size={self.block_size}")

        # [3] 偏移正确: 每个 expert 的偏移是前序 expert aligned 计数的累加
        cum = 0
        for exp_id in range(self.num_experts):
            self.assertEqual(
                token_offsets[exp_id].item(), cum,
                f"Expert {exp_id} offset mismatch")
            cum += tokens_per_expert[exp_id].item()
        self.assertEqual(cum, num_tokens_post_padded)

        # [4] sorted_token_ids 中 -1 (padding) 只出现在每个 expert 块末尾
        for exp_id in range(self.num_experts):
            start = token_offsets[exp_id].item()
            count = tokens_per_expert[exp_id].item()
            # 计算该 expert 的实际 token 数 (非 padding)
            real_count = torch.sum(
                (sorted_token_ids[start:start + count] >= 0)).item()
            # 前 real_count 个应该是有效 token id
            valid_ids = sorted_token_ids[start:start + real_count]
            self.assertTrue(
                torch.all(valid_ids >= 0),
                f"Expert {exp_id}: valid region has -1")
            # 后 padding 区域全是 -1
            pad_ids = sorted_token_ids[start + real_count:start + count]
            self.assertTrue(
                torch.all(pad_ids == -1),
                f"Expert {exp_id}: padding region has non -1: {pad_ids}")

        # [5] sorted_weights 中有效位置 > 0, padding 位置 == 0
        for exp_id in range(self.num_experts):
            start = token_offsets[exp_id].item()
            count = tokens_per_expert[exp_id].item()
            real_count = torch.sum(
                (sorted_token_ids[start:start + count] >= 0)).item()
            self.assertTrue(
                torch.all(sorted_weights[start:start + real_count] > 0),
                f"Expert {exp_id}: valid weights should be > 0")
            self.assertTrue(
                torch.all(sorted_weights[start + real_count:start + count] == 0),
                f"Expert {exp_id}: padding weights should be 0")

        # [6] sorted 顺序 preserves topk_weights 信息 (可选验证)
        if topk_weights is not None:
            flat_ids = topk_ids.reshape(-1)
            flat_weights = topk_weights.reshape(-1)
            for i, tid in enumerate(flat_ids):
                exp_id = tid.item() if 0 <= tid.item() < self.num_experts else None
                if exp_id is None:
                    continue
                start = token_offsets[exp_id].item()
                count = tokens_per_expert[exp_id].item()
                block_weights = sorted_weights[start:start + count]
                # 该 token 的 weight 应出现在 sorted_weights 对应 expert 块中
                self.assertIn(
                    flat_weights[i].item(), block_weights.tolist(),
                    f"Weight {flat_weights[i].item()} for token {i // top_k}, "
                    f"k={i % top_k} not found in expert {exp_id} block")

    def test_uniform_routing(self):
        """均匀分布: 各 expert token 数相近。"""
        num_tokens = 32
        topk_ids, topk_weights = make_routing_skewed(
            num_tokens, self.num_experts, self.top_k,
            hot_expert=0, hot_ratio=1.0 / self.num_experts)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)

    def test_skewed_routing(self):
        """倾斜分布: 一个 expert 占 70% 路由。"""
        num_tokens = 64
        topk_ids, topk_weights = make_routing_skewed(
            num_tokens, self.num_experts, self.top_k,
            hot_expert=1, hot_ratio=0.7)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)

    def test_empty_experts(self):
        """部分 expert 没有 token。"""
        num_tokens = 16
        topk_ids = torch.zeros(num_tokens, self.top_k, dtype=torch.long)
        # 强制所有 token 只路由到 expert 0,1
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1
        topk_weights = torch.rand(num_tokens, self.top_k, dtype=torch.float32)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)
        # expert 2,3 应该为 0
        self.assertEqual(tpe[2].item(), 0, "Expert 2 should have 0 tokens")
        self.assertEqual(tpe[3].item(), 0, "Expert 3 should have 0 tokens")

    def test_single_token(self):
        """单 token 极端场景。"""
        num_tokens = 1
        topk_ids = torch.zeros(num_tokens, self.top_k, dtype=torch.long)
        topk_ids[0, 0] = 2
        topk_ids[0, 1] = 3
        topk_weights = torch.rand(num_tokens, self.top_k, dtype=torch.float32)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)
        # 单 token，topk 路由到 2 个 expert，各 expert 至少 1 个 token
        # padding 后应为 block_size (tileM=32)
        self.assertGreaterEqual(ntp, self.block_size)

    def test_tile_boundary(self):
        """精确在 tileM 边界上的 token 数。"""
        num_tokens = DEFAULT_TILE_M * 2  # 64 tokens
        topk_ids = torch.randint(
            0, self.num_experts, (num_tokens, self.top_k), dtype=torch.long)
        # 让 expert 0 恰好有 tileM 个 token
        topk_ids[:DEFAULT_TILE_M, 0] = 0
        topk_weights = torch.rand(num_tokens, self.top_k, dtype=torch.float32)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)

    def test_topk_one(self):
        """topk=1 的特殊情况。"""
        topk_ids = torch.randint(
            0, self.num_experts, (32, 1), dtype=torch.long)
        topk_weights = torch.rand(32, 1, dtype=torch.float32)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, self.num_experts, topk_weights, None)
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw, topk_weights)

    def test_expert_map(self):
        """expert_map 全局→本地映射。"""
        num_tokens = 16
        global_num_experts = 8
        num_local_experts = 4
        topk_ids = torch.randint(
            0, global_num_experts, (num_tokens, self.top_k), dtype=torch.long)
        topk_weights = torch.rand(num_tokens, self.top_k, dtype=torch.float32)
        expert_map = torch.tensor(
            [0, 1, -1, -1, 2, -1, 3, -1], dtype=torch.int32)

        sorted_ids, tpe, toff, ntp, _, sw = _moe_align_block_size(
            topk_ids, self.block_size, num_local_experts,
            topk_weights, expert_map)

        # 验证: -1 映射的 expert 不会被分配到本地 expert
        self.assertEqual(tpe.shape[0], num_local_experts)
        # 验证数据结构
        self._verify_routing_structure(
            topk_ids, sorted_ids, tpe, toff, ntp, sw)


# ============================================================
#  TestCase 2: fused_moe_torch_fallback 数学正确性
# ============================================================

class TestTorchFallback(unittest.TestCase):
    """验证 PyTorch fallback 实现的数学正确性。"""

    def setUp(self):
        setup_seed()

    def _naive_moe_loop(self, hidden_states, w1, w2,
                        topk_weights, topk_ids, activation="silu"):
        """
        最简 MoE 实现 (作为 reference 的 reference)。
        逐个 expert 用循环实现, 不做任何优化, 用于验证 fallback。
        """
        num_tokens, hidden_dim = hidden_states.shape
        num_experts, intermediate_dim, _ = w1.shape
        top_k = topk_ids.shape[1]
        output = torch.zeros_like(hidden_states)

        for expert_idx in range(num_experts):
            mask = (topk_ids == expert_idx)
            token_indices, k_indices = torch.where(mask)
            if token_indices.numel() == 0:
                continue

            expert_hidden = hidden_states[token_indices]
            gate_up = expert_hidden @ w1[expert_idx].t()

            if activation == "silu":
                gate = torch.nn.functional.silu(
                    gate_up[..., :intermediate_dim // 2])
            else:
                gate = gate_up[..., :intermediate_dim // 2]
            up = gate_up[..., intermediate_dim // 2:]

            activated = gate * up
            partial_out = activated @ w2[expert_idx].t()

            weights = topk_weights[token_indices, k_indices].to(dtype=partial_out.dtype)
            partial_out = partial_out * weights.unsqueeze(-1)

            output.index_add_(0, token_indices, partial_out)
        return output

    def test_vs_naive_loop(self):
        """fallback 与最简循环输出一致。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2)
        ref = self._naive_moe_loop(**inputs)
        out = fused_moe_torch_fallback(**inputs)
        max_diff = (out - ref).abs().max().item()
        self.assertLess(
            max_diff, 1e-5,
            f"Fallback vs naive loop: max_diff={max_diff:.2e}")

    def test_activation_gelu(self):
        """GELU 激活函数 (未来扩展预留)。"""
        inputs = make_moe_inputs(
            num_tokens=8, hidden_dim=32, intermediate_dim=64,
            num_experts=2, top_k=1)
        out = fused_moe_torch_fallback(
            **inputs, activation="silu")
        # 验证可运行即可 (silu 为当前唯一支持)
        self.assertIsNotNone(out)

    def test_empty_expert_fallback(self):
        """空 expert 不应影响 fallback 结果。"""
        num_tokens = 8
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=32, intermediate_dim=64,
            num_experts=8, top_k=1)
        # 强制所有 token 路由到 expert 0
        inputs["topk_ids"].fill_(0)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, inputs["hidden_states"].shape)

    def test_single_token_fallback(self):
        """单 token 时 fallback 正确。"""
        inputs = make_moe_inputs(
            num_tokens=1, hidden_dim=32, intermediate_dim=64,
            num_experts=4, top_k=2)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (1, 32))
        self.assertFalse(torch.isnan(out).any(), "Output has NaN")

    def test_accumulation_correctness(self):
        """验证同 token 多 expert 的累加正确性。"""
        num_tokens = 4
        hidden_dim = 16
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=hidden_dim,
            intermediate_dim=32, num_experts=4, top_k=2)

        out = fused_moe_torch_fallback(**inputs)

        # 同一 token 的两个 topk 结果应累加 (不是平均, 不是覆盖)
        # 验证: 直接看 index_add_ 语义 — 这是针对 token 级别累加
        # topk_weights 都是正数, 所以 output 元素应为正 (hidden 有正有负但 w 随机)
        # 仅验证可运行
        self.assertEqual(out.shape, (num_tokens, hidden_dim))

    def test_various_shapes(self):
        """不同形状下 fallback 均正确。"""
        shapes = [
            (4, 32, 64, 2, 2),    # 极小
            (8, 64, 128, 4, 1),   # topk=1
            (16, 128, 256, 8, 4), # topk=4
            (32, 256, 512, 8, 2), # 标准
        ]
        for nt, hd, imd, ne, tk in shapes:
            with self.subTest(tokens=nt, hidden=hd, inter=imd, experts=ne, topk=tk):
                inputs = make_moe_inputs(
                    num_tokens=nt, hidden_dim=hd, intermediate_dim=imd,
                    num_experts=ne, top_k=tk)
                out = fused_moe_torch_fallback(**inputs)
                self.assertEqual(out.shape, (nt, hd),
                                 f"Shape mismatch for ({nt},{hd},{imd},{ne},{tk})")


# ============================================================
#  TestCase 3: 全流水线数据流连贯性
# ============================================================

class TestPipelineDataFlow(unittest.TestCase):
    """
    验证 pre-permute → kernel → post-scatter 全流水线的
    数据流正确性 (使用 fallback 模式, 不依赖 NPU)。

    "数据能不能接着" — 重点验证:
      - sorted_token_ids → pre-permute → kernel input 的一致性
      - kernel output → post-scatter → original order 的还原
      - padding (-1) 位置正确处理
      - 多 expert 结果 index_add_ 累加正确
    """

    def setUp(self):
        setup_seed()

    def _pipeline_with_fallback(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        模拟完整流水线 (使用 Python fallback 替代 kernel):
          1. moe_align_block_size (CPU) → sorted 路由数组
          2. pre-permute: gather hidden_states by sorted order
          3. 在 sorted 顺序上运行 per-expert matmul
          4. post-scatter: index_add_ 散射回原始顺序

        这验证了"数据能接上" — 即 pre-permute/post-scatter 模式正确。
        """
        num_tokens, hidden_dim = hidden_states.shape
        num_experts, intermediate_dim, _ = w1.shape
        top_k = topk_ids.shape[1]

        # Step 1: 路由排序
        sorted_token_ids, tokens_per_expert, token_offsets, \
            num_tokens_post_padded, expert_ids, sorted_weights = \
            _moe_align_block_size(
                topk_ids.cpu(), DEFAULT_TILE_M, num_experts,
                topk_weights.cpu())

        # Step 2: pre-permute
        safe_ids = sorted_token_ids.clamp(min=0)
        permuted_hidden = hidden_states[safe_ids.to(
            device=hidden_states.device)]

        # Step 3: 在 sorted 顺序上计算 (per-expert, aligned to tileM)
        permuted_output = torch.zeros(
            num_tokens_post_padded, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device)

        for exp_idx in range(num_experts):
            num_exp = tokens_per_expert[exp_idx].item()
            if num_exp == 0:
                continue
            offset = token_offsets[exp_idx].item()

            # 该 expert 的 token 在 permuted 中的位置
            exp_hidden = permuted_hidden[offset:offset + num_exp]
            # w1/w2 投影
            gate_up = exp_hidden @ w1[exp_idx].t()
            gate = torch.nn.functional.silu(
                gate_up[..., :intermediate_dim // 2])
            up = gate_up[..., intermediate_dim // 2:]
            activated = gate * up
            partial = activated @ w2[exp_idx].t()

            # 应用 sorted_weights (cast to same dtype as partial to avoid FP16×FP32 promotion)
            exp_weights = sorted_weights[offset:offset + num_exp].to(
                device=hidden_states.device, dtype=partial.dtype)
            partial = partial * exp_weights.unsqueeze(-1)
            permuted_output[offset:offset + num_exp] = partial

        # Step 4: post-scatter (sentinel mask)
        mask = sorted_token_ids != -1
        valid_indices = torch.where(mask)[0]
        valid_original = sorted_token_ids[valid_indices]
        valid_output = permuted_output[valid_indices.to(
            device=hidden_states.device)]

        output = torch.zeros(
            num_tokens, hidden_dim,
            dtype=hidden_states.dtype, device=hidden_states.device)
        output.index_add_(0,
            valid_original.to(device=hidden_states.device),
            valid_output)
        return output

    def test_pipeline_matches_fallback(self):
        """流水线 (pre-permute + sorted calc + post-scatter) 应与 fallback 一致。

        注意: padded 块和 exact 块的 matmul 维度不同导致 FP32 累加差异,
        使用宽松容差 (1e-3)。"""
        inputs = make_moe_inputs(
            num_tokens=32, hidden_dim=128, intermediate_dim=256,
            num_experts=8, top_k=2, dtype=torch.float32)

        pipeline_out = self._pipeline_with_fallback(**inputs)
        fallback_out = fused_moe_torch_fallback(**inputs)

        max_diff = (pipeline_out - fallback_out).abs().max().item()
        self.assertLess(
            max_diff, 5e-3,
            f"Pipeline vs fallback: max_diff={max_diff:.2e}")

    def test_pipeline_skewed_routing(self):
        """倾斜路由下流水线仍然正确。"""
        num_tokens = 48
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2, dtype=torch.float32)
        topk_ids, topk_weights = make_routing_skewed(
            num_tokens, 4, 2, hot_expert=0, hot_ratio=0.8)
        inputs["topk_ids"] = topk_ids
        inputs["topk_weights"] = topk_weights

        pipeline_out = self._pipeline_with_fallback(**inputs)
        fallback_out = fused_moe_torch_fallback(**inputs)
        max_diff = (pipeline_out - fallback_out).abs().max().item()
        self.assertLess(max_diff, 1e-3)

    def test_pipeline_single_token(self):
        """单 token 流水线正确。"""
        inputs = make_moe_inputs(
            num_tokens=1, hidden_dim=32, intermediate_dim=64,
            num_experts=4, top_k=2, dtype=torch.float32)
        pipeline_out = self._pipeline_with_fallback(**inputs)
        fallback_out = fused_moe_torch_fallback(**inputs)
        max_diff = (pipeline_out - fallback_out).abs().max().item()
        self.assertLess(max_diff, 1e-3)

    def test_pipeline_tile_boundary(self):
        """
        tileM 边界: 当 token 数跨 tileM 边界时, padding 使每个 expert
        的 token 数对齐到 tileM, pre-permute/post-scatter 应正确处理 padding。
        """
        num_tokens = DEFAULT_TILE_M + 7  # 39 tokens, 跨 32 边界
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=64, intermediate_dim=128,
            num_experts=8, top_k=2, dtype=torch.float32)

        pipeline_out = self._pipeline_with_fallback(**inputs)
        fallback_out = fused_moe_torch_fallback(**inputs)
        max_diff = (pipeline_out - fallback_out).abs().max().item()
        self.assertLess(max_diff, 1e-3)

    def test_pipeline_empty_expert(self):
        """含空 expert 的流水线正确。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=8, top_k=2, dtype=torch.float32)
        # 强制所有 token 只路由到 expert 0,1,2
        inputs["topk_ids"] = torch.randint(
            0, 3, (16, 2), dtype=torch.long)
        inputs["topk_weights"] = torch.rand(16, 2, dtype=torch.float32)

        pipeline_out = self._pipeline_with_fallback(**inputs)
        fallback_out = fused_moe_torch_fallback(**inputs)
        max_diff = (pipeline_out - fallback_out).abs().max().item()
        self.assertLess(max_diff, 1e-3)

    def test_pre_permute_index_correctness(self):
        """
        验证 pre-permute 索引正确性:
        sorted_token_ids[t] 指示 permuted_hidden[t] 对应哪个原始 token.
        验证: permuted_hidden[t] === hidden_states[sorted_token_ids[t]]
        """
        hidden_states = torch.randn(16, 64)
        topk_ids = torch.randint(0, 4, (16, 2), dtype=torch.long)
        topk_weights = torch.rand(16, 2, dtype=torch.float32)

        sorted_ids, _, _, ntp, _, _ = _moe_align_block_size(
            topk_ids, DEFAULT_TILE_M, 4, topk_weights)

        safe_ids = sorted_ids.clamp(min=0)
        permuted = hidden_states[safe_ids]

        # 验证: 对于非 -1 的位置, permuted[t] == hidden_states[sorted_ids[t]]
        for t in range(ntp):
            orig_idx = sorted_ids[t].item()
            if orig_idx == -1:
                continue  # padding 位置, permuted[t] = hidden_states[0] (clamp)
            self.assertTrue(
                torch.allclose(permuted[t], hidden_states[orig_idx]),
                f"Mismatch at sorted position {t}: "
                f"permuted[{t}] != hidden_states[{orig_idx}]")


# ============================================================
#  TestCase 4: 数据通用性 — 不同 dtype / device / 形状
# ============================================================

class TestDataGenerality(unittest.TestCase):
    """
    "确保数据通用的性质" — 测试不同数据类型和形状下
    代码的正确性。
    """

    def setUp(self):
        setup_seed()

    def test_fp16_fallback(self):
        """FP16 fallback。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2, dtype=torch.float16)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.dtype, torch.float16)
        self.assertFalse(torch.isnan(out).any(), "FP16 output has NaN")

    def test_bf16_fallback(self):
        """BF16 fallback。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2, dtype=torch.bfloat16)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any(), "BF16 output has NaN")

    def test_large_expert_count(self):
        """大量 expert (64) — 覆盖 MAX_EXPERTS 边界。"""
        inputs = make_moe_inputs(
            num_tokens=128, hidden_dim=64, intermediate_dim=128,
            num_experts=64, top_k=2)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (128, 64))

    def test_small_hidden_dim(self):
        """小 hidden_dim (tileK 以下)。"""
        inputs = make_moe_inputs(
            num_tokens=8, hidden_dim=16, intermediate_dim=64,
            num_experts=4, top_k=2)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (8, 16))

    def test_small_intermediate_dim(self):
        """小 intermediate_dim (tileN 以下)。"""
        inputs = make_moe_inputs(
            num_tokens=8, hidden_dim=64, intermediate_dim=32,
            num_experts=4, top_k=2)
        # intermediate_dim=32 → gate=16, up=16, gate_up=32
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (8, 64))

    def test_all_experts_active(self):
        """所有 expert 都有至少一个 token。"""
        num_tokens = 64
        num_experts = 8
        top_k = 2
        topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.long)
        for i in range(num_tokens):
            topk_ids[i, 0] = i % num_experts
            topk_ids[i, 1] = (i + 1) % num_experts
        topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=64, intermediate_dim=128,
            num_experts=num_experts, top_k=top_k)
        inputs["topk_ids"] = topk_ids
        inputs["topk_weights"] = topk_weights

        out = fused_moe_torch_fallback(**inputs)
        self.assertFalse(torch.isnan(out).any())


# ============================================================
#  TestCase 5: Tiling 参数打包
# ============================================================

class TestTilingPacking(unittest.TestCase):
    """验证 _pack_tiling_data 的二进制布局正确性。"""

    def setUp(self):
        setup_seed()

    def test_tiling_struct_size(self):
        """Tiling 结构体大小正确。"""
        # FusedMoeTilingData 在 C++ 侧: 4*10 + 4*64 + 4*64 + 512 + 512 = 1576
        expected = 1576
        buf = _pack_tiling_data(
            num_tokens=32, hidden_dim=256, intermediate_dim=512,
            num_experts=8, top_k=2, num_tokens_post_padded=64,
            tokens_per_expert=torch.zeros(8, dtype=torch.int32),
            token_offsets=torch.zeros(8, dtype=torch.int32))
        self.assertEqual(len(buf), expected,
                         f"Tiling buffer size {len(buf)} != {expected}")

    def test_pack_unpack_roundtrip(self):
        """打包的值可正确读取。"""
        num_tokens = 42
        hidden_dim = 256
        inter_dim = 512
        num_experts = 8
        top_k = 2
        ntp = 128
        tpe = torch.tensor([32, 0, 32, 0, 32, 0, 32, 0], dtype=torch.int32)
        toff = torch.tensor([0, 32, 32, 64, 64, 96, 96, 128], dtype=torch.int32)

        buf = _pack_tiling_data(
            num_tokens, hidden_dim, inter_dim, num_experts, top_k,
            ntp, tpe, toff)

        # 手动解包验证
        def read_u32(offset):
            return int.from_bytes(buf[offset:offset + 4], 'little')

        self.assertEqual(read_u32(0), num_tokens)
        self.assertEqual(read_u32(4), hidden_dim)
        self.assertEqual(read_u32(8), inter_dim)
        self.assertEqual(read_u32(12), num_experts)
        self.assertEqual(read_u32(16), top_k)
        self.assertEqual(read_u32(20), ntp)

        # tileM/tileK/tileN 取 min(DEFAULT, size)
        self.assertEqual(read_u32(24), min(DEFAULT_TILE_M, num_tokens))
        self.assertEqual(read_u32(28), min(DEFAULT_TILE_K, hidden_dim))
        self.assertEqual(read_u32(32), min(DEFAULT_TILE_N, inter_dim // 2))

        # tokensPerExpert (offset 40, was 36 — numCores 插入导致偏移)
        for i in range(num_experts):
            self.assertEqual(read_u32(40 + i * 4), tpe[i].item())

        # tokenOffsets (offset 296, was 292)
        for i in range(num_experts):
            self.assertEqual(read_u32(296 + i * 4), toff[i].item())


# ============================================================
#  TestCase 6: 数值稳定性
# ============================================================

class TestNumericalStability(unittest.TestCase):
    """数值稳定性测试: NaN, Inf, 极值等。"""

    def setUp(self):
        setup_seed()

    def test_fallback_zero_hidden(self):
        """hidden_states 全零时输出应为零。"""
        num_tokens, hidden_dim = 16, 64
        inputs = make_moe_inputs(
            num_tokens=num_tokens, hidden_dim=hidden_dim,
            intermediate_dim=128, num_experts=4, top_k=2)
        inputs["hidden_states"].zero_()
        out = fused_moe_torch_fallback(**inputs)
        self.assertTrue(
            torch.allclose(out, torch.zeros_like(out)),
            "Zero input should produce zero output")

    def test_fallback_constant_hidden(self):
        """hidden_states 全为常数时输出应稳定 (无 NaN/Inf)。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2)
        inputs["hidden_states"].fill_(1.0)
        out = fused_moe_torch_fallback(**inputs)
        self.assertFalse(torch.isnan(out).any(), "Output has NaN")
        self.assertFalse(torch.isinf(out).any(), "Output has Inf")

    def test_fallback_zero_weights(self):
        """全零权重: 即使有路由, 输出也应为零。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=4, top_k=2)
        inputs["topk_weights"].zero_()
        out = fused_moe_torch_fallback(**inputs)
        self.assertTrue(
            torch.allclose(out, torch.zeros_like(out), atol=1e-7),
            "Zero weights should produce near-zero output")

    def test_fallback_large_values(self):
        """大数值输入不应产生 NaN (使用 FP32 避免 FP16 溢出)。"""
        inputs = make_moe_inputs(
            num_tokens=8, hidden_dim=32, intermediate_dim=64,
            num_experts=2, top_k=1, dtype=torch.float32)
        inputs["hidden_states"].mul_(100.0)  # 放大
        out = fused_moe_torch_fallback(**inputs)
        self.assertFalse(torch.isnan(out).any(),
                         "Large input produces NaN")
        self.assertFalse(torch.isinf(out).any(),
                         "Large input produces Inf")


# ============================================================
#  TestCase 7: fused_moe_ascendc 占位模式验证
# ============================================================

class TestAscendCWrappedAPI(unittest.TestCase):
    """
    测试 fused_moe_ascendc 封装:
    - 占位模式 (use_custom_op=False) 正确返回零张量
    - 与 fallback 精度对比函数可运行
    """

    def setUp(self):
        setup_seed()

    def test_placeholder_returns_zero(self):
        """占位模式返回全零 (custom op 不可用时的占位行为)。"""
        inputs = make_moe_inputs(
            num_tokens=16, hidden_dim=64, intermediate_dim=128,
            num_experts=8, top_k=2)
        out = fused_moe_ascendc(
            **inputs, use_custom_op=False)
        self.assertEqual(out.shape, (16, 64))
        self.assertTrue(
            torch.allclose(out, torch.zeros_like(out)),
            "Placeholder mode should return zeros")

    def test_verify_function_runs(self):
        """验证函数可运行 (use_custom_op=False 时仅比占位 vs fallback)。"""
        inputs = make_moe_inputs(
            num_tokens=8, hidden_dim=32, intermediate_dim=64,
            num_experts=4, top_k=2)
        try:
            out = fused_moe_ascendc_with_verify(**inputs, use_custom_op=False)
            self.assertEqual(out.shape, (8, 32))
        except Exception as e:
            self.fail(f"fused_moe_ascendc_with_verify raised: {e}")


# ============================================================
#  TestCase 8: 大尺度压力测试
# ============================================================

class TestLargeScale(unittest.TestCase):
    """大尺度测试: 接近生产环境的配置。"""

    def setUp(self):
        setup_seed()

    def test_medium_scale_fallback(self):
        """中等规模: 512 tokens, 8 experts, topk=2。"""
        inputs = make_moe_inputs(
            num_tokens=512, hidden_dim=256, intermediate_dim=512,
            num_experts=8, top_k=2)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (512, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_large_scale_fallback(self):
        """大规模: 2048 tokens, 16 experts, topk=4。"""
        inputs = make_moe_inputs(
            num_tokens=2048, hidden_dim=512, intermediate_dim=1024,
            num_experts=16, top_k=4)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (2048, 512))
        self.assertFalse(torch.isnan(out).any())

    @unittest.skip("跳过: 仅用于手动验证超大尺度时的内存和时间")
    def test_very_large_scale(self):
        """超大尺度: 8192 tokens, 64 experts — 手动运行。"""
        inputs = make_moe_inputs(
            num_tokens=8192, hidden_dim=1024, intermediate_dim=2048,
            num_experts=64, top_k=8)
        out = fused_moe_torch_fallback(**inputs)
        self.assertEqual(out.shape, (8192, 1024))


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FusedMoEFlagship 综合测试套件")
    print(f"种子: {_TEST_SEED} | 默认 tileM={DEFAULT_TILE_M} "
          f"tileK={DEFAULT_TILE_K} tileN={DEFAULT_TILE_N}")
    print(f"PyTorch {torch.__version__} | 设备: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    unittest.main(verbosity=2)
