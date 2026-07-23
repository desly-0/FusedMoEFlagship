# FusedMoEFlagship FL 插件集成方案

## 1. 替换点定位

在 FL 插件的 MoE FFN 调用链中, 需要替换的核心函数是逐 expert 循环的 `_torch_fused_experts_impl`。

**典型路径**: `vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py`

```python
# 替换前: 逐 expert 循环, 每次循环触发 3 次 Slice
def fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, ...):
    output = torch.zeros_like(hidden_states)
    for expert_idx in range(num_experts):       # 64 次循环
        # 每次循环 3 次 Slice → 5,376 次总 Slice 调用
        expert_hidden = hidden_states[mask]     # Slice
        gate_weight = w1[expert_idx]            # Slice
        down_weight = w2[expert_idx]            # Slice
        ...

# 替换后: 单次自定义算子调用
def fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, ...):
    if _is_fused_moe_custom_available():
        return fused_moe_ascendc(hidden_states, w1, w2, topk_weights, topk_ids, ...)
    else:
        return _torch_fused_experts_impl(hidden_states, w1, w2, ...)
```

## 2. 算子可用性检测

```python
import importlib
import torch

_FUSED_MOE_CUSTOM_AVAILABLE = None

def _is_fused_moe_custom_available() -> bool:
    global _FUSED_MOE_CUSTOM_AVAILABLE
    if _FUSED_MOE_CUSTOM_AVAILABLE is not None:
        return _FUSED_MOE_CUSTOM_AVAILABLE

    try:
        # 方式 1: torch.ops 注册
        if hasattr(torch.ops, 'fl_custom') and \
           hasattr(torch.ops.fl_custom, 'fused_moe_flagship'):
            _FUSED_MOE_CUSTOM_AVAILABLE = True
            return True

        # 方式 2: 尝试加载 .so
        import os
        so_path = os.environ.get(
            'FUSED_MOE_CUSTOM_SO',
            '/usr/local/Ascend/operator/lib/libfused_moe_flagship.so'
        )
        if os.path.exists(so_path):
            torch.ops.load_library(so_path)
            _FUSED_MOE_CUSTOM_AVAILABLE = True
            return True
    except Exception as e:
        print(f"[FusedMoE] Custom op not available: {e}")

    _FUSED_MOE_CUSTOM_AVAILABLE = False
    return False
```

## 3. 编译与部署

### 3.1 编译

```bash
# 设置 CANN 路径
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

# 构建
cd FusedMoEFlagship
mkdir -p build && cd build
cmake .. -DASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME
make -j$(nproc)

# 打包
make fused_moe_package
# 输出在 build/package/
#   op_kernel/fused_moe_flagship.o
#   op_host/libfused_moe_flagship.so
```

### 3.2 部署

```bash
# 部署到 CANN 算子目录
cp build/package/op_kernel/fused_moe_flagship.o \
    $ASCEND_TOOLKIT_HOME/opp/vendors/custom/op_kernel/
cp build/package/op_host/libfused_moe_flagship.so \
    $ASCEND_TOOLKIT_HOME/opp/vendors/custom/op_host/
```

### 3.3 在 FL 插件中加载

```python
# 在 FL 插件初始化时加载
import torch

def init_custom_ops():
    """初始化自定义算子"""
    custom_so = os.path.join(
        os.path.dirname(__file__),
        'libfused_moe_flagship.so'
    )
    if os.path.exists(custom_so):
        torch.ops.load_library(custom_so)
        print("[FusedMoE] Custom operator loaded")
        return True
    return False
```

## 4. 精度验证

### 4.1 单算子验证

```bash
python -c "
from fused_moe_ascendc import fused_moe_ascendc_with_verify

# 构造测试输入
T, H, E, N, K = 128, 7168, 64, 4096, 2
hidden = torch.randn(T, H, dtype=torch.float16).npu()
w1 = torch.randn(E, N, H, dtype=torch.float16).npu()
w2 = torch.randn(E, H, N//2, dtype=torch.float16).npu()
topk_weights = torch.randn(T, K, dtype=torch.float32).npu()
topk_ids = torch.randint(0, E, (T, K), dtype=torch.int32).npu()

output = fused_moe_ascendc_with_verify(hidden, w1, w2, topk_weights, topk_ids)
"
```

### 4.2 端到端验证

在 FL 插件配置中启用验证模式:

```python
os.environ['FUSED_MOE_VERIFY'] = '1'  # 启用对比验证
```

## 5. 性能监控

集成后需要关注的关键指标:

| 指标 | 预期 | 说明 |
|:----:|:----:|:------|
| Slice 占比 | < 5% | 从 50.10% 下降 |
| Vector Core 利用率 | ~50% | 从 78.95% 下降 |
| TTFT | ~550ms | 从 970ms 下降 |
| TPOT | ~58ms | 从 99ms 下降 |
| Output Throughput | +70% | 从 9.99 tok/s 提升 |

通过 `npu-smi` 和 CANN Profiling 工具采集数据。

## 6. 回退策略

当自定义算子不可用时, 自动回退到原始实现:

```python
def fused_experts_impl(...):
    if _is_fused_moe_custom_available():
        try:
            return fused_moe_ascendc(...)
        except Exception as e:
            print(f"[FusedMoE] Custom op failed: {e}, falling back")
    
    return _torch_fused_experts_impl(...)
```
