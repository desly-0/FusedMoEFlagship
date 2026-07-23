# FusedMoEFlagship

Ascend C 实现的自定义融合 MoE 算子。主要目标是将 MoE FFN 计算中逐 expert 循环 + 多次 Slice 调用的模式合并为单次算子调用，减少中间数据搬运开销。

## 背景

在 MoE 模型的推理中，FL 插件原始的 `fused_experts_impl` 实现采用逐 expert 的循环方式。每次循环需要从大张量中切片（Slice）取出当前 expert 对应的 gate 权重、up 权重和 down 权重。当 expert 数量较多时（如 64 个 expert），这种模式会产生大量 Slice 调用，在 Ascend NPU 上 Slice 操作的开销较高。

这个项目尝试用自定义 Ascend C 算子替代原有的实现方式。核心思路是用指针偏移代替数据拷贝（通过 `SetGlobalBuffer` 设置起始地址），将两次 MatMul 和一次 SiLU 激活融合到一个算子内部。

## 整体计算流程

```
hidden_states           w1 (gate+up)
      │                      │
      └────── MatMul ────────┘
                  │
             gate / up 分割
                  │
             SiLU(gate) * up
                  │
      ┌──────────┘
      │           w2 (down)
      └── MatMul ─┘
            │
         output (按 expert 累加)
```

## 代码结构

```
├── CMakeLists.txt                     # 算子编译配置
├── op_kernel/
│   ├── fused_moe_flagship.cpp         # Ascend C kernel 实现
│   └── fused_moe_tiling.h             # Tiling 结构体（Host 和 Kernel 共享）
├── op_host/
│   └── fused_moe_flagship.cpp         # Host 侧 Tiling 和算子注册
├── python/
│   ├── fused_moe_ascendc.py           # Python 封装（排序、Tiling组装、算子调用）
│   └── test_fused_moe.py              # 测试用例
└── vllm-plugin-FL/                    # FL 插件（算子集成目标系统）
```

## 环境要求

- **CANN 版本**: 8.5.0 (必须)
- **目标平台**: Ascend 910B (dav-2201)
- **工具链**: bisheng_compiler (kernel .o) + g++ (host .so)
- **API 参考**: CANN 社区版 8.5.0 Ascend C 算子开发接口参考 (2026-04-02)

> 注意: 本项目依赖 CANN 8.5.0 特有的 API (gert::TilingContext, RuntimeAttrs 等)，
> 不保证与旧版本 CANN 兼容。使用前请确认 `ASCEND_TOOLKIT_HOME` 指向 CANN 8.5.0 安装路径。

## 使用方法

### 编译

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
mkdir build && cd build
cmake .. -DASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME
make -j$(nproc)
```

编译产物：
- `build/op_kernel/fused_moe_flagship.o` — Kernel 二进制
- `build/op_host/libfused_moe_flagship.so` — Host 侧注册库

### 部署

将编译产物部署到 CANN 自定义算子目录后，FL 插件可以通过 `torch.ops.load_library` 加载使用。

### 单算子验证

```bash
python python/test_fused_moe.py
```

测试覆盖包括：`_moe_align_block_size` 路由结构正确性、PyTorch 参考实现对比、全流水线数据流、边界情况（空 expert、单 token、topK 变化）以及不同形状组合。

## 集成方式

FL 插件通过 `vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py` 中的 `fused_experts_impl` 函数调用该算子。当自定义算子不可用时，会自动回退到原始的 PyTorch 参考实现。

## 参考文档

- CANN 8.5.0 Ascend C 算子开发指南
- CANN 8.5.0 Ascend C 算子开发接口参考
