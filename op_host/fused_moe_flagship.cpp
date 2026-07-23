// ============================================================
// fused_moe_flagship.cpp — Host 侧算子注册 + Tiling 实现
//
// CANN 8.5.0 API (接口参考 6.3/6.4/6.2, 开发指南 2.9.2)
// - OpDef 原型注册: register/op_def.h + op_def_registry.h
// - TilingData: 直调工程, 手动序列化 POD 结构体
// - Tiling 函数: ge::graphStatus(gert::TilingContext*)
// - InferShape/DataType: gert::InferShapeContext/InferDataTypeContext
// - 属性访问: context->GetAttrs() → RuntimeAttrs::GetAttr/GetAttrPointer
//
// 平台: Ascend 910B (dav-2201), CANN 8.5.0
// ============================================================

#include <cstdint>
#include <algorithm>

// CANN 8.5.0 原型注册 API (接口参考 6.3.3 OpDef, 6.3.1 OP_ADD)
#include "register/op_def.h"
#include "register/op_def_registry.h"

// Matmul Tiling API (接口参考 5.2.2)
#include "adv_api/matmul/matmul_tiling.h"

// 平台信息 API (接口参考 6.2.1 PlatformAscendC)
#include "tiling/platform/platform_ascendc.h"

// TilingData 结构体 (Kernel 与 Host 共享)
#include "fused_moe_tiling.h"

using namespace AscendC;

// ========== 分块常量 (910B: L1=256KB, UB=256KB) ==========
namespace {
    constexpr int32_t DEFAULT_TILE_M = 32;
    constexpr int32_t DEFAULT_TILE_K = 64;
    constexpr int32_t DEFAULT_TILE_N = 128;
    constexpr int32_t MAX_EXPERTS = 64;
}

namespace optiling {

// ---------- 辅助: 生成单个 MatMul 的 Cube Tiling ----------
// 接口参考 5.2.2 MultiCoreMatmulTiling
static bool GenerateMatmulTiling(
    matmul_tiling::MultiCoreMatmulTiling& tilingObj,
    optiling::TCubeTiling& outCubeTiling,
    int32_t m, int32_t n, int32_t k,
    bool transB)
{
    tilingObj.SetDim(1);  // 单核执行每路 MatMul

    // 矩阵类型配置: A/B/C 均为 GM, ND, FP16
    tilingObj.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingObj.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, transB);
    tilingObj.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);

    // 形状配置
    tilingObj.SetShape(m, n, k);
    tilingObj.SetOrgShape(m, n, k);
    tilingObj.SetSingleShape(m, n, k);

    // Norm 模板
    tilingObj.SetMatmulConfigParams(0);

    // 生成 Tiling 参数
    return (tilingObj.GetTiling(outCubeTiling) != -1);
}

// ---------- Tiling 主函数 ----------
// 开发指南 2.9.2.5: Tiling 函数签名
//   入参: gert::TilingContext*  (接口参考 6.3.6 SetTiling)
//   返回值: ge::graphStatus
ge::graphStatus FusedMoEFlagshipTiling(gert::TilingContext* context)
{
    // ---- 1. 读取输入形状 ----
    // TilingContext::GetInputShape → const StorageShape*
    // StorageShape::GetShape() → const Shape&  (storage_shape.h)
    // Shape::GetDim(n) → int64_t                (shape.h)
    const auto* hiddenShape = context->GetInputShape(0);
    const auto* w1Shape     = context->GetInputShape(1);

    int32_t numTokens   = static_cast<int32_t>(hiddenShape->GetShape().GetDim(0));
    int32_t hiddenDim   = static_cast<int32_t>(hiddenShape->GetShape().GetDim(1));
    int32_t interDim    = static_cast<int32_t>(w1Shape->GetShape().GetDim(1));  // w1: [E, N, D]
    int32_t numExperts  = static_cast<int32_t>(w1Shape->GetShape().GetDim(0));
    int32_t topK        = static_cast<int32_t>(context->GetInputShape(4)->GetShape().GetDim(1));

    // ---- 2. Tile 大小 ----
    int32_t tileM = std::min(DEFAULT_TILE_M, numTokens);
    int32_t tileK = std::min(DEFAULT_TILE_K, hiddenDim);
    int32_t tileN = std::min(DEFAULT_TILE_N, interDim / 2);

    // ---- 3. 核心数计算 ----
    // PlatformAscendC 获取硬件信息 (接口参考 6.2.1)
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t maxCores = static_cast<int32_t>(platform.GetCoreNumAic());

    // ---- 属性访问 (接口参考 XOR 官方示例 p.1005) ----
    // context->GetAttrs() → const RuntimeAttrs*
    //   Int() 标量属性: *(attrs->GetAttrPointer<T>(idx))
    //   ListInt() 属性:  attrs->GetAttrPointer<T>(idx)
    // 索引顺序: 0=tokensPerExpert, 1=tokenOffsets, 2=activation,
    //           3=numTokensPostPadded, 4=numCores
    const auto* attrs = context->GetAttrs();
    int32_t numCoresFromAttr = static_cast<int32_t>(
        *(attrs->GetAttrPointer<int64_t>(4)));
    int32_t numCores = std::max(1, std::min({maxCores, numExperts, numCoresFromAttr}));

    // ---- 4. 填充 TilingData ----
    // 直调工程: 使用 POD 结构体 + 手动字节拷贝到 rawTiling 缓冲区
    FusedMoeTilingData tiling;
    tiling.numTokens       = static_cast<uint32_t>(numTokens);
    tiling.hiddenDim       = static_cast<uint32_t>(hiddenDim);
    tiling.intermediateDim = static_cast<uint32_t>(interDim);
    tiling.numExperts      = static_cast<uint32_t>(numExperts);
    tiling.topK            = static_cast<uint32_t>(topK);
    tiling.tileM           = static_cast<uint32_t>(tileM);
    tiling.tileK           = static_cast<uint32_t>(tileK);
    tiling.tileN           = static_cast<uint32_t>(tileN);
    tiling.numCores        = static_cast<uint32_t>(numCores);

    // ListInt 属性: GetAttrPointer<T>(index) 返回数组指针
    const int64_t* tokensPerExpertVal = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* tokenOffsetsVal    = attrs->GetAttrPointer<int64_t>(1);
    int32_t expCount = std::min(numExperts, MAX_EXPERTS);
    for (int32_t i = 0; i < expCount; i++) {
        tiling.tokensPerExpert[i] = static_cast<uint32_t>(tokensPerExpertVal[i]);
        tiling.tokenOffsets[i]    = static_cast<uint32_t>(tokenOffsetsVal[i]);
    }

    // Int 属性: 官方模式使用 *(GetAttrPointer<T>(idx)) 解引用
    tiling.numTokensPostPadded = static_cast<uint32_t>(
        *(attrs->GetAttrPointer<int64_t>(3)));

    // ---- 5. 生成 Cube Tiling 参数 ----
    // MM1: hidden_states × w1_gate_up^T → gate_up [tileM, interDim]
    {
        matmul_tiling::MultiCoreMatmulTiling mm1Tiling(platform);
        if (!GenerateMatmulTiling(
                mm1Tiling,
                reinterpret_cast<optiling::TCubeTiling&>(tiling.cubeTilingMM1),
                tileM, interDim, tileK, true)) {
            context->SetTilingKey(-1);
            return ge::GRAPH_FAILED;
        }
    }

    // MM2: activated × w2^T → output [tileM, hiddenDim]
    {
        matmul_tiling::MultiCoreMatmulTiling mm2Tiling(platform);
        if (!GenerateMatmulTiling(
                mm2Tiling,
                reinterpret_cast<optiling::TCubeTiling&>(tiling.cubeTilingMM2),
                tileM, hiddenDim, tileN, true)) {
            context->SetTilingKey(-1);
            return ge::GRAPH_FAILED;
        }
    }

    // ---- 6. Workspace 大小 ----
    // gate_up 中间结果: numCores × tileM × interDim × sizeof(half)
    size_t wsUser = static_cast<size_t>(numCores) *
                    static_cast<size_t>(tileM) *
                    static_cast<size_t>(interDim) *
                    sizeof(uint16_t);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = platform.GetLibApiWorkSpaceSize() + wsUser;

    // ---- 7. 输出 Tiling 信息 ----
    context->SetBlockDim(static_cast<uint32_t>(numCores));
    context->SetTilingKey(1);  // TilingKey = 1 → 正常模式

    // 序列化 TilingData 到 raw buffer (直调工程, 手动拷贝)
    auto* rawTiling = context->GetRawTilingData();
    if (sizeof(FusedMoeTilingData) > rawTiling->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    auto* src = reinterpret_cast<const uint8_t*>(&tiling);
    auto* dst = rawTiling->GetData();
    for (uint64_t i = 0; i < sizeof(FusedMoeTilingData); i++) {
        dst[i] = src[i];
    }
    rawTiling->SetDataSize(sizeof(FusedMoeTilingData));

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// ========== Shape 推导函数 ==========
// 开发指南 2.9.2.3: InferShape/InferDataType 签名
//   ge::graphStatus(gert::InferShapeContext*)  (接口参考 6.3.3 SetInferShape)
//   ge::graphStatus(gert::InferDataTypeContext*) (接口参考 6.3.3 SetInferDataType)
namespace ge {

static uint32_t InferShape(gert::InferShapeContext* context)
{
    // output shape = hidden_states shape (input 0)
    const auto* inputShape = context->GetInputShape(0);
    auto* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return 1;  // ge::GRAPH_FAILED
    }
    *outputShape = *inputShape;
    return 0;  // ge::GRAPH_SUCCESS
}

static uint32_t InferDataType(gert::InferDataTypeContext* context)
{
    // output dtype = hidden_states dtype = FLOAT16
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return 0;
}

}  // namespace ge

// ========== 算子原型定义 ==========
// 接口参考 6.3.3 OpDef, 6.3.1 OP_ADD
namespace ops {

class FusedMoEFlagship : public OpDef {
public:
    explicit FusedMoEFlagship(const char* name) : OpDef(name)
    {
        // ---- 输入 (按索引 0-6) ----
        this->Input("hiddenStates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("w1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("w2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("tempBuffer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("sortedTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        this->Input("topkWeights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Input("sortedWeights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // ---- 输出 ----
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // ---- 属性 (索引: 0=tokensPerExpert, 1=tokenOffsets, 2=activation,
        //              3=numTokensPostPadded, 4=numCores) ----
        this->Attr("tokensPerExpert")
            .AttrType(REQUIRED)
            .ListInt();

        this->Attr("tokenOffsets")
            .AttrType(REQUIRED)
            .ListInt();

        this->Attr("activation")
            .AttrType(OPTIONAL)
            .String("silu");

        this->Attr("numTokensPostPadded")
            .AttrType(REQUIRED)
            .Int();

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
// 接口参考 6.3.1 OP_ADD
OP_ADD(ops::FusedMoEFlagship);
