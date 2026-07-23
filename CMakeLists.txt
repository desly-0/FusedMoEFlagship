# ============================================================
# FusedMoEFlagship — AscendC 自定义算子 CMake 构建配置
#
# 构建方式:
#   mkdir build && cd build
#   cmake .. -DASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
#   make
#
# 输出:
#   build/op_kernel/fused_moe_flagship.o  (kernel 二进制)
#   build/op_host/libfused_moe_flagship.so  (host 注册库)
#
# 平台:
#   - CANN 8.5.0 / Ascend 910B (dav-2201)
#   - 需要设置 ASCEND_TOOLKIT_HOME 环境变量
#
# 参考:
#   CANN 8.5.0 Ascend C算子开发指南, 章节 2.3.1.3 / 2.3.1.4
#   cmake_minimum_required: ≥3.16 (doc p39)
#   bisheng flags: -x asc --npu-arch=dav-2201 -O2 (doc p45-46)
# ============================================================

cmake_minimum_required(VERSION 3.16)
project(FusedMoEFlagship LANGUAGES CXX)

# ========== CANN 工具链路径 ==========
if(NOT DEFINED ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(NOT ASCEND_TOOLKIT_HOME)
    message(FATAL_ERROR "ASCEND_TOOLKIT_HOME not set. "
                        "Please set it to your CANN toolkit path, e.g.\n"
                        "  export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}")

# ========== Kernel 编译 (AI Core .o) ==========
# 使用 bisheng (AscendC 编译器) 编译核函数
# 注意: 需要 -x asc 标志以将 .cpp 识别为 Ascend C 源码 (而非 C++)
#        --npu-arch=dav-2201 对应 Atlas A2 系列 (含 910B)
#        参考 CANN 算子开发指南 2.3.1.2 bisheng 命令行

set(KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_flagship.cpp
)

set(KERNEL_OUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/op_kernel
)

add_custom_command(
    OUTPUT ${KERNEL_OUT_DIR}/fused_moe_flagship.o
    DEPENDS ${KERNEL_SRC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUT_DIR}
    COMMAND ${ASCEND_TOOLKIT_HOME}/tools/bisheng_compiler/bin/bisheng
        -x asc
        --npu-arch=dav-2201
        -O2
        -I ${ASCEND_TOOLKIT_HOME}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        -c ${KERNEL_SRC}
        -o ${KERNEL_OUT_DIR}/fused_moe_flagship.o
    COMMENT "Compiling AscendC kernel (fused_moe_flagship.o)"
)

add_custom_target(fused_moe_kernel ALL
    DEPENDS ${KERNEL_OUT_DIR}/fused_moe_flagship.o
)

# ========== Host 编译 (.so) ==========
# 使用 g++ 编译 Host 端注册 + Tiling 代码
# 参考 CANN 算子开发指南 2.3.1.3 常用编译选项

set(HOST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/fused_moe_flagship.cpp
)

set(HOST_OUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/op_host
)

set(HOST_INCLUDE_DIRS
    ${ASCEND_TOOLKIT_HOME}/include
    ${ASCEND_TOOLKIT_HOME}/include/toolchain
    ${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc/include   # adv_api/matmul/bmm_tiling.h
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel               # fused_moe_tiling.h
)

set(HOST_LINK_DIRS
    ${ASCEND_TOOLKIT_HOME}/lib64
)

add_library(fused_moe_flagship_host SHARED ${HOST_SRC})
target_include_directories(fused_moe_flagship_host PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(fused_moe_flagship_host PRIVATE ${HOST_LINK_DIRS})
target_link_libraries(fused_moe_flagship_host
    ge_compiler
    runtime
)

# 设置 C++ 标准
target_compile_features(fused_moe_flagship_host PUBLIC cxx_std_17)

# ========== 打包规则 ==========
# 将 kernel .o 和 host .so 打包为自定义算子包

set(PACKAGE_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/package)

add_custom_target(fused_moe_package ALL
    DEPENDS fused_moe_kernel fused_moe_flagship_host
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${PACKAGE_OUT_DIR}/op_kernel
        ${PACKAGE_OUT_DIR}/op_host
    COMMAND ${CMAKE_COMMAND} -E copy
        ${KERNEL_OUT_DIR}/fused_moe_flagship.o
        ${PACKAGE_OUT_DIR}/op_kernel/
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:fused_moe_flagship_host>
        ${PACKAGE_OUT_DIR}/op_host/
    COMMENT "Packaging FusedMoEFlagship custom operator"
)

# ========== 自定义 target: 直接构建 python 可用的 .so ==========
# 用于 FL 插件集成 (kernel + tiling 合为单个 .so)
# 实际需要将 kernel .o 链接进 host .so

message(STATUS "\n=== FusedMoEFlagship Build Configuration ===")
message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}")
message(STATUS "Kernel output: ${KERNEL_OUT_DIR}/fused_moe_flagship.o")
message(STATUS "Host output: $<TARGET_FILE:fused_moe_flagship_host>")
message(STATUS "")
message(STATUS "To build:  cmake .. -DASCEND_TOOLKIT_HOME=<path> && make")
message(STATUS "To install: make fused_moe_package")
