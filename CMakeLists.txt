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

# ========== ASC CMake 框架 (提供 ascendc_fatbin_library 等宏) ==========
# find_package(ASC) 在 CANN 安装中可用, 搜索 ${ASCEND_TOOLKIT_HOME}/cmake/
# PDF §2.9.7 p292: ascendc_fatbin_library 编译 kernel 为 fatbin .o 格式
# (含 .ascend.kernel 段), 供 aclrtBinaryLoadFromData + aclrtBinaryGetFunction 使用
if(NOT DEFINED ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(ASCEND_TOOLKIT_HOME)
    list(APPEND CMAKE_PREFIX_PATH "${ASCEND_TOOLKIT_HOME}")
endif()
find_package(ASC REQUIRED)

project(FusedMoEFlagship LANGUAGES ASC CXX)

# ========== CANN 路径 ==========
if(NOT DEFINED ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(NOT ASCEND_TOOLKIT_HOME)
    message(FATAL_ERROR "ASCEND_TOOLKIT_HOME not set. "
                        "Please set it to your CANN toolkit path, e.g.\n"
                        "  export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}")

# Set install prefix to a knowable build-dir path for ascendc_fatbin_library
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/asc_install")

# ========== Kernel 编译 (fatbin .o for aclrtBinaryLoadFromData) ==========
# ascendc_fatbin_library 编译 kernel 源文件为 fatbin .o,
# 输出至 ${CMAKE_INSTALL_PREFIX}/fatbin/${target_name}/${target_name}.o
# fatbin 包含 .ascend.kernel 段, aclrtBinaryGetFunction 可从中找到函数入口。
# PDF p292: ascendc_fatbin_library 语法说明

set(KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_flagship.cpp
)

# 标记 .cpp 为 ASC 语言 (bisheng 编译)
set_source_files_properties(
    ${KERNEL_SRC}
    PROPERTIES LANGUAGE ASC
)

ascendc_fatbin_library(fused_moe_flagship_kernel
    ${KERNEL_SRC}
)

ascendc_compile_options(fused_moe_flagship_kernel PRIVATE
    --npu-arch=dav-2201 -O2
)

ascendc_include_directories(fused_moe_flagship_kernel PRIVATE
    ${ASCEND_TOOLKIT_HOME}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# Copy fatbin .o 到已知位置供 torch_op 运行时加载
set(KERNEL_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/op_kernel)
set(FATBIN_O_PATH "${CMAKE_INSTALL_PREFIX}/fatbin/fused_moe_flagship_kernel/fused_moe_flagship_kernel.o")

# 旧 kernel add_custom_target 被 ascendc_fatbin_library 替代,
# 用 fused_moe_kernel 做 post-build 拷贝
add_custom_target(fused_moe_kernel ALL
    DEPENDS fused_moe_flagship_kernel
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_OUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FATBIN_O_PATH}"
        "${KERNEL_OUT_DIR}/fused_moe_flagship.o"
    COMMENT "Copy fatbin kernel to ${KERNEL_OUT_DIR}/fused_moe_flagship.o"
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
# CANN 8.5.0 开发指南 表2-5/2-6:
# - ascendc/ascendeclib 在 CANN 8.5.0 中不存在（已移除）
# - AscendC 运行时由 ge_compiler 提供
# - 使用高阶 API (bmm_tiling PlatformAscendC) 需链接 register/platform/graph_base
target_link_libraries(fused_moe_flagship_host
    ge_compiler
    runtime
    register        # 表2-5: Tiling注册 (OpDef/OP_ADD)
    graph_base      # 表2-5: ge::Shape, ge::DataType
    platform        # 表2-5: PlatformAscendC
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

# ========== PyTorch Extension: libfused_moe_flagship.so ==========
# 编译为 PyTorch custom op, 可供 vLLM FL 插件直接加载使用。
# 通过 torch.ops.load_library("libfused_moe_flagship.so") 加载，
# 注册 torch.ops.fl_custom.fused_moe_flagship。
#
# 包含:
#   - kernel 源码 (嵌入为字符串, RTC 运行时编译)
#   - tiling 生成 (MultiCoreMatmulTiling)
#   - ACL runtime kernel 启动

# ---- 1. 查找 PyTorch (通过 Python 确定 cmake 路径) ----
execute_process(
    COMMAND python3 -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)
    message(STATUS "PyTorch found: ${TORCH_CMAKE_PREFIX}")
else()
    message(FATAL_ERROR "PyTorch not found. Install PyTorch first.")
endif()

# ---- 1b. 查找 Python 开发头文件 (Python.h) ----
# torch/extension.h -> python_headers.h -> Python.h
find_package(Python3 COMPONENTS Development QUIET)
if(NOT Python3_FOUND)
    # fallback: 常见自定义安装路径
    set(_PY_INC "/usr/local/python3.11.14/include/python3.11")
    if(EXISTS "${_PY_INC}/Python.h")
        set(Python3_INCLUDE_DIRS "${_PY_INC}")
        set(Python3_FOUND TRUE)
        message(STATUS "Python3 dev headers (manual): ${Python3_INCLUDE_DIRS}")
    endif()
endif()
if(NOT Python3_FOUND)
    message(WARNING "Python.h not found. "
        "If compilation fails, set -DPython3_INCLUDE_DIRS=<path-to-Python.h>")
endif()

# ---- 2. 嵌入 kernel 源码 → C++ 用于 RTC (PDF §2.3.1.5) ----
set(KERNEL_SRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/torch_op/kernel_src.cpp)

add_custom_command(
    OUTPUT ${KERNEL_SRC_CPP}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_flagship.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_tiling.h
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/torch_op
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/torch_op/embed_source.py
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_flagship.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fused_moe_tiling.h
        ${KERNEL_SRC_CPP}
    COMMENT "Embedding kernel source for RTC (torch_op/kernel_src.cpp)"
)

# ---- 3. 编译 shared library ----
set(TORCH_OP_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_op/fused_moe_flagship_op.cpp
    ${KERNEL_SRC_CPP}
)

add_library(fused_moe_flagship SHARED ${TORCH_OP_SRC})

target_include_directories(fused_moe_flagship PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel    # fused_moe_tiling.h
    ${ASCEND_TOOLKIT_HOME}/include           # acl/acl.h, tiling/...
    ${ASCEND_TOOLKIT_HOME}/aarch64-linux/asc/include  # adv_api/matmul/...
    ${TORCH_INCLUDE_DIRS}                    # torch/extension.h, ...
    ${Python3_INCLUDE_DIRS}                  # Python.h
)

target_link_directories(fused_moe_flagship PRIVATE
    ${ASCEND_TOOLKIT_HOME}/lib64
    ${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64
)

target_link_libraries(fused_moe_flagship
    ${TORCH_LIBRARIES}       # libtorch.so, libtorch_cpu.so
    -Wl,--no-as-needed       # 强制链接 CANN 库 (避免 --as-needed 丢弃 vtable 符号)
    ascendcl                 # ACL runtime: aclrtBinaryLoadFromData, aclrtLaunchKernelWithConfig
    register                 # MultiCoreMatmulTiling (adv_api/matmul/bmm_tiling.h)
    tiling_api               # PlatformAscendCManager (接口参考 6.2.2)
    platform                 # PlatformAscendC (tiling/platform/platform_ascendc.h)
    graph_base               # ge::Shape (dependency of tiling headers)
    acl_rtc                  # RTC: aclrtcCreateProg, aclrtcCompileProg (PDF §2.3.1.5)
    -Wl,--as-needed          # 恢复默认行为
)

target_compile_features(fused_moe_flagship PUBLIC cxx_std_17)

# Optional: pre-compile kernel .o for build-time validation (RTC will re-compile at runtime)
add_dependencies(fused_moe_flagship fused_moe_kernel)

# Set RPATH so the .so can find its CANN dependencies
set_target_properties(fused_moe_flagship PROPERTIES
    BUILD_RPATH "${ASCEND_TOOLKIT_HOME}/lib64;${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64"
    INSTALL_RPATH "${ASCEND_TOOLKIT_HOME}/lib64;${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64"
)

message(STATUS "Torch op target: libfused_moe_flagship.so")
message(STATUS "  For FL integration: set FUSED_MOE_FLAGSHIP_LIB=<path>/libfused_moe_flagship.so")

# ---- 更新 package 目标以包含新的 .so ----
add_custom_command(TARGET fused_moe_package POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:fused_moe_flagship>
        ${PACKAGE_OUT_DIR}/op_host/
    COMMENT "Adding libfused_moe_flagship.so to package"
)

message(STATUS "\n=== FusedMoEFlagship Build Configuration ===")
message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}")
message(STATUS "Kernel output: ${KERNEL_OUT_DIR}/fused_moe_flagship.o")
message(STATUS "Host output: $<TARGET_FILE:fused_moe_flagship_host>")
message(STATUS "Torch op output: $<TARGET_FILE:fused_moe_flagship>")
message(STATUS "")
message(STATUS "To build:  cmake .. -DASCEND_TOOLKIT_HOME=<path> && make")
message(STATUS "To install: make fused_moe_package")
message(STATUS "For FL: export FUSED_MOE_FLAGSHIP_LIB=<build-dir>/libfused_moe_flagship.so")
