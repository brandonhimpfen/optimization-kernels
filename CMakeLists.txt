cmake_minimum_required(VERSION 3.16)

project(optimization-kernels VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OPTIMIZATION_KERNELS_BUILD_EXAMPLES "Build examples" ON)
option(OPTIMIZATION_KERNELS_BUILD_BENCHMARKS "Build benchmarks" ON)
option(OPTIMIZATION_KERNELS_BUILD_TESTS "Build tests" ON)

add_library(optimization_kernels
    src/quantization.cpp
    src/matmul.cpp
)

add_library(optimization_kernels::optimization_kernels ALIAS optimization_kernels)

target_include_directories(optimization_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(optimization_kernels PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(OPTIMIZATION_KERNELS_BUILD_EXAMPLES)
    add_executable(compare_f32_i8 examples/compare_f32_i8.cpp)
    target_link_libraries(compare_f32_i8 PRIVATE optimization_kernels::optimization_kernels)
endif()

if(OPTIMIZATION_KERNELS_BUILD_BENCHMARKS)
    add_executable(matmul_benchmark benchmarks/matmul_benchmark.cpp)
    target_link_libraries(matmul_benchmark PRIVATE optimization_kernels::optimization_kernels)
endif()

if(OPTIMIZATION_KERNELS_BUILD_TESTS)
    enable_testing()

    add_executable(test_quantization tests/test_quantization.cpp)
    target_link_libraries(test_quantization PRIVATE optimization_kernels::optimization_kernels)
    add_test(NAME test_quantization COMMAND test_quantization)

    add_executable(test_matmul tests/test_matmul.cpp)
    target_link_libraries(test_matmul PRIVATE optimization_kernels::optimization_kernels)
    add_test(NAME test_matmul COMMAND test_matmul)
endif()
