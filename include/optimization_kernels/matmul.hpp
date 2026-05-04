#pragma once

#include <cstdint>
#include <vector>

#include "optimization_kernels/quantization.hpp"

namespace optimization_kernels {

std::vector<float> matmul_f32(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

std::vector<int32_t> matmul_i8_accum(
    const std::vector<int8_t>& a,
    const std::vector<int8_t>& b,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

std::vector<float> matmul_i8_dequantized(
    const std::vector<int8_t>& a,
    const std::vector<int8_t>& b,
    const QuantizationParams& a_params,
    const QuantizationParams& b_params,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

} // namespace optimization_kernels
