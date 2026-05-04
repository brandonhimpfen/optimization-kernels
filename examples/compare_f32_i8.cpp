#include "optimization_kernels/optimization_kernels.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::size_t m = 2;
    const std::size_t k = 3;
    const std::size_t n = 2;

    const std::vector<float> a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const std::vector<float> b = {
        0.5f, 1.0f,
        1.5f, 2.0f,
        2.5f, 3.0f
    };

    const auto f32_result = optimization_kernels::matmul_f32(a, b, m, k, n);

    const auto a_params = optimization_kernels::calculate_quantization_params(a);
    const auto b_params = optimization_kernels::calculate_quantization_params(b);

    const auto a_i8 = optimization_kernels::quantize_vector(a, a_params);
    const auto b_i8 = optimization_kernels::quantize_vector(b, b_params);
    const auto i8_result = optimization_kernels::matmul_i8_dequantized(a_i8, b_i8, a_params, b_params, m, k, n);

    std::cout << "Float32 vs int8-dequantized matrix multiplication\n\n";

    for (std::size_t i = 0; i < f32_result.size(); ++i) {
        const float error = std::fabs(f32_result[i] - i8_result[i]);
        std::cout << "index " << i
                  << " | f32=" << f32_result[i]
                  << " | i8_dequantized=" << i8_result[i]
                  << " | abs_error=" << error << '\n';
    }

    return 0;
}
