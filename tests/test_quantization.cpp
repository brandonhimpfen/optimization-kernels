#include "optimization_kernels/optimization_kernels.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> values = {-1.0f, 0.0f, 1.0f};
    const auto params = optimization_kernels::calculate_quantization_params(values);
    const auto quantized = optimization_kernels::quantize_vector(values, params);
    const auto dequantized = optimization_kernels::dequantize_vector(quantized, params);

    assert(quantized.size() == values.size());
    assert(dequantized.size() == values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        assert(std::fabs(values[i] - dequantized[i]) < 0.02f);
    }

    std::cout << "test_quantization passed\n";
    return 0;
}
