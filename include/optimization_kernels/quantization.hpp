#pragma once

#include <cstdint>
#include <vector>

namespace optimization_kernels {

struct QuantizationParams {
    float scale{1.0f};
    int32_t zero_point{0};
};

QuantizationParams calculate_quantization_params(const std::vector<float>& values);

int8_t quantize_value(float value, const QuantizationParams& params);
float dequantize_value(int8_t value, const QuantizationParams& params);

std::vector<int8_t> quantize_vector(const std::vector<float>& values, const QuantizationParams& params);
std::vector<float> dequantize_vector(const std::vector<int8_t>& values, const QuantizationParams& params);

} // namespace optimization_kernels
