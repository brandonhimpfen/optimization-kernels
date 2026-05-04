#include "optimization_kernels/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace optimization_kernels {

QuantizationParams calculate_quantization_params(const std::vector<float>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate quantization parameters for an empty vector.");
    }

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const float min_value = *min_it;
    const float max_value = *max_it;

    constexpr int32_t qmin = -128;
    constexpr int32_t qmax = 127;

    if (std::fabs(max_value - min_value) < std::numeric_limits<float>::epsilon()) {
        return QuantizationParams{1.0f, 0};
    }

    const float scale = (max_value - min_value) / static_cast<float>(qmax - qmin);
    const float zero_point_from_min = static_cast<float>(qmin) - min_value / scale;
    const auto zero_point = static_cast<int32_t>(std::round(std::clamp(zero_point_from_min, static_cast<float>(qmin), static_cast<float>(qmax))));

    return QuantizationParams{scale, zero_point};
}

int8_t quantize_value(float value, const QuantizationParams& params) {
    if (params.scale <= 0.0f) {
        throw std::invalid_argument("Quantization scale must be positive.");
    }

    constexpr int32_t qmin = -128;
    constexpr int32_t qmax = 127;

    const auto transformed = static_cast<int32_t>(std::round(value / params.scale)) + params.zero_point;
    const auto clamped = std::clamp(transformed, qmin, qmax);
    return static_cast<int8_t>(clamped);
}

float dequantize_value(int8_t value, const QuantizationParams& params) {
    if (params.scale <= 0.0f) {
        throw std::invalid_argument("Quantization scale must be positive.");
    }

    return (static_cast<int32_t>(value) - params.zero_point) * params.scale;
}

std::vector<int8_t> quantize_vector(const std::vector<float>& values, const QuantizationParams& params) {
    std::vector<int8_t> output;
    output.reserve(values.size());

    for (float value : values) {
        output.push_back(quantize_value(value, params));
    }

    return output;
}

std::vector<float> dequantize_vector(const std::vector<int8_t>& values, const QuantizationParams& params) {
    std::vector<float> output;
    output.reserve(values.size());

    for (int8_t value : values) {
        output.push_back(dequantize_value(value, params));
    }

    return output;
}

} // namespace optimization_kernels
