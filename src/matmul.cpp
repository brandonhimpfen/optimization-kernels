#include "optimization_kernels/matmul.hpp"

#include <stdexcept>

namespace optimization_kernels {

namespace {
void validate_shapes(std::size_t a_size, std::size_t b_size, std::size_t m, std::size_t k, std::size_t n) {
    if (a_size != m * k) {
        throw std::invalid_argument("Matrix A size does not match m * k.");
    }
    if (b_size != k * n) {
        throw std::invalid_argument("Matrix B size does not match k * n.");
    }
}
}

std::vector<float> matmul_f32(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    validate_shapes(a.size(), b.size(), m, k, n);

    std::vector<float> c(m * n, 0.0f);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }

    return c;
}

std::vector<int32_t> matmul_i8_accum(
    const std::vector<int8_t>& a,
    const std::vector<int8_t>& b,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    validate_shapes(a.size(), b.size(), m, k, n);

    std::vector<int32_t> c(m * n, 0);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            int32_t sum = 0;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += static_cast<int32_t>(a[row * k + inner]) * static_cast<int32_t>(b[inner * n + col]);
            }
            c[row * n + col] = sum;
        }
    }

    return c;
}

std::vector<float> matmul_i8_dequantized(
    const std::vector<int8_t>& a,
    const std::vector<int8_t>& b,
    const QuantizationParams& a_params,
    const QuantizationParams& b_params,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    validate_shapes(a.size(), b.size(), m, k, n);

    std::vector<float> c(m * n, 0.0f);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (std::size_t inner = 0; inner < k; ++inner) {
                const float av = dequantize_value(a[row * k + inner], a_params);
                const float bv = dequantize_value(b[inner * n + col], b_params);
                sum += av * bv;
            }
            c[row * n + col] = sum;
        }
    }

    return c;
}

} // namespace optimization_kernels
