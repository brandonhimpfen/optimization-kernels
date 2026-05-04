#include "optimization_kernels/optimization_kernels.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace {
std::vector<float> random_vector(std::size_t size) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> values(size);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

template <typename Fn>
double time_ms(Fn&& fn, int iterations) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}
}

int main() {
    const std::size_t m = 64;
    const std::size_t k = 64;
    const std::size_t n = 64;
    const int iterations = 50;

    const auto a = random_vector(m * k);
    const auto b = random_vector(k * n);

    const auto a_params = optimization_kernels::calculate_quantization_params(a);
    const auto b_params = optimization_kernels::calculate_quantization_params(b);
    const auto a_i8 = optimization_kernels::quantize_vector(a, a_params);
    const auto b_i8 = optimization_kernels::quantize_vector(b, b_params);

    volatile float sink = 0.0f;

    const auto f32_ms = time_ms([&]() {
        const auto result = optimization_kernels::matmul_f32(a, b, m, k, n);
        sink += result[0];
    }, iterations);

    const auto i8_ms = time_ms([&]() {
        const auto result = optimization_kernels::matmul_i8_dequantized(a_i8, b_i8, a_params, b_params, m, k, n);
        sink += result[0];
    }, iterations);

    std::cout << "Matrix multiplication benchmark\n";
    std::cout << "Shape: " << m << "x" << k << " * " << k << "x" << n << '\n';
    std::cout << "Iterations: " << iterations << '\n';
    std::cout << "f32 average: " << f32_ms << " ms\n";
    std::cout << "i8 dequantized average: " << i8_ms << " ms\n";
    std::cout << "sink: " << sink << '\n';

    return 0;
}
