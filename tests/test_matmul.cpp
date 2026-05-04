#include "optimization_kernels/optimization_kernels.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> a = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    const std::vector<float> b = {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    const auto result = optimization_kernels::matmul_f32(a, b, 2, 2, 2);

    assert(result.size() == 4);
    assert(std::fabs(result[0] - 19.0f) < 0.001f);
    assert(std::fabs(result[1] - 22.0f) < 0.001f);
    assert(std::fabs(result[2] - 43.0f) < 0.001f);
    assert(std::fabs(result[3] - 50.0f) < 0.001f);

    std::cout << "test_matmul passed\n";
    return 0;
}
