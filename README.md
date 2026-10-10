# optimization-kernels

[![Support Open Work](https://img.shields.io/badge/Support-Open%20Work-0A0A0A?style=flat&logo=github)](https://github.com/brandonhimpfen/support) 
[![GitHub Sponsor](https://srv-cdn.himpfen.io/badges/github/github-flat.svg)](https://github.com/sponsors/brandonhimpfen) 
[![Buy Me a Coffee](https://srv-cdn.himpfen.io/badges/buymeacoffee/buymeacoffee-flat.svg)](https://buymeacoffee.com/brandonhimpfen) 
[![Ko-Fi](https://srv-cdn.himpfen.io/badges/kofi/kofi-flat.svg)](https://ko-fi.com/brandonhimpfen) 
[![PayPal](https://srv-cdn.himpfen.io/badges/paypal/paypal-flat.svg)](https://paypal.me/brandonhimpfen)

C++ kernels and utilities for quantization and inference optimization experiments.

`optimization-kernels` is a small, focused C++ project for exploring low-level building blocks used in AI inference optimization. Version 0.1 provides int8 quantization helpers, dequantization helpers, a simple float32 matrix multiplication kernel, an int8 inference-style matrix multiplication path, a benchmark harness, and a comparison example.

This project is intentionally not a full inference engine. It is a practical foundation for learning, testing, and extending optimization kernels.

## Features

- int8 quantization parameter calculation
- float32 to int8 quantization helpers
- int8 to float32 dequantization helpers
- simple float32 matrix multiplication kernel
- simple int8 accumulation and dequantized matrix multiplication path
- benchmark harness for float32 versus int8-style operations
- example comparing float32 and quantized outputs
- lightweight CMake build
- dependency-free C++17 implementation

## Project structure

```text
optimization-kernels/
├── include/
│   └── optimization_kernels/
├── src/
├── examples/
├── benchmarks/
├── tests/
├── docs/
├── CMakeLists.txt
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run the example

```bash
./build/compare_f32_i8
```

On Windows with Visual Studio generators, the executable may be under a configuration directory such as:

```bash
./build/Debug/compare_f32_i8.exe
```

## Run tests

```bash
ctest --test-dir build
```

## Run the benchmark

```bash
./build/matmul_benchmark
```

The benchmark is intentionally simple. It is useful for comparing paths inside this project, but it should not be treated as a production-grade performance claim.

## Example usage

```cpp
#include "optimization_kernels/optimization_kernels.hpp"

#include <vector>

int main() {
    std::vector<float> values = {-1.0f, 0.0f, 1.0f};

    auto params = optimization_kernels::calculate_quantization_params(values);
    auto q = optimization_kernels::quantize_vector(values, params);
    auto restored = optimization_kernels::dequantize_vector(q, params);

    return 0;
}
```

## Current limitations

- Kernels are simple reference implementations.
- No SIMD intrinsics are included in v0.1.
- No GPU acceleration is included.
- The int8 matrix multiplication path dequantizes inside the loop for readability.
- Per-tensor quantization is supported; per-channel quantization is not yet included.
- The benchmark is basic and does not replace a full profiling suite.

## Roadmap

Potential future work:

- SIMD optimized kernels
- per-channel quantization
- tiled matrix multiplication
- cache-aware loop layouts
- activation quantization examples
- benchmark output in JSON or CSV
- comparison against external BLAS or inference runtimes
- CI workflow for Linux, macOS, and Windows

## License

MIT License.
