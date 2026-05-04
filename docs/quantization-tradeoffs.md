# Quantization Tradeoffs

Quantization is a technique for representing values with lower-precision data types. In AI inference, this often means converting float32 weights or activations into int8 values. The goal is usually to reduce memory usage, improve cache efficiency, and make inference faster on hardware that supports low-precision operations.

## Why quantization helps

Float32 values use four bytes per number. Int8 values use one byte per number. For large models, this can significantly reduce memory bandwidth requirements and storage size.

Lower memory usage can improve practical inference performance because many workloads are limited not only by arithmetic speed, but also by how quickly data can move through memory and cache.

## What can go wrong

Quantization introduces approximation. A continuous float32 range must be mapped onto a much smaller set of integer values. This can create rounding error, clipping, and loss of small details.

The effect depends on the model, the distribution of values, the quantization scheme, and the tolerance of the application. Some models remain accurate after int8 quantization. Others need calibration, retraining, or mixed precision.

## Scale and zero point

This v0.1 project uses a simple affine quantization model:

```text
quantized = round(real_value / scale) + zero_point
real_value = (quantized - zero_point) * scale
```

The scale controls the size of each quantized step. The zero point allows the integer range to represent real values that are not centered around zero.

## Per-tensor versus per-channel

This version uses per-tensor quantization, where one scale and zero point apply to the entire vector or tensor.

Per-channel quantization can be more accurate for model weights because each output channel can have its own scale. That is a good candidate for a future release.

## Speed versus clarity

The v0.1 int8 matrix multiplication path prioritizes readability. It demonstrates the relationship between quantized inputs and dequantized output, but it is not yet a highly optimized inference kernel.

Future versions could add tiling, SIMD instructions, cache-aware memory access, and fused dequantization strategies.

## Practical takeaway

Quantization is useful when the reduced precision produces acceptable error while improving memory usage, deployment cost, or runtime performance. It should be measured, not assumed.
