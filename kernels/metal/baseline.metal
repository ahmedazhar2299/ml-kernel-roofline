#include <metal_stdlib>
using namespace metal;

// Straightforward elementwise baselines. These are source kernels for the
// custom Metal implementation phase; the initial Python runner benchmarks
// PyTorch CPU and MPS operations.
kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) c[index] = a[index] + b[index];
}

kernel void elementwise_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) c[index] = a[index] * b[index];
}

// Naive square GEMM baseline. Each output element is computed by one thread.
kernel void matmul_naive(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint row = position.y;
    const uint column = position.x;
    if (row >= n || column >= n) return;
    float sum = 0.0f;
    for (uint k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + column];
    }
    c[row * n + column] = sum;
}
