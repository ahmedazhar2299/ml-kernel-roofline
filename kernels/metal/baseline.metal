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

// Tiled GEMM reuses loaded tiles from threadgroup memory. The host dispatches
// 16x16 threadgroups, so each group computes one 16x16 output tile.
kernel void matmul_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 16;
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    const uint row = group.y * TILE + tid.y;
    const uint column = group.x * TILE + tid.x;
    float sum = 0.0f;

    for (uint base = 0; base < n; base += TILE) {
        const uint a_column = base + tid.x;
        const uint b_row = base + tid.y;
        tile_a[tid.y][tid.x] = (row < n && a_column < n) ? a[row * n + a_column] : 0.0f;
        tile_b[tid.y][tid.x] = (b_row < n && column < n) ? b[b_row * n + column] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) {
            sum += tile_a[tid.y][k] * tile_b[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < n && column < n) c[row * n + column] = sum;
}

// One fused multiply-add pass replaces separate multiply and add kernels.
kernel void fused_mul_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = a[index] * b[index] + c[index];
}
