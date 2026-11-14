#include <metal_stdlib>
using namespace metal;

// Naive and four-wide optimized elementwise kernels.
kernel void elementwise_add_naive(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) c[index] = a[index] + b[index];
}

kernel void elementwise_multiply_naive(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) c[index] = a[index] * b[index];
}

#define ELEMENTWISE_OPTIMIZED(NAME, EXPR) \
kernel void NAME( \
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], \
    device float* c [[buffer(2)]], constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]]) { \
    const uint base = index * 4; \
    for (uint lane = 0; lane < 4; ++lane) { \
        const uint i = base + lane; \
        if (i < count) c[i] = (EXPR); \
    } \
}

ELEMENTWISE_OPTIMIZED(elementwise_add_optimized, a[i] + b[i])
ELEMENTWISE_OPTIMIZED(elementwise_multiply_optimized, a[i] * b[i])

#undef ELEMENTWISE_OPTIMIZED

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

// The optimized GEMM reuses loaded tiles from threadgroup memory. The host dispatches
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

// NCHW input, OIHW weights, stride one, and one-pixel zero padding.
kernel void conv2d_naive(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint CIN = 8, COUT = 16, K = 3;
    const uint width = height;
    const uint total = COUT * height * width;
    if (index >= total) return;
    const uint ox = index % width;
    const uint oy = (index / width) % height;
    const uint oc = index / (width * height);
    float sum = 0.0f;
    for (uint ic = 0; ic < CIN; ++ic) {
        for (uint ky = 0; ky < K; ++ky) {
            for (uint kx = 0; kx < K; ++kx) {
                const int iy = int(oy) + int(ky) - 1;
                const int ix = int(ox) + int(kx) - 1;
                if (iy < 0 || iy >= int(height) || ix < 0 || ix >= int(width)) continue;
                const uint x_index = ic * height * width + uint(iy) * width + uint(ix);
                const uint w_index = ((oc * CIN + ic) * K + ky) * K + kx;
                sum += input[x_index] * weights[w_index];
            }
        }
    }
    output[index] = sum;
}

// Each thread reuses one filter value across four adjacent output pixels.
kernel void conv2d_optimized(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    uint block [[thread_position_in_grid]]) {
    constexpr uint CIN = 8, COUT = 16, K = 3, TILE_X = 4;
    const uint width = height;
    const uint blocks_per_row = (width + TILE_X - 1) / TILE_X;
    const uint total = COUT * height * blocks_per_row;
    if (block >= total) return;
    const uint bx = block % blocks_per_row;
    const uint oy = (block / blocks_per_row) % height;
    const uint oc = block / (blocks_per_row * height);
    const uint first_x = bx * TILE_X;
    float sums[TILE_X] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ic = 0; ic < CIN; ++ic) {
        for (uint ky = 0; ky < K; ++ky) {
            const int iy = int(oy) + int(ky) - 1;
            if (iy < 0 || iy >= int(height)) continue;
            for (uint kx = 0; kx < K; ++kx) {
                const float weight = weights[((oc * CIN + ic) * K + ky) * K + kx];
                for (uint lane = 0; lane < TILE_X; ++lane) {
                    const uint ox = first_x + lane;
                    const int ix = int(ox) + int(kx) - 1;
                    if (ox < width && ix >= 0 && ix < int(width)) {
                        const uint x_index = ic * height * width + uint(iy) * width + uint(ix);
                        sums[lane] += input[x_index] * weight;
                    }
                }
            }
        }
    }
    for (uint lane = 0; lane < TILE_X; ++lane) {
        const uint ox = first_x + lane;
        if (ox < width) output[oc * height * width + oy * width + ox] = sums[lane];
    }
}

kernel void softmax_naive(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& columns [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const uint offset = row * columns;
    float max_value = -INFINITY;
    for (uint col = 0; col < columns; ++col) max_value = max(max_value, input[offset + col]);
    float denominator = 0.0f;
    for (uint col = 0; col < columns; ++col) denominator += exp(input[offset + col] - max_value);
    for (uint col = 0; col < columns; ++col)
        output[offset + col] = exp(input[offset + col] - max_value) / denominator;
}

kernel void softmax_optimized(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& columns [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float partial[256];
    const uint row = group.y;
    if (row >= rows) return;
    const uint offset = row * columns;
    float local_max = -INFINITY;
    for (uint col = lane; col < columns; col += 256) local_max = max(local_max, input[offset + col]);
    partial[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_value = partial[0];
    float local_sum = 0.0f;
    for (uint col = lane; col < columns; col += 256) {
        const float value = exp(input[offset + col] - max_value);
        output[offset + col] = value;
        local_sum += value;
    }
    partial[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denominator = partial[0];
    for (uint col = lane; col < columns; col += 256) output[offset + col] /= denominator;
}

kernel void layernorm_naive(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& columns [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const uint offset = row * columns;
    float mean = 0.0f;
    for (uint col = 0; col < columns; ++col) mean += input[offset + col];
    mean /= float(columns);
    float variance = 0.0f;
    for (uint col = 0; col < columns; ++col) {
        const float delta = input[offset + col] - mean;
        variance += delta * delta;
    }
    variance /= float(columns);
    const float inv_std = rsqrt(variance + 1e-5f);
    for (uint col = 0; col < columns; ++col)
        output[offset + col] = (input[offset + col] - mean) * inv_std;
}

kernel void layernorm_optimized(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant uint& columns [[buffer(2)]], constant uint& rows [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float partial[256];
    const uint row = group.y;
    if (row >= rows) return;
    const uint offset = row * columns;
    float local_sum = 0.0f;
    for (uint col = lane; col < columns; col += 256) local_sum += input[offset + col];
    partial[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partial[0] / float(columns);
    float local_squares = 0.0f;
    for (uint col = lane; col < columns; col += 256) {
        const float delta = input[offset + col] - mean;
        local_squares += delta * delta;
    }
    partial[lane] = local_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_std = rsqrt(partial[0] / float(columns) + 1e-5f);
    for (uint col = lane; col < columns; col += 256)
        output[offset + col] = (input[offset + col] - mean) * inv_std;
}
