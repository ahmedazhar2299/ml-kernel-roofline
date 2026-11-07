# CPU and GPU GEMM comparison

Measured on an Apple M2 Pro using FP32, with five warmup iterations and twenty timed repetitions per case. CPU custom kernels are native C++ with multiple worker threads. GPU custom kernels are Metal. PyTorch provides library baselines on CPU and GPU. Raw measurements are in `optimization_comparison.csv`.

## CPU GEMM

| N | Naive C++ | Tiled C++ | PyTorch CPU | Tiled speedup over naive |
|---:|---:|---:|---:|---:|
| 128 | 0.417 ms | 0.137 ms | 0.006 ms | 3.05× |
| 256 | 2.853 ms | 0.502 ms | 0.031 ms | 5.68× |
| 512 | 28.947 ms | 3.474 ms | 0.205 ms | 8.33× |
| 1024 | 223.532 ms | 23.976 ms | 0.788 ms | 9.32× |

## GPU GEMM

| N | Naive Metal | Tiled Metal | PyTorch GPU | Tiled speedup over naive |
|---:|---:|---:|---:|---:|
| 128 | 0.542 ms | 0.500 ms | 0.573 ms | 1.08× |
| 256 | 0.508 ms | 0.365 ms | 0.195 ms | 1.39× |
| 512 | 1.238 ms | 0.736 ms | 0.297 ms | 1.68× |
| 1024 | 6.326 ms | 3.515 ms | 1.022 ms | 1.80× |

Tiling improved the custom naive GEMM on both CPU and GPU. Its benefit increased with matrix size. PyTorch's tuned CPU and GPU kernels were substantially faster than our custom implementations at larger sizes.

## Fused multiply-add

| Vector length | PyTorch CPU eager | PyTorch GPU eager | Fused Metal | Metal speedup over GPU eager |
|---:|---:|---:|---:|---:|
| 65,536 | 0.065 ms | 0.460 ms | 0.160 ms | 2.87× |
| 262,144 | 0.079 ms | 0.507 ms | 0.161 ms | 3.14× |
| 1,048,576 | 0.113 ms | 0.332 ms | 0.199 ms | 1.67× |

Fusion reduces modeled traffic from six transfers per element to four, raising arithmetic intensity from 0.083 to 0.125 operations/byte. The CPU eager implementation was fastest at these vector sizes. The fused Metal kernel was faster than two eager GPU operations in this run.

The CPU naive and tiled kernels use the same number of worker threads for each size, up to ten workers. PyTorch used six CPU threads in this run. The CPU custom timing includes thread creation and join overhead on every repetition.

Traffic values used for the Roofline plot are modeled, not hardware-counter readings. PyTorch GEMM uses idealized one-read-per-input traffic. Naive GEMM counts repeated buffer reads, some of which may be served by cache. This is why some points can appear above the published 200 GB/s memory-bandwidth reference. Treat the Roofline as an illustrative comparison, not a calibrated off-chip traffic model.
