# Initial optimization comparison

Measured on an Apple M2 Pro GPU using FP32, with two warmup iterations and six timed repetitions per case. The CSV contains the measurements used for the plots.

| Kernel | Size | Baseline | Optimized | Speedup | Arithmetic intensity |
|---|---:|---:|---:|---:|---:|
| GEMM | 128 | 0.541 ms, naive Metal | 0.469 ms, tiled Metal | 1.15× | 0.25 → 3.76 ops/byte |
| GEMM | 256 | 0.755 ms, naive Metal | 0.638 ms, tiled Metal | 1.18× | 0.25 → 3.88 ops/byte |
| GEMM | 512 | 0.876 ms, naive Metal | 0.579 ms, tiled Metal | 1.51× | 0.25 → 3.94 ops/byte |
| GEMM | 1024 | 6.264 ms, naive Metal | 3.513 ms, tiled Metal | 1.78× | 0.25 → 3.97 ops/byte |
| Fused multiply-add | 65,536 | 0.446 ms, eager PyTorch | 0.150 ms, fused Metal | 2.97× | 0.083 → 0.125 ops/byte |
| Fused multiply-add | 262,144 | 0.484 ms, eager PyTorch | 0.162 ms, fused Metal | 2.98× | 0.083 → 0.125 ops/byte |
| Fused multiply-add | 1,048,576 | 0.648 ms, eager PyTorch | 0.198 ms, fused Metal | 3.28× | 0.083 → 0.125 ops/byte |

The tiled kernel improved on the naive Metal kernel at every tested matrix size. PyTorch's optimized GEMM remained faster than both custom kernels. Fusion reduced the modeled vector traffic from six transfers per element to four and improved measured latency relative to two eager PyTorch operations. The small vector operations still include dispatch and synchronization costs, so the speedups do not isolate memory bandwidth alone.

The Roofline uses modeled traffic and a published 200 GB/s memory reference. Some naive GEMM points exceed that line because GPU caches can serve repeated buffer reads. These points should be discussed as evidence that the simple model does not represent off-chip traffic for every kernel. A calibrated memory Roofline would require physical memory counters or a separate cache and streaming-bandwidth characterization.
