# Initial optimization comparison

Measured on an Apple M2 Pro using FP32, with five warmup iterations and twenty timed repetitions per case. CPU measurements use PyTorch CPU operations. GPU measurements include PyTorch MPS and custom Metal kernels. The CSV contains all measured rows.

| GEMM N | Naive Metal | Tiled Metal | PyTorch MPS | PyTorch CPU | Tiled speedup over naive |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.525 ms | 0.206 ms | 0.216 ms | 0.006 ms | 2.55× |
| 256 | 0.257 ms | 0.203 ms | 0.213 ms | 0.030 ms | 1.27× |
| 512 | 0.876 ms | 0.566 ms | 0.258 ms | 0.124 ms | 1.55× |
| 1024 | 6.251 ms | 3.526 ms | 0.666 ms | 0.808 ms | 1.77× |

| Vector length | PyTorch CPU eager | PyTorch MPS eager | Fused Metal | Fused speedup over MPS eager |
|---:|---:|---:|---:|---:|
| 65,536 | 0.048 ms | 0.463 ms | 0.154 ms | 3.01× |
| 262,144 | 0.091 ms | 0.493 ms | 0.167 ms | 2.95× |
| 1,048,576 | 0.119 ms | 0.619 ms | 0.180 ms | 3.43× |

Tiled Metal GEMM was faster than the naive Metal version at all tested sizes. PyTorch's built-in GEMM was comparable to the tiled kernel for 128 and 256, and faster at 512 and 1024. The CPU PyTorch baseline was faster than the custom Metal GEMM at 128, 256, and 512; at 1024, PyTorch MPS was fastest.

The fused Metal operation was about three times faster than two eager PyTorch MPS operations in these runs. CPU eager execution was faster than both GPU approaches for the tested vector sizes. Fusion reduced modeled traffic from six transfers per element to four, raising arithmetic intensity from 0.083 to 0.125 operations/byte.

The model places naive GEMM at about 0.25 operations/byte and tiled GEMM around 3.8 to 4.0 operations/byte. Tiling reuses values in threadgroup memory, reducing modeled global loads. PyTorch GEMM uses an idealized one-read-per-input estimate, so its arithmetic intensity is higher; it should not be interpreted as a measurement of PyTorch's physical memory traffic.

These are initial measurements, not final performance claims. Short workloads are sensitive to timer resolution, dispatch overhead, system load, and thermal state. The Roofline uses modeled traffic and a published 200 GB/s memory reference. Some naive GEMM points exceed that line because GPU caches can serve repeated buffer reads. A calibrated memory Roofline would require physical memory counters or a separate cache and streaming-bandwidth characterization.
