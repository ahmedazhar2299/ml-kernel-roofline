# Metrics and assumptions

Each benchmark row records the operation, input size, device, precision, implementation variant, median execution time, throughput, modeled bytes moved, estimated bandwidth, arithmetic intensity, and maximum absolute error.

- **GEMM operations:** `2 × N³` floating-point operations for an `N × N` matrix product.
- **Elementwise operations:** one operation per element for addition or multiplication. These are reported as GFLOP/s for consistency, with the operation count stated explicitly.
- **GEMM traffic model:** The naive kernel counts two input reads per multiply and one output write, or approximately `(2N³ + N²) × 4` bytes. The 16×16 tiled kernel counts approximately `(2N² ceil(N/16) + N²) × 4` bytes. The PyTorch baseline uses an idealized one-read-per-input estimate of `3N² × 4` bytes.
- **Fused multiply-add traffic model:** The eager PyTorch baseline counts six vector transfers per element because it materializes an intermediate. The fused Metal kernel counts three input reads and one output write, or four transfers per element.
- **Elementwise traffic model:** Addition and multiplication count two input reads and one output write.
- **Estimated bandwidth:** modeled bytes divided by median execution time.
- **Arithmetic intensity:** operation count divided by modeled bytes. A fused multiply-add counts as two floating-point operations per element.
- **Numerical error:** The PyTorch results are compared with an FP32 CPU reference. The tiled Metal GEMM is compared with the naive Metal GEMM using identical inputs; the fused Metal kernel is compared with a CPU reference.

These estimates help compare implementations and place points on a Roofline plot, but they are modeled traffic, not hardware-counter measurements. Cache behavior, compiler transformations, and launch overhead are not measured. The PyTorch GEMM estimate is intentionally idealized, so it should not be interpreted as measured device traffic.

The notebook uses the best measured GEMM throughput for each device and precision as an empirical compute ceiling unless a sourced compute peak is entered. The intersection of that ceiling with the 200 GB/s memory-bandwidth reference defines the ridge point. Since the compute ceiling is empirical and the bandwidth is a published peak, the plot is an illustrative Roofline model rather than a calibrated hardware Roofline.
