# Metrics and assumptions

Each benchmark row records the operation, input size, device, precision, median execution time, throughput, logical bytes moved, estimated bandwidth, arithmetic intensity, and maximum absolute error against an FP32 reference.

- **GEMM operations:** `2 × N³` floating-point operations for an `N × N` matrix product.
- **Elementwise operations:** one operation per element for addition or multiplication. These are reported as GFLOP/s for consistency, with the operation count stated explicitly.
- **Logical traffic:** GEMM counts one read of each input matrix and one write of the output. Elementwise kernels count two input reads and one output write. This is a lower-bound traffic estimate and does not capture cache-line effects or physical memory traffic.
- **Estimated bandwidth:** logical bytes divided by median execution time.
- **Arithmetic intensity:** operation count divided by logical bytes moved.
- **Numerical error:** maximum absolute difference from the corresponding FP32 CPU result for the same inputs.

These estimates are useful for comparing runs and placing points on a Roofline plot, but they are not hardware-counter measurements. In particular, measured bandwidth can appear above or below expectations when traffic is served from cache or when launch and timing overhead dominate small workloads.
