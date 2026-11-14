# ML Kernel Roofline

A microbenchmark suite for studying machine learning kernel performance on Apple M2 Pro. The initial scope compares matrix multiplication and elementwise operations across the CPU and Apple GPU, examines input-size effects, and uses Roofline analysis to interpret the results.

## Initial kernels

- Matrix multiplication (GEMM)
- Elementwise addition
- Elementwise multiplication
- Fused multiply-add
- 2D convolution
- Softmax
- LayerNorm

The benchmark runner compares PyTorch CPU and GPU operations with custom CPU and Metal kernels for GEMM, add, multiply, fused multiply-add, 2D convolution, softmax, and LayerNorm. Every operation has a naive and optimized custom CPU and Metal variant, alongside PyTorch baselines. GEMM's optimized path uses tiling; elementwise optimized paths process several values per iteration; fused multiply-add compares two separate operations against one fused pass. The benchmark currently uses FP32 for all implementations. The runner compiles its C++ and Swift helpers automatically with `clang++` and `swiftc`.

## Setup

Use Python 3.10 or later on an Apple silicon Mac with macOS 14 or later. Install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run benchmarks

Run CPU and MPS baselines, plus custom Metal kernels when MPS is available:

```sh
python benchmarks/run_benchmarks.py --device both --output results/measurements.csv
```

For a focused kernel comparison:

```sh
python benchmarks/run_benchmarks.py \
  --operations matmul,fused_mul_add \
  --matmul-sizes 128,256,512,1024 \
  --elementwise-sizes 65536,262144,1048576 \
  --precisions fp32 \
  --device both \
  --warmup 2 --repeats 6 \
  --output results/optimization_comparison.csv
```

Open `notebooks/roofline_analysis.ipynb` to select a kernel. `SHOW_NAIVE`, `SHOW_OPTIMIZED`, and `SHOW_PYTORCH` control which implementations appear. The comparison currently uses FP32. See `docs/metrics.md` for metric definitions and assumptions.

## Measurement notes

The benchmark's per-kernel traffic estimates are models, not physical DRAM counters. For GEMM, the notebook plots arithmetic intensity from compulsory input/output tensor traffic so it is comparable to the M2 Pro's published 200 GB/s unified-memory bandwidth reference. Its compute ceiling is the best measured GEMM throughput for the selected device and precision. The plot remains illustrative because it combines modeled traffic, published bandwidth, and an empirical compute ceiling. Benchmark results vary with problem size, precision, and system state.
