# ML Kernel Roofline

A microbenchmark suite for studying machine learning kernel performance on Apple M2 Pro. The initial scope compares matrix multiplication and elementwise operations across the CPU and Apple GPU, examines input-size and precision effects, and uses Roofline analysis to interpret the results.

## Initial kernels

- Matrix multiplication (GEMM)
- Elementwise addition
- Elementwise multiplication

The benchmark runner compares PyTorch CPU and GPU operations with native CPU and custom Metal kernels. The CPU implementation includes a parallel naive GEMM and a 32×32 cache-tiled GEMM. The Metal implementation includes a naive GEMM baseline, a 16×16 tiled GEMM, and a fused elementwise multiply-add kernel. The runner compiles its C++ and Swift helpers automatically with `clang++` and `swiftc`.

FP16 support depends on the operation and device. Unsupported combinations are reported and skipped, so use the MPS device for the primary GPU FP16 comparison.

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

For a focused FP32 optimization comparison:

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

Open `notebooks/roofline_analysis.ipynb` to load the optimization comparison CSV, plot performance against input size, and compare GEMM and fused multiply-add on a Roofline plot. See `docs/metrics.md` for metric definitions and assumptions.

## Measurement notes

Arithmetic intensity and bandwidth use modeled traffic, not physical DRAM counters. The notebook uses the M2 Pro's published 200 GB/s unified-memory bandwidth as a reference ceiling and the best measured GEMM throughput as an empirical compute ceiling. Benchmark results vary with problem size, precision, and system state.

Some naive GEMM points may appear above the 200 GB/s line because their modeled buffer reads can be served by GPU caches. This is a limit of the simple traffic model, not evidence of DRAM bandwidth above Apple's specification. The report should discuss these points as cache effects; hardware memory counters would be needed for a calibrated DRAM Roofline.
