# ML Kernel Roofline

A microbenchmark suite for studying machine learning kernel performance on Apple M2 Pro. The initial scope compares matrix multiplication and elementwise operations across the CPU and Apple GPU, examines input-size and precision effects, and uses Roofline analysis to interpret the results.

## Initial kernels

- Matrix multiplication (GEMM)
- Elementwise addition
- Elementwise multiplication

The benchmark runner uses PyTorch CPU and MPS operations as initial baselines. Metal shader sources are in `kernels/metal`; the next implementation step is to connect custom Metal kernels to the runner so that the project compares framework baselines with kernels we implement.

FP16 support depends on the operation and device. Unsupported combinations are reported and skipped, so use the MPS device for the primary GPU FP16 comparison.

## Setup

Use Python 3.10 or later on an Apple silicon Mac with macOS 14 or later. Install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run benchmarks

Run both CPU and MPS when MPS is available:

```sh
python benchmarks/run_benchmarks.py --device both --output results/measurements.csv
```

Select operations, sizes, and precision as needed:

```sh
python benchmarks/run_benchmarks.py \
  --operations matmul,add,multiply \
  --sizes 128,256,512,1024 \
  --precisions fp32,fp16 \
  --device both \
  --output results/measurements.csv
```

Open `notebooks/roofline_analysis.ipynb` to load the CSV, plot performance against input size, and create a Roofline plot. See `docs/metrics.md` for metric definitions and assumptions.

## Measurement notes

The initial arithmetic-intensity and bandwidth metrics use logical tensor traffic. They do not measure physical DRAM traffic, cache behavior, or achieved hardware utilization. The Roofline notebook therefore treats the values as estimates and labels the M2 Pro's published 200 GB/s unified-memory bandwidth as a theoretical reference ceiling. Benchmark results will vary with the exact M2 Pro configuration and system state.
