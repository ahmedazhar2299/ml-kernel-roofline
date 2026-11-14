#!/usr/bin/env python3
"""Benchmark PyTorch CPU/MPS baselines and custom M2 Metal kernels."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
METAL_SOURCE = ROOT / "benchmarks" / "MetalBench.swift"
METAL_KERNEL_SOURCE = ROOT / "kernels" / "metal" / "baseline.metal"
METAL_BINARY = ROOT / "build" / "metal_bench"
CPU_SOURCE = ROOT / "benchmarks" / "CPUMatmul.cpp"
CPU_BINARY = ROOT / "build" / "cpu_matmul"
ELEMENTWISE_DEFAULTS = (1 << 16, 1 << 18, 1 << 20, 1 << 22)
MATMUL_DEFAULTS = (128, 256, 512, 1024)
CONV_DEFAULTS = (16, 32, 64)
SOFTMAX_DEFAULTS = (256, 1024, 4096)
LAYERNORM_DEFAULTS = (256, 1024, 4096)
SOFTMAX_ROWS = 256
LAYERNORM_ROWS = 1024
CONV_IN_CHANNELS = 8
CONV_OUT_CHANNELS = 16
CONV_KERNEL = 3
DTYPES = {"fp32": torch.float32}
FIELDS = [
    "operation", "size", "device", "precision", "variant", "median_ms",
    "operations", "gflops", "modeled_bytes", "estimated_gb_per_s",
    "arithmetic_intensity", "max_abs_error", "worker_threads",
]


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def make_case(operation: str, size: int, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(2026 + size)
    if operation == "matmul":
        a32 = torch.randn((size, size), generator=generator, dtype=torch.float32)
        b32 = torch.randn((size, size), generator=generator, dtype=torch.float32)
        reference = a32 @ b32
        a, b = a32.to(device=device, dtype=dtype), b32.to(device=device, dtype=dtype)
        fn = lambda: a @ b
        operation_count = 2 * size**3
        # The library baseline is assigned idealized one-read-per-input traffic.
        modeled_bytes = 3 * size**2 * torch.empty((), dtype=dtype).element_size()
    elif operation == "fused_mul_add":
        a32 = torch.randn(size, generator=generator, dtype=torch.float32)
        b32 = torch.randn(size, generator=generator, dtype=torch.float32)
        c32 = torch.randn(size, generator=generator, dtype=torch.float32)
        reference = a32 * b32 + c32
        a, b, c = (x.to(device=device, dtype=dtype) for x in (a32, b32, c32))
        fn = lambda: a * b + c  # two eager operations, used as the fusion baseline
        operation_count = 2 * size
        # Two eager elementwise operations read/write a temporary between kernels.
        modeled_bytes = 6 * size * torch.empty((), dtype=dtype).element_size()
    elif operation in {"add", "multiply"}:
        a32 = torch.randn(size, generator=generator, dtype=torch.float32)
        b32 = torch.randn(size, generator=generator, dtype=torch.float32)
        reference = a32 + b32 if operation == "add" else a32 * b32
        a, b = a32.to(device=device, dtype=dtype), b32.to(device=device, dtype=dtype)
        fn = (lambda: a + b) if operation == "add" else (lambda: a * b)
        operation_count = size
        modeled_bytes = 3 * size * torch.empty((), dtype=dtype).element_size()
    elif operation == "conv2d":
        height = width = size
        x32 = torch.randn((1, CONV_IN_CHANNELS, height, width),
                          generator=generator, dtype=torch.float32)
        w32 = torch.randn((CONV_OUT_CHANNELS, CONV_IN_CHANNELS,
                           CONV_KERNEL, CONV_KERNEL),
                          generator=generator, dtype=torch.float32)
        reference = torch.nn.functional.conv2d(x32, w32, padding=1)
        x, weight = (v.to(device=device, dtype=dtype) for v in (x32, w32))
        fn = lambda: torch.nn.functional.conv2d(x, weight, padding=1)
        outputs = CONV_OUT_CHANNELS * height * width
        macs = outputs * CONV_IN_CHANNELS * CONV_KERNEL**2
        operation_count = 2 * macs
        elements = x32.numel() + w32.numel() + outputs
        modeled_bytes = elements * torch.empty((), dtype=dtype).element_size()
    elif operation == "softmax":
        x32 = torch.randn((SOFTMAX_ROWS, size), generator=generator, dtype=torch.float32)
        reference = torch.softmax(x32, dim=-1)
        x = x32.to(device=device, dtype=dtype)
        fn = lambda: torch.softmax(x, dim=-1)
        elements = SOFTMAX_ROWS * size
        operation_count = 5 * elements  # Approximate arithmetic; excludes exp cost weighting.
        modeled_bytes = 4 * elements * torch.empty((), dtype=dtype).element_size()
    elif operation == "layernorm":
        x32 = torch.randn((LAYERNORM_ROWS, size), generator=generator, dtype=torch.float32)
        reference = torch.nn.functional.layer_norm(x32, (size,), eps=1e-5)
        x = x32.to(device=device, dtype=dtype)
        fn = lambda: torch.nn.functional.layer_norm(x, (size,), eps=1e-5)
        elements = LAYERNORM_ROWS * size
        operation_count = 5 * elements  # Approximate arithmetic, including variance and scaling.
        modeled_bytes = 3 * elements * torch.empty((), dtype=dtype).element_size()
    return fn, reference, operation_count, modeled_bytes


def measure_torch(operation: str, size: int, device_label: str, device: torch.device,
                  precision: str, variant: str, warmup: int, repeats: int) -> dict:
    dtype = DTYPES[precision]
    fn, reference, operation_count, modeled_bytes = make_case(operation, size, dtype, device)
    for _ in range(warmup):
        result = fn()
    sync(device)

    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1_000.0)

    elapsed_ms = statistics.median(samples)
    result_cpu = result.detach().to(device="cpu", dtype=torch.float32)
    max_error = (result_cpu - reference).abs().max().item()
    seconds = elapsed_ms / 1_000.0
    return {
        "operation": operation, "size": size, "device": device_label,
        "precision": precision, "variant": variant, "median_ms": elapsed_ms,
        "operations": operation_count, "gflops": operation_count / seconds / 1e9,
        "modeled_bytes": modeled_bytes,
        "estimated_gb_per_s": modeled_bytes / seconds / 1e9,
        "arithmetic_intensity": operation_count / modeled_bytes,
        "max_abs_error": max_error,
        "worker_threads": torch.get_num_threads() if device.type == "cpu" else 0,
    }


def build_metal_runner() -> None:
    METAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    sources = (METAL_SOURCE, METAL_KERNEL_SOURCE)
    if METAL_BINARY.exists() and all(
        METAL_BINARY.stat().st_mtime >= source.stat().st_mtime for source in sources
    ):
        return
    subprocess.run([
        "swiftc", "-O", "-framework", "Metal", "-framework", "Foundation",
        str(METAL_SOURCE), "-o", str(METAL_BINARY),
    ], check=True)


def measure_metal(cases: list[dict]) -> list[dict]:
    if not cases:
        return []
    build_metal_runner()
    result = subprocess.run(
        [str(METAL_BINARY)], input=json.dumps({"cases": cases}), text=True,
        capture_output=True, check=True, cwd=ROOT,
    )
    return json.loads(result.stdout)


def build_cpu_runner() -> None:
    CPU_BINARY.parent.mkdir(parents=True, exist_ok=True)
    if CPU_BINARY.exists() and CPU_BINARY.stat().st_mtime >= CPU_SOURCE.stat().st_mtime:
        return
    subprocess.run([
        "clang++", "-O3", "-std=c++17", "-pthread", str(CPU_SOURCE),
        "-o", str(CPU_BINARY),
    ], check=True)


def measure_cpu_kernels(cases: list[dict]) -> list[dict]:
    if not cases:
        return []
    build_cpu_runner()
    manifest = "".join(
        f"{item['operation']},{item['size']},{item['variant']},{item['precision']},"
        f"{item['warmup']},{item['repeats']}\n"
        for item in cases
    )
    result = subprocess.run(
        [str(CPU_BINARY)], input=manifest, text=True, capture_output=True,
        check=True, cwd=ROOT,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations",
                        default="matmul,add,multiply,fused_mul_add,conv2d,softmax,layernorm",
                        help="subset of matmul,add,multiply,fused_mul_add,conv2d,softmax,layernorm")
    parser.add_argument("--sizes", default=None,
                        help="use these comma-separated sizes for every operation")
    parser.add_argument("--matmul-sizes", default=None,
                        help="override matrix dimensions independently")
    parser.add_argument("--elementwise-sizes", default=None,
                        help="override vector lengths for add, multiply, and fused multiply-add")
    parser.add_argument("--conv-sizes", default=None,
                        help="override 2D convolution input heights/widths")
    parser.add_argument("--softmax-sizes", default=None,
                        help="override softmax row widths")
    parser.add_argument("--layernorm-sizes", default=None,
                        help="override LayerNorm feature counts")
    parser.add_argument("--precisions", default="fp32",
                        help="benchmark precision (currently fp32 only)")
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/measurements.csv"))
    args = parser.parse_args()

    valid_ops = {"matmul", "add", "multiply", "fused_mul_add",
                 "conv2d", "softmax", "layernorm"}
    operations = [x.strip() for x in args.operations.split(",") if x.strip()]
    precisions = [x.strip() for x in args.precisions.split(",") if x.strip()]
    if not operations or set(operations) - valid_ops:
        parser.error(f"operations must be selected from {', '.join(sorted(valid_ops))}")
    if not precisions or set(precisions) - set(DTYPES):
        parser.error("only fp32 benchmarks are currently available")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")

    devices = []
    if args.device in ("cpu", "both"):
        devices.append(("cpu", torch.device("cpu")))
    if args.device in ("mps", "both"):
        if torch.backends.mps.is_available():
            devices.append(("mps", torch.device("mps")))
        elif args.device == "mps":
            parser.error("MPS is unavailable. Use an Apple silicon Mac with MPS-enabled PyTorch.")
        else:
            print("MPS unavailable; running CPU benchmarks only.")

    rows = []
    custom_cpu_cases = []
    custom_metal_cases = []
    for operation in operations:
        default_sizes = {
            "matmul": MATMUL_DEFAULTS,
            "conv2d": CONV_DEFAULTS,
            "softmax": SOFTMAX_DEFAULTS,
            "layernorm": LAYERNORM_DEFAULTS,
        }.get(operation, ELEMENTWISE_DEFAULTS)
        selected_sizes = args.sizes
        if selected_sizes is None:
            override = {
                "matmul": args.matmul_sizes,
                "conv2d": args.conv_sizes,
                "softmax": args.softmax_sizes,
                "layernorm": args.layernorm_sizes,
            }.get(operation, args.elementwise_sizes)
            selected_sizes = override
        sizes = [int(x) for x in selected_sizes.split(",")] if selected_sizes else default_sizes
        if any(size <= 0 for size in sizes):
            parser.error("all sizes must be positive integers")
        for size in sizes:
            for precision in precisions:
                for device_label, device in devices:
                    variant = "torch_eager" if operation == "fused_mul_add" else "torch"
                    try:
                        row = measure_torch(operation, size, device_label, device,
                                            precision, variant, args.warmup, args.repeats)
                        rows.append(row)
                    except RuntimeError as error:
                        print(f"Skipping {operation} {size} {device_label} {precision}: {error}")

                if "mps" in [name for name, _ in devices]:
                    custom_metal_cases.extend([
                        {"operation": operation, "size": size, "variant": variant,
                         "precision": precision, "warmup": args.warmup,
                         "repeats": args.repeats}
                        for variant in ("naive", "optimized")
                    ])
                if "cpu" in [name for name, _ in devices]:
                    custom_cpu_cases.extend([
                        {"operation": operation, "size": size, "variant": variant,
                         "precision": precision, "warmup": args.warmup,
                         "repeats": args.repeats}
                        for variant in ("naive", "optimized")
                    ])

    try:
        rows.extend(measure_cpu_kernels(custom_cpu_cases))
        rows.extend(measure_metal(custom_metal_cases))
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        parser.error(f"could not build or run a native benchmark: {error}")

    rows.sort(key=lambda row: (row["operation"], row["size"], row["precision"],
                               row["device"], row["variant"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        thread_info = f", threads={row.get('worker_threads', 0)}" if row.get("worker_threads", 0) else ""
        print(f"{row['operation']:12s} n={row['size']:>8} {row['device']:8s} "
              f"{row['precision']} {row['variant']:12s} {row['median_ms']:.4f} ms, "
              f"{row['gflops']:.3f} GFLOP/s{thread_info}")
    print(f"Wrote {len(rows)} measurements to {args.output}")


if __name__ == "__main__":
    main()
