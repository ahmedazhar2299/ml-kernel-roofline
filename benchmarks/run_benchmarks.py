#!/usr/bin/env python3
"""Run reproducible CPU and Apple MPS microbenchmarks and write CSV results."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch


ELEMENTWISE_DEFAULTS = (1 << 16, 1 << 18, 1 << 20, 1 << 22)
MATMUL_DEFAULTS = (128, 256, 512, 1024)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
FIELDS = [
    "operation", "size", "device", "precision", "median_ms", "operations",
    "gflops", "logical_bytes", "estimated_gb_per_s", "arithmetic_intensity",
    "max_abs_error",
]


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def make_case(operation: str, size: int, dtype: torch.dtype, device: torch.device):
    # Generate the same inputs on CPU first, so CPU and MPS runs are comparable.
    generator = torch.Generator(device="cpu").manual_seed(2026 + size)
    if operation == "matmul":
        a32 = torch.randn((size, size), generator=generator, dtype=torch.float32)
        b32 = torch.randn((size, size), generator=generator, dtype=torch.float32)
        reference = a32 @ b32
        a, b = a32.to(device=device, dtype=dtype), b32.to(device=device, dtype=dtype)
        fn = lambda: a @ b
        operation_count = 2 * size**3
        logical_bytes = (3 * size**2) * torch.empty((), dtype=dtype).element_size()
    else:
        a32 = torch.randn(size, generator=generator, dtype=torch.float32)
        b32 = torch.randn(size, generator=generator, dtype=torch.float32)
        reference = a32 + b32 if operation == "add" else a32 * b32
        a, b = a32.to(device=device, dtype=dtype), b32.to(device=device, dtype=dtype)
        fn = (lambda: a + b) if operation == "add" else (lambda: a * b)
        operation_count = size
        logical_bytes = (3 * size) * torch.empty((), dtype=dtype).element_size()
    return fn, reference, operation_count, logical_bytes


def measure(operation: str, size: int, label: str, device: torch.device,
            dtype: torch.dtype, warmup: int, repeats: int) -> dict:
    fn, reference, operation_count, logical_bytes = make_case(operation, size, dtype, device)
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
        "operation": operation,
        "size": size,
        "device": label,
        "precision": "fp32" if dtype == torch.float32 else "fp16",
        "median_ms": elapsed_ms,
        "operations": operation_count,
        "gflops": operation_count / seconds / 1e9,
        "logical_bytes": logical_bytes,
        "estimated_gb_per_s": logical_bytes / seconds / 1e9,
        "arithmetic_intensity": operation_count / logical_bytes,
        "max_abs_error": max_error,
    }


def parse_csv(value: str, valid: set[str], name: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    unknown = set(items) - valid
    if not items or unknown:
        raise argparse.ArgumentTypeError(f"invalid {name}: {', '.join(sorted(unknown))}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", default="matmul,add,multiply",
                        help="comma-separated subset of matmul,add,multiply")
    parser.add_argument("--sizes", default=None,
                        help="comma-separated sizes; defaults vary by operation")
    parser.add_argument("--precisions", default="fp32,fp16",
                        help="comma-separated subset of fp32,fp16")
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/measurements.csv"))
    args = parser.parse_args()

    operations = parse_csv(args.operations, {"matmul", "add", "multiply"}, "operations")
    precisions = parse_csv(args.precisions, set(DTYPES), "precisions")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")

    devices = []
    if args.device in ("cpu", "both"):
        devices.append(("cpu", torch.device("cpu")))
    if args.device in ("mps", "both"):
        if not torch.backends.mps.is_available():
            if args.device == "mps":
                parser.error("MPS is unavailable. Use an Apple silicon Mac with a supported PyTorch build.")
            print("MPS unavailable; running CPU benchmarks only.")
        else:
            devices.append(("mps", torch.device("mps")))

    rows = []
    for operation in operations:
        default_sizes = MATMUL_DEFAULTS if operation == "matmul" else ELEMENTWISE_DEFAULTS
        sizes = [int(v) for v in args.sizes.split(",")] if args.sizes else default_sizes
        if any(size <= 0 for size in sizes):
            parser.error("all sizes must be positive integers")
        for size in sizes:
            for precision in precisions:
                for device_label, device in devices:
                    try:
                        row = measure(operation, size, device_label, device, DTYPES[precision],
                                      args.warmup, args.repeats)
                    except RuntimeError as error:
                        print(f"Skipping {operation} size={size} {device_label} {precision}: {error}")
                        continue
                    rows.append(row)
                    print(f"{operation:8s} size={size:>8} device={device_label:3s} "
                          f"{precision} {row['median_ms']:.4f} ms, {row['gflops']:.3f} GFLOP/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measurements to {args.output}")


if __name__ == "__main__":
    main()
