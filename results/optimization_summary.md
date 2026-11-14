# Kernel benchmark comparison

Measured on an Apple M2 Pro with two warmup iterations and four timed repetitions per case. All measurements use FP32. The runner compares native CPU kernels, Metal GPU kernels, and PyTorch baselines. GPU is labeled `mps` in the CSV because that is the PyTorch device interface used to dispatch work to the Apple GPU.

GEMM sizes: 128, 256, 512, and 1,024. Elementwise and fused multiply-add sizes: 65,536, 262,144, and 1,048,576. Convolution input widths: 16, 32, and 64, with 8 input channels, 16 output channels, and a 3×3 filter. Softmax uses 256 rows and LayerNorm uses 1,024 rows; each is measured at widths 256, 1,024, and 4,096.

| Operation | Device | Size | Naive (ms) | Optimized (ms) | PyTorch (ms) |
|---|---|---:|---:|---:|---:|
| GEMM | CPU | 128 | 0.3423 | 0.2728 | 0.0066 |
| GEMM | CPU | 256 | 2.5718 | 1.5763 | 0.0301 |
| GEMM | CPU | 512 | 23.0980 | 12.4659 | 0.1148 |
| GEMM | CPU | 1,024 | 213.2720 | 120.8760 | 0.9852 |
| GEMM | GPU | 128 | 0.2452 | 0.2205 | 0.2027 |
| GEMM | GPU | 256 | 0.5128 | 0.4889 | 0.2119 |
| GEMM | GPU | 512 | 1.7998 | 1.3587 | 0.3832 |
| GEMM | GPU | 1,024 | 6.5527 | 3.9122 | 2.1470 |
| Add | CPU | 65,536 | 0.1182 | 0.1142 | 0.0347 |
| Add | CPU | 262,144 | 0.1415 | 0.1306 | 0.0744 |
| Add | CPU | 1,048,576 | 0.2337 | 0.2397 | 0.0669 |
| Add | GPU | 65,536 | 0.1860 | 0.3669 | 0.2783 |
| Add | GPU | 262,144 | 0.3004 | 0.4182 | 0.2987 |
| Add | GPU | 1,048,576 | 0.2385 | 0.3795 | 0.2962 |
| Multiply | CPU | 65,536 | 0.1291 | 0.1143 | 0.0335 |
| Multiply | CPU | 262,144 | 0.1438 | 0.1328 | 0.0259 |
| Multiply | CPU | 1,048,576 | 0.1966 | 0.2170 | 0.0592 |
| Multiply | GPU | 65,536 | 0.1483 | 0.1700 | 0.1647 |
| Multiply | GPU | 262,144 | 0.2186 | 0.2212 | 0.2538 |
| Multiply | GPU | 1,048,576 | 0.2516 | 0.4334 | 0.2321 |
| Fused multiply-add | CPU | 65,536 | 0.1234 | 0.1260 | 0.0730 |
| Fused multiply-add | CPU | 262,144 | 0.1303 | 0.1235 | 0.0966 |
| Fused multiply-add | CPU | 1,048,576 | 0.2041 | 0.1602 | 0.1293 |
| Fused multiply-add | GPU | 65,536 | 0.4051 | 0.1628 | 0.2709 |
| Fused multiply-add | GPU | 262,144 | 0.3722 | 0.1668 | 0.2188 |
| Fused multiply-add | GPU | 1,048,576 | 0.4055 | 0.2631 | 0.3136 |
| 2D convolution | CPU | 16 | 0.1343 | 0.1131 | 0.0391 |
| 2D convolution | CPU | 32 | 0.2110 | 0.1429 | 0.0464 |
| 2D convolution | CPU | 64 | 0.6467 | 0.2992 | 0.0832 |
| 2D convolution | GPU | 16 | 0.1838 | 0.2960 | 0.1941 |
| 2D convolution | GPU | 32 | 0.1843 | 0.2564 | 0.2019 |
| 2D convolution | GPU | 64 | 0.4206 | 0.2839 | 0.1929 |
| Softmax | CPU | 256 | 0.1835 | 0.1275 | 0.0599 |
| Softmax | CPU | 1,024 | 0.4444 | 0.2305 | 0.1457 |
| Softmax | CPU | 4,096 | 1.5789 | 0.6476 | 0.2776 |
| Softmax | GPU | 256 | 0.4033 | 0.1486 | 0.2242 |
| Softmax | GPU | 1,024 | 1.1998 | 0.1581 | 0.2615 |
| Softmax | GPU | 4,096 | 4.4322 | 0.1980 | 0.4313 |
| LayerNorm | CPU | 256 | 0.1668 | 0.1434 | 0.0508 |
| LayerNorm | CPU | 1,024 | 0.4468 | 0.3083 | 0.0986 |
| LayerNorm | CPU | 4,096 | 1.5092 | 1.1214 | 0.4403 |
| LayerNorm | GPU | 256 | 0.4138 | 0.1629 | 0.1962 |
| LayerNorm | GPU | 1,024 | 1.2662 | 0.2089 | 0.2939 |
| LayerNorm | GPU | 4,096 | 4.6067 | 0.3499 | 0.7499 |

Convolution sizes refer to the square input width. Softmax and LayerNorm sizes refer to row width or feature count, respectively. Operation counts and traffic assumptions are documented in [`docs/metrics.md`](metrics.md).

The notebook uses `SHOW_NAIVE`, `SHOW_OPTIMIZED`, and `SHOW_PYTORCH` to control which implementations appear, and `SELECTED_KERNELS` to choose operations. Short operations can be dominated by thread creation or GPU dispatch overhead, so compare timings across larger sizes as well.
