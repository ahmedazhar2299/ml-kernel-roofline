#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kTile = 32;
constexpr int kConvInChannels = 8;
constexpr int kConvOutChannels = 16;
constexpr int kConvKernel = 3;
constexpr int kSoftmaxRows = 256;
constexpr int kLayerNormRows = 1024;

struct Case {
    std::string operation;
    int size;
    std::string variant;
    std::string precision;
    int warmup;
    int repeats;
};

template <typename T>
T cast_input(float value) { return static_cast<T>(value); }

template <typename T>
float as_float(T value) { return static_cast<float>(value); }

std::vector<float> make_values(int n, int seed) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        values[i] = static_cast<float>((i * seed % 101) - 50) / 50.0f;
    }
    return values;
}

template <typename T>
std::vector<T> convert(const std::vector<float>& values) {
    std::vector<T> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), cast_input<T>);
    return result;
}

template <typename T>
void naive_gemm(const std::vector<T>& a, const std::vector<T>& b,
                std::vector<T>& c, int n, int first_row, int last_row) {
    for (int i = first_row; i < last_row; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += as_float(a[static_cast<std::size_t>(i) * n + k]) *
                       as_float(b[static_cast<std::size_t>(k) * n + j]);
            }
            c[static_cast<std::size_t>(i) * n + j] = cast_input<T>(sum);
        }
    }
}

template <typename T>
void optimized_gemm(const std::vector<T>& a, const std::vector<T>& b,
                    std::vector<T>& c, std::vector<float>& accum,
                    int n, int first_tile_row, int last_tile_row) {
    const int first_row = first_tile_row * kTile;
    const int last_row = std::min(n, last_tile_row * kTile);
    std::fill(accum.begin() + static_cast<std::size_t>(first_row) * n,
              accum.begin() + static_cast<std::size_t>(last_row) * n, 0.0f);

    for (int ii = first_row; ii < last_row; ii += kTile) {
        const int i_end = std::min(ii + kTile, last_row);
        for (int kk = 0; kk < n; kk += kTile) {
            const int k_end = std::min(kk + kTile, n);
            for (int jj = 0; jj < n; jj += kTile) {
                const int j_end = std::min(jj + kTile, n);
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        for (int k = kk; k < k_end; ++k) {
                            accum[static_cast<std::size_t>(i) * n + j] +=
                                as_float(a[static_cast<std::size_t>(i) * n + k]) *
                                as_float(b[static_cast<std::size_t>(k) * n + j]);
                        }
                    }
                }
            }
        }
    }
    for (int i = first_row; i < last_row; ++i) {
        for (int j = 0; j < n; ++j) {
            const auto index = static_cast<std::size_t>(i) * n + j;
            c[index] = cast_input<T>(accum[index]);
        }
    }
}

template <typename T>
void fused_mul_add(const std::vector<T>& a, const std::vector<T>& b,
                   const std::vector<T>& c, std::vector<T>& out,
                   int first, int last) {
    #pragma clang loop vectorize(enable)
    for (int i = first; i < last; ++i) {
        const float value = as_float(a[i]) * as_float(b[i]) + as_float(c[i]);
        out[i] = cast_input<T>(value);
    }
}

template <typename T>
void naive_fused_mul_add(const std::vector<T>& a, const std::vector<T>& b,
                         const std::vector<T>& c, std::vector<T>& temp,
                         std::vector<T>& out, int first, int last) {
    #pragma clang loop vectorize(disable)
    for (int i = first; i < last; ++i) {
        temp[i] = cast_input<T>(as_float(a[i]) * as_float(b[i]));
    }
    #pragma clang loop vectorize(disable)
    for (int i = first; i < last; ++i) {
        out[i] = cast_input<T>(as_float(temp[i]) + as_float(c[i]));
    }
}

template <typename T>
void elementwise(const std::vector<T>& a, const std::vector<T>& b,
                 std::vector<T>& out, const std::string& operation,
                 bool optimized, int first, int last) {
    auto apply = [&](int i) {
        const float x = as_float(a[i]);
        const float y = as_float(b[i]);
        out[i] = cast_input<T>(operation == "add" ? x + y : x * y);
    };
    if (optimized) {
        int i = first;
        for (; i + 3 < last; i += 4) {
            apply(i); apply(i + 1); apply(i + 2); apply(i + 3);
        }
        for (; i < last; ++i) apply(i);
    } else {
        #pragma clang loop vectorize(disable)
        for (int i = first; i < last; ++i) apply(i);
    }
}

template <typename T>
void conv2d_naive(const std::vector<T>& input, const std::vector<T>& weights,
                  std::vector<T>& output, int height, int first, int last) {
    const int width = height;
    const int output_count = kConvOutChannels * height * width;
    for (int index = first; index < last; ++index) {
        const int ox = index % width;
        const int oy = (index / width) % height;
        const int oc = index / (width * height);
        float sum = 0.0f;
        for (int ic = 0; ic < kConvInChannels; ++ic) {
            for (int ky = 0; ky < kConvKernel; ++ky) {
                for (int kx = 0; kx < kConvKernel; ++kx) {
                    const int iy = oy + ky - 1;
                    const int ix = ox + kx - 1;
                    if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                    const auto x_index = static_cast<std::size_t>(ic) * height * width + iy * width + ix;
                    const auto w_index = ((static_cast<std::size_t>(oc) * kConvInChannels + ic) * 3 + ky) * 3 + kx;
                    sum += as_float(input[x_index]) * as_float(weights[w_index]);
                }
            }
        }
        output[index] = cast_input<T>(sum);
    }
    (void)output_count;
}

template <typename T>
void conv2d_optimized(const std::vector<T>& input, const std::vector<T>& weights,
                      std::vector<T>& output, int height, int first_block, int last_block) {
    const int width = height;
    const int blocks_per_row = (width + 3) / 4;
    for (int block = first_block; block < last_block; ++block) {
        const int block_x = block % blocks_per_row;
        const int oy = (block / blocks_per_row) % height;
        const int oc = block / (blocks_per_row * height);
        const int first_x = block_x * 4;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ic = 0; ic < kConvInChannels; ++ic) {
            for (int ky = 0; ky < kConvKernel; ++ky) {
                const int iy = oy + ky - 1;
                if (iy < 0 || iy >= height) continue;
                for (int kx = 0; kx < kConvKernel; ++kx) {
                    const float weight = as_float(weights[
                        ((static_cast<std::size_t>(oc) * kConvInChannels + ic) * 3 + ky) * 3 + kx]);
                    for (int lane = 0; lane < 4; ++lane) {
                        const int ox = first_x + lane;
                        const int ix = ox + kx - 1;
                        if (ox < width && ix >= 0 && ix < width) {
                            const auto x_index = static_cast<std::size_t>(ic) * height * width + iy * width + ix;
                            sums[lane] += as_float(input[x_index]) * weight;
                        }
                    }
                }
            }
        }
        for (int lane = 0; lane < 4 && first_x + lane < width; ++lane) {
            const auto out_index = static_cast<std::size_t>(oc) * height * width +
                                   oy * width + first_x + lane;
            output[out_index] = cast_input<T>(sums[lane]);
        }
    }
}

template <typename T>
void softmax_naive(const std::vector<T>& input, std::vector<T>& output,
                   int columns, int first_row, int last_row) {
    for (int row = first_row; row < last_row; ++row) {
        const auto offset = static_cast<std::size_t>(row) * columns;
        float max_value = -INFINITY;
        for (int col = 0; col < columns; ++col)
            max_value = std::max(max_value, as_float(input[offset + col]));
        float denominator = 0.0f;
        for (int col = 0; col < columns; ++col)
            denominator += std::exp(as_float(input[offset + col]) - max_value);
        for (int col = 0; col < columns; ++col)
            output[offset + col] = cast_input<T>(
                std::exp(as_float(input[offset + col]) - max_value) / denominator);
    }
}

template <typename T>
void softmax_optimized(const std::vector<T>& input, std::vector<T>& output,
                       std::vector<T>& temporary, int columns,
                       int first_row, int last_row) {
    for (int row = first_row; row < last_row; ++row) {
        const auto offset = static_cast<std::size_t>(row) * columns;
        float max_value = -INFINITY;
        for (int col = 0; col < columns; ++col)
            max_value = std::max(max_value, as_float(input[offset + col]));
        float denominator = 0.0f;
        for (int col = 0; col < columns; ++col) {
            const float value = std::exp(as_float(input[offset + col]) - max_value);
            temporary[offset + col] = cast_input<T>(value);
            denominator += value;
        }
        for (int col = 0; col < columns; ++col)
            output[offset + col] = cast_input<T>(as_float(temporary[offset + col]) / denominator);
    }
}

template <typename T>
void layernorm_naive(const std::vector<T>& input, std::vector<T>& output,
                     int columns, int first_row, int last_row) {
    for (int row = first_row; row < last_row; ++row) {
        const auto offset = static_cast<std::size_t>(row) * columns;
        float mean = 0.0f;
        for (int col = 0; col < columns; ++col) mean += as_float(input[offset + col]);
        mean /= columns;
        float variance = 0.0f;
        for (int col = 0; col < columns; ++col) {
            const float delta = as_float(input[offset + col]) - mean;
            variance += delta * delta;
        }
        variance /= columns;
        const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
        for (int col = 0; col < columns; ++col)
            output[offset + col] = cast_input<T>((as_float(input[offset + col]) - mean) * inv_std);
    }
}

template <typename T>
void layernorm_optimized(const std::vector<T>& input, std::vector<T>& output,
                         int columns, int first_row, int last_row) {
    for (int row = first_row; row < last_row; ++row) {
        const auto offset = static_cast<std::size_t>(row) * columns;
        float sum = 0.0f, sum_squares = 0.0f;
        for (int col = 0; col < columns; ++col) {
            const float value = as_float(input[offset + col]);
            sum += value;
            sum_squares += value * value;
        }
        const float mean = sum / columns;
        const float variance = std::max(0.0f, sum_squares / columns - mean * mean);
        const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
        for (int col = 0; col < columns; ++col)
            output[offset + col] = cast_input<T>((as_float(input[offset + col]) - mean) * inv_std);
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) return (values[middle - 1] + values[middle]) / 2.0;
    return values[middle];
}

double max_error(const std::vector<float>& actual, const std::vector<float>& reference) {
    double error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        error = std::max(error, std::abs(static_cast<double>(actual[i]) - reference[i]));
    }
    return error;
}

Case parse_case(const std::string& line) {
    std::istringstream input(line);
    Case item{};
    std::string field;
    if (!std::getline(input, item.operation, ',')) std::exit(2);
    if (!std::getline(input, field, ',')) std::exit(2);
    item.size = std::stoi(field);
    if (!std::getline(input, item.variant, ',')) std::exit(2);
    if (!std::getline(input, item.precision, ',')) std::exit(2);
    if (!std::getline(input, field, ',')) std::exit(2);
    item.warmup = std::stoi(field);
    if (!std::getline(input, field, ',')) std::exit(2);
    item.repeats = std::stoi(field);
    const bool supported_operation = item.operation == "matmul" || item.operation == "add" ||
        item.operation == "multiply" || item.operation == "fused_mul_add" ||
        item.operation == "conv2d" || item.operation == "softmax" ||
        item.operation == "layernorm";
    const bool valid_variant = item.variant == "naive" || item.variant == "optimized";
    if (item.size <= 0 || item.warmup < 0 || item.repeats <= 0 || !valid_variant ||
        !supported_operation || item.precision != "fp32") {
        std::cerr << "invalid CPU benchmark case: " << line << '\n';
        std::exit(2);
    }
    return item;
}

template <typename T>
std::vector<float> to_float(const std::vector<T>& values) {
    std::vector<float> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), as_float<T>);
    return result;
}

template <typename T>
void run_case(const Case& item, std::vector<float>& reference) {
    const int n = item.size;
    const bool conv = item.operation == "conv2d";
    const int rows = item.operation == "softmax" ? kSoftmaxRows : kLayerNormRows;
    const std::size_t count = item.operation == "matmul"
        ? static_cast<std::size_t>(n) * n
        : (conv ? static_cast<std::size_t>(kConvOutChannels) * n * n
                : ((item.operation == "softmax" || item.operation == "layernorm")
                   ? static_cast<std::size_t>(rows) * n : static_cast<std::size_t>(n)));
    const std::size_t input_count = conv ? static_cast<std::size_t>(kConvInChannels) * n * n : count;
    const std::size_t weight_count = conv
        ? static_cast<std::size_t>(kConvOutChannels) * kConvInChannels * kConvKernel * kConvKernel
        : count;
    const auto af = make_values(static_cast<int>(input_count), 17);
    const auto bf = make_values(static_cast<int>(weight_count), 29);
    const auto cf = make_values(static_cast<int>(count), 43);
    const auto a = convert<T>(af);
    const auto b = convert<T>(bf);
    const auto c_input = convert<T>(cf);
    std::vector<T> output(count, cast_input<T>(0.0f));
    std::vector<T> temp(count, cast_input<T>(0.0f));
    std::vector<float> accumulation(count, 0.0f);

    const unsigned int available = std::max(1u, std::thread::hardware_concurrency());
    const int tile_rows = (n + kTile - 1) / kTile;
    const int conv_work = item.variant == "optimized"
        ? kConvOutChannels * n * ((n + 3) / 4)
        : static_cast<int>(count);
    const int work_items = item.operation == "matmul"
        ? (item.variant == "optimized" ? tile_rows : n)
        : (conv ? conv_work
                : ((item.operation == "softmax" || item.operation == "layernorm") ? rows : n));
    const int workers = std::max(1, std::min<int>(available, work_items));
    auto execute = [&]() {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            if (item.operation == "matmul" && item.variant == "naive") {
                const int first = n * worker / workers;
                const int last = n * (worker + 1) / workers;
                threads.emplace_back(naive_gemm<T>, std::cref(a), std::cref(b),
                                     std::ref(output), n, first, last);
            } else if (item.operation == "matmul") {
                const int first = tile_rows * worker / workers;
                const int last = tile_rows * (worker + 1) / workers;
                threads.emplace_back(optimized_gemm<T>, std::cref(a), std::cref(b),
                                     std::ref(output), std::ref(accumulation), n, first, last);
            } else if (item.operation == "fused_mul_add" && item.variant == "naive") {
                const int first = n * worker / workers;
                const int last = n * (worker + 1) / workers;
                threads.emplace_back(naive_fused_mul_add<T>, std::cref(a), std::cref(b),
                                     std::cref(c_input), std::ref(temp), std::ref(output),
                                     first, last);
            } else if (item.operation == "fused_mul_add") {
                const int first = n * worker / workers;
                const int last = n * (worker + 1) / workers;
                threads.emplace_back(fused_mul_add<T>, std::cref(a), std::cref(b),
                                     std::cref(c_input), std::ref(output), first, last);
            } else if (item.operation == "conv2d" && item.variant == "naive") {
                const int first = conv_work * worker / workers;
                const int last = conv_work * (worker + 1) / workers;
                threads.emplace_back(conv2d_naive<T>, std::cref(a), std::cref(b),
                                     std::ref(output), n, first, last);
            } else if (item.operation == "conv2d") {
                const int first = conv_work * worker / workers;
                const int last = conv_work * (worker + 1) / workers;
                threads.emplace_back(conv2d_optimized<T>, std::cref(a), std::cref(b),
                                     std::ref(output), n, first, last);
            } else if (item.operation == "softmax") {
                const int first = rows * worker / workers;
                const int last = rows * (worker + 1) / workers;
                if (item.variant == "naive") {
                    threads.emplace_back(softmax_naive<T>, std::cref(a), std::ref(output),
                                         n, first, last);
                } else {
                    threads.emplace_back(softmax_optimized<T>, std::cref(a), std::ref(output),
                                         std::ref(temp), n, first, last);
                }
            } else if (item.operation == "layernorm") {
                const int first = rows * worker / workers;
                const int last = rows * (worker + 1) / workers;
                if (item.variant == "naive") {
                    threads.emplace_back(layernorm_naive<T>, std::cref(a), std::ref(output),
                                         n, first, last);
                } else {
                    threads.emplace_back(layernorm_optimized<T>, std::cref(a), std::ref(output),
                                         n, first, last);
                }
            } else {
                const int first = n * worker / workers;
                const int last = n * (worker + 1) / workers;
                threads.emplace_back(elementwise<T>, std::cref(a), std::cref(b),
                                     std::ref(output), item.operation,
                                     item.variant == "optimized", first, last);
            }
        }
        for (auto& thread : threads) thread.join();
    };

    for (int i = 0; i < item.warmup; ++i) execute();
    std::vector<double> samples;
    samples.reserve(item.repeats);
    for (int i = 0; i < item.repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        execute();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::vector<float> expected;
    std::int64_t operations;
    std::int64_t bytes;
    if (item.operation == "matmul") {
        expected.assign(count, 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += af[static_cast<std::size_t>(i) * n + k] *
                           bf[static_cast<std::size_t>(k) * n + j];
                }
                expected[static_cast<std::size_t>(i) * n + j] = sum;
            }
        }
        operations = 2LL * n * n * n;
        const std::int64_t tiles = (n + kTile - 1) / kTile;
        const std::int64_t input_elements = item.variant == "naive"
            ? 2LL * n * n * n : 2LL * n * n * tiles;
        bytes = (input_elements + static_cast<std::int64_t>(count)) *
                4;
    } else if (item.operation == "fused_mul_add") {
        expected.resize(count);
        for (std::size_t i = 0; i < count; ++i) expected[i] = af[i] * bf[i] + cf[i];
        operations = 2LL * n;
        bytes = (item.variant == "naive" ? 6LL : 4LL) * n *
                4;
    } else if (item.operation == "conv2d") {
        expected.assign(count, 0.0f);
        for (int oc = 0; oc < kConvOutChannels; ++oc) {
            for (int oy = 0; oy < n; ++oy) {
                for (int ox = 0; ox < n; ++ox) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < kConvInChannels; ++ic) {
                        for (int ky = 0; ky < kConvKernel; ++ky) {
                            for (int kx = 0; kx < kConvKernel; ++kx) {
                                const int iy = oy + ky - 1, ix = ox + kx - 1;
                                if (iy < 0 || iy >= n || ix < 0 || ix >= n) continue;
                                const auto x_index = static_cast<std::size_t>(ic) * n * n + iy * n + ix;
                                const auto w_index = ((static_cast<std::size_t>(oc) * kConvInChannels + ic) * 3 + ky) * 3 + kx;
                                sum += af[x_index] * bf[w_index];
                            }
                        }
                    }
                    expected[static_cast<std::size_t>(oc) * n * n + oy * n + ox] = sum;
                }
            }
        }
        operations = 2LL * count * kConvInChannels * kConvKernel * kConvKernel;
        bytes = 4LL * (input_count + weight_count + count);
    } else if (item.operation == "softmax") {
        expected.resize(count);
        for (int row = 0; row < rows; ++row) {
            const auto offset = static_cast<std::size_t>(row) * n;
            float max_value = -INFINITY;
            for (int col = 0; col < n; ++col) max_value = std::max(max_value, af[offset + col]);
            float denominator = 0.0f;
            for (int col = 0; col < n; ++col) denominator += std::exp(af[offset + col] - max_value);
            for (int col = 0; col < n; ++col)
                expected[offset + col] = std::exp(af[offset + col] - max_value) / denominator;
        }
        operations = 5LL * count;
        bytes = (item.variant == "naive" ? 4LL : 5LL) * count * 4;
    } else if (item.operation == "layernorm") {
        expected.resize(count);
        for (int row = 0; row < rows; ++row) {
            const auto offset = static_cast<std::size_t>(row) * n;
            float mean = 0.0f;
            for (int col = 0; col < n; ++col) mean += af[offset + col];
            mean /= n;
            float variance = 0.0f;
            for (int col = 0; col < n; ++col) {
                const float delta = af[offset + col] - mean;
                variance += delta * delta;
            }
            variance /= n;
            const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
            for (int col = 0; col < n; ++col)
                expected[offset + col] = (af[offset + col] - mean) * inv_std;
        }
        operations = 5LL * count;
        bytes = (item.variant == "naive" ? 4LL : 3LL) * count * 4;
    } else {
        expected.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            expected[i] = item.operation == "add" ? af[i] + bf[i] : af[i] * bf[i];
        }
        operations = n;
        bytes = 3LL * n * 4;
    }
    const auto actual = to_float(output);
    const double error = max_error(actual, expected);
    const double elapsed_ms = median(samples);
    const double seconds = elapsed_ms / 1000.0;
    const double gflops = static_cast<double>(operations) / seconds / 1e9;
    const double bandwidth = static_cast<double>(bytes) / seconds / 1e9;
    std::cout << "{\"operation\":\"" << item.operation << "\",\"size\":" << n
              << ",\"device\":\"cpu\",\"precision\":\"" << item.precision
              << "\",\"variant\":\"" << item.variant << "\",\"median_ms\":" << elapsed_ms
              << ",\"operations\":" << operations << ",\"gflops\":" << gflops
              << ",\"modeled_bytes\":" << bytes << ",\"estimated_gb_per_s\":" << bandwidth
              << ",\"arithmetic_intensity\":" << static_cast<double>(operations) / bytes
              << ",\"max_abs_error\":" << error << ",\"worker_threads\":" << workers << "}\n";
    reference = std::move(expected);
}

}  // namespace

int main() {
    std::vector<Case> cases;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty()) cases.push_back(parse_case(line));
    }
    for (const auto& item : cases) {
        std::vector<float> reference;
        run_case<float>(item, reference);
    }
    return 0;
}
