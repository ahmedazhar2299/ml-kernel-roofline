#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kTile = 32;

struct Case {
    int size;
    std::string variant;
    int warmup;
    int repeats;
};

std::vector<float> make_matrix(int n, int seed) {
    std::vector<float> values(static_cast<std::size_t>(n) * n);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * seed) % 101) - 50) / 50.0f;
    }
    return values;
}

void naive_gemm(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, int n, int first_row, int last_row) {
    for (int i = first_row; i < last_row; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[static_cast<std::size_t>(i) * n + k] *
                       b[static_cast<std::size_t>(k) * n + j];
            }
            c[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }
}

void tiled_gemm(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, int n, int first_tile_row, int last_tile_row) {
    const int first_row = first_tile_row * kTile;
    const int last_row = std::min(n, last_tile_row * kTile);
    for (int i = first_row; i < last_row; ++i) {
        std::fill(c.begin() + static_cast<std::size_t>(i) * n,
                  c.begin() + static_cast<std::size_t>(i + 1) * n, 0.0f);
    }

    // Block rows, columns, and the reduction dimension to reuse cache lines.
    for (int ii = first_row; ii < last_row; ii += kTile) {
        const int i_end = std::min(ii + kTile, last_row);
        for (int kk = 0; kk < n; kk += kTile) {
            const int k_end = std::min(kk + kTile, n);
            for (int jj = 0; jj < n; jj += kTile) {
                const int j_end = std::min(jj + kTile, n);
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const float av = a[static_cast<std::size_t>(i) * n + k];
                        for (int j = jj; j < j_end; ++j) {
                            c[static_cast<std::size_t>(i) * n + j] +=
                                av * b[static_cast<std::size_t>(k) * n + j];
                        }
                    }
                }
            }
        }
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) return (values[middle - 1] + values[middle]) / 2.0;
    return values[middle];
}

double max_error(const std::vector<float>& x, const std::vector<float>& y) {
    double error = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        error = std::max(error, std::abs(static_cast<double>(x[i]) - y[i]));
    }
    return error;
}

Case parse_case(const std::string& line) {
    std::istringstream input(line);
    std::string field;
    Case item{};
    if (!std::getline(input, field, ',')) std::exit(2);
    item.size = std::stoi(field);
    if (!std::getline(input, item.variant, ',')) std::exit(2);
    if (!std::getline(input, field, ',')) std::exit(2);
    item.warmup = std::stoi(field);
    if (!std::getline(input, field, ',')) std::exit(2);
    item.repeats = std::stoi(field);
    if (item.size <= 0 || item.warmup < 0 || item.repeats <= 0 ||
        (item.variant != "naive" && item.variant != "tiled")) {
        std::cerr << "invalid CPU GEMM case: " << line << '\n';
        std::exit(2);
    }
    return item;
}

void run_case(const Case& item, std::vector<float>& reference) {
    const int n = item.size;
    const std::size_t count = static_cast<std::size_t>(n) * n;
    const auto a = make_matrix(n, 17);
    const auto b = make_matrix(n, 29);
    std::vector<float> c(count, 0.0f);

    const unsigned int available = std::max(1u, std::thread::hardware_concurrency());
    const int tile_rows = (n + kTile - 1) / kTile;
    const int workers = std::min<int>(available, tile_rows);
    auto execute = [&]() {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int worker = 0; worker < workers; ++worker) {
            if (item.variant == "naive") {
                const int first = n * worker / workers;
                const int last = n * (worker + 1) / workers;
                threads.emplace_back(naive_gemm, std::cref(a), std::cref(b),
                                     std::ref(c), n, first, last);
            } else {
                const int first_tile = tile_rows * worker / workers;
                const int last_tile = tile_rows * (worker + 1) / workers;
                threads.emplace_back(tiled_gemm, std::cref(a), std::cref(b),
                                     std::ref(c), n, first_tile, last_tile);
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
    const double elapsed_ms = median(samples);
    double error = 0.0;
    if (item.variant == "naive") {
        reference = c;
    } else if (reference.size() == c.size()) {
        error = max_error(c, reference);
    }

    const std::int64_t nn = n;
    const std::int64_t operations = 2 * nn * nn * nn;
    const std::int64_t tiles = (n + kTile - 1) / kTile;
    const std::int64_t input_elements = item.variant == "naive"
        ? 2 * nn * nn * nn
        : 2 * nn * nn * tiles;
    const std::int64_t bytes = (input_elements + nn * nn) * sizeof(float);
    const double seconds = elapsed_ms / 1000.0;
    std::cout << "{\"operation\":\"matmul\",\"size\":" << n
              << ",\"device\":\"cpu\",\"precision\":\"fp32\",\"variant\":\""
              << item.variant << "\",\"median_ms\":" << elapsed_ms
              << ",\"operations\":" << operations
              << ",\"gflops\":" << operations / seconds / 1e9
              << ",\"modeled_bytes\":" << bytes
              << ",\"estimated_gb_per_s\":" << bytes / seconds / 1e9
              << ",\"arithmetic_intensity\":" << static_cast<double>(operations) / bytes
              << ",\"max_abs_error\":" << error
              << ",\"worker_threads\":" << workers << "}\n";
}

}  // namespace

int main() {
    std::vector<Case> cases;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty()) cases.push_back(parse_case(line));
    }
    std::vector<float> reference;
    for (const auto& item : cases) run_case(item, reference);
    return 0;
}
