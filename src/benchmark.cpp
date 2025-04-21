#include "benchmark.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gemm.h"
#include "gemm_blas.h"
#include "gemm_naive.h"
#include "gemm_omp.h"
#include "gemm_simd.h"
#include "gemm_metal.h"  // Added Metal implementation header

namespace gemm
{

GemmBenchmark::GemmBenchmark()
{
    // Set default matrix sizes if none are specified
    if (matrix_sizes_.empty())
    {
        matrix_sizes_ = {
            {1024, 1024, 1024} // Default: 1Kx1K matrices
        };
    }
}

void GemmBenchmark::addImplementation(std::unique_ptr<GemmImplementation> impl)
{
    if (impl)
    {
        std::cout << "Adding implementation: " << impl->getName() << std::endl;
        implementations_.push_back(std::move(impl));
    }
}

void GemmBenchmark::addAllImplementations()
{
    // Get all registered implementations
    // We'll create each one and add it
    std::vector<std::string> impl_names = {// "cpu_naive",
                                           "cpu_simd", "BLAS",
#ifdef __APPLE__
                                           "Metal",  // Add Metal on Apple platforms
#endif
//                                           "cpu_omp"
                                           };

    for (const auto &name : impl_names)
    {
        auto impl = createImplementation(name);
        if (impl)
        {
            addImplementation(std::unique_ptr<GemmImplementation>(impl));
        }
    }
}

void GemmBenchmark::setMatrixSizes(const std::vector<std::tuple<size_t, size_t, size_t>> &sizes)
{
    matrix_sizes_ = sizes;
}

std::vector<BenchmarkResult> GemmBenchmark::runAll()
{
    std::vector<BenchmarkResult> results;

    // Find reference implementation if validation is enabled
    GemmImplementation *ref_impl = nullptr;
    if (validate_results_)
    {
        for (const auto &impl : implementations_)
        {
            if (impl->getName() == reference_impl_)
            {
                ref_impl = impl.get();
                break;
            }
        }

        if (!ref_impl)
        {
            std::cout << "Warning: Reference implementation '" << reference_impl_
                      << "' not found. Result validation will be skipped." << std::endl;
            validate_results_ = false;
        }
    }

    // Run benchmarks for each implementation and matrix size
    for (const auto &size : matrix_sizes_)
    {
        size_t M, N, K;
        std::tie(M, N, K) = size;

        std::cout << "Benchmarking matrix size: " << M << "x" << N << "x" << K << std::endl;
        const size_t LD = 3000;
        // Create matrices for this benchmark
        Matrix<float> A(M, K, LD);
        Matrix<float> B(K, N, LD);
        Matrix<float> C(M, N, LD);

        // Initialize matrices with random values
        A.randomize(-1.0f, 1.0f);
        B.randomize(-1.0f, 1.0f);
        C.fill(0.0f);
        // Reference result for validation
        std::unique_ptr<Matrix<float>> ref_result;
        if (validate_results_ && ref_impl)
        {
            ref_result = std::make_unique<Matrix<float>>(M, N);
            ref_result->fill(0.0f);

            // Compute reference result
            std::cout << "Computing reference result using " << ref_impl->getName() << "..." << std::endl;
            ref_impl->execute(1.0f, A, B, 0.0f, *ref_result);
        }

        // Benchmark each implementation
        for (const auto &impl : implementations_)
        {
            std::cout << "  Running " << impl->getName() << "... ";
            std::cout.flush();

            // Initialize the implementation
            if (!impl->initialize())
            {
                std::cout << "Initialization failed, skipping." << std::endl;
                continue;
            }

            // Run the benchmark
            BenchmarkResult result = benchmarkImplementation(impl.get(), A, B, C, ref_result.get());

            results.push_back(result);

            // Clean up
            impl->cleanup();

            std::cout << "Done! " << result.flops / 1e9 << " GFLOPS, " << result.time_ms << " ms" << std::endl;
        }

        std::cout << std::endl;
    }

    return results;
}

BenchmarkResult GemmBenchmark::benchmarkImplementation(GemmImplementation *impl, Matrix<float> A, Matrix<float> B,
                                                       Matrix<float> C, Matrix<float> *reference_result)
{
    // Prepare result struct
    BenchmarkResult result;
    result.implementation_name = impl->getName();
    size_t M = C.rows(), N = C.cols(), K = A.cols();
    result.m = M;
    result.n = N;
    result.k = K;
    result.validated = false;

    // Warmup runs
    for (size_t i = 0; i < warmup_runs_; ++i)
    {
        C.fill(0.0f);
        impl->execute(1.0f, A, B, 0.0f, C);
    }

    // Benchmark runs
    Timer timer;
    std::vector<double> run_times;

    for (size_t i = 0; i < benchmark_runs_; ++i)
    {
        C.fill(0.0f);

        timer.start();
        impl->execute(1.0f, A, B, 0.0f, C);
        timer.stop();

        run_times.push_back(timer.elapsedMilliseconds());
    }

    // Calculate average execution time (excluding outliers)
    std::sort(run_times.begin(), run_times.end());
    // Exclude potential outliers (10% from each end)
    size_t exclude_count = std::max(size_t(1), run_times.size() / 10);
    double avg_time = 0.0;

    for (size_t i = exclude_count; i < run_times.size() - exclude_count; ++i)
    {
        avg_time += run_times[i];
    }
    avg_time /= (run_times.size() - 2 * exclude_count);

    // Calculate FLOPS
    double flops = calculateGemmFlops(M, N, K) / (avg_time / 1000.0); // Convert ms to seconds

    // Save results
    result.time_ms = avg_time;
    result.flops = flops;

    // Validate result if a reference is provided
    if (reference_result)
    {
        result.validated = C.isEqual(*reference_result, 1e-4f);

        if (!result.validated)
        {
            std::cout << "Warning: Result validation failed for " << impl->getName() << "!" << std::endl;
        }
    }

    return result;
}

bool GemmBenchmark::saveResults(const std::string &filename, const std::vector<BenchmarkResult> &results)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        return false;
    }

    // Write header
    file << BenchmarkResult::getCsvHeader() << std::endl;

    // Write results
    for (const auto &result : results)
    {
        file << result.toCsvRow() << std::endl;
    }

    return true;
}

void registerImplementation(const std::string &name, std::function<GemmImplementation *()> factory)
{
    implementation_factories[name] = factory;
}

// Factory function to create a specific GEMM implementation
GemmImplementation *createImplementation(const std::string &name)
{
    auto it = implementation_factories.find(name);
    if (it != implementation_factories.end())
    {
        return it->second();
    }

    // Implementation not found
    std::cerr << "Warning: Implementation '" << name << "' not found" << std::endl;
    return nullptr;
}

// Register all available implementations
void registerImplementations()
{
    // Register implementations
    registerImplementation("BLAS", []() -> GemmImplementation * { return new GemmBLAS(); });
    registerImplementation("cpu_naive", []() -> GemmImplementation * { return new GemmNaive(); });
    registerImplementation("cpu_simd", []() -> GemmImplementation * { return new GemmSIMD(); });
    registerImplementation("cpu_omp", []() -> GemmImplementation * { return new GemmOMP(); });
    
#ifdef __APPLE__
    // Register Metal implementation on Apple platforms
    registerImplementation("Metal", []() -> GemmImplementation * { return new GemmMetal(); });
#endif

    std::cout << "Registered " << implementation_factories.size() << " GEMM implementations" << std::endl;
}

} // namespace gemm