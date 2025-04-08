#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <algorithm>
#include <functional>
#include "gemm.h"
#include "common.h"

namespace gemm {

struct BenchmarkResult {
    std::string implementation_name;
    size_t m, n, k;  // Matrix dimensions
    double flops;    // Floating point operations per second (FLOPS)
    double time_ms;  // Execution time in milliseconds
    bool validated;  // Whether the result was validated for correctness
    
    // For CSV output
    static std::string getCsvHeader() {
        return "Implementation,M,N,K,GFLOPS,TimeMS,Validated";
    }
    
    std::string toCsvRow() const {
        return implementation_name + "," + 
               std::to_string(m) + "," +
               std::to_string(n) + "," +
               std::to_string(k) + "," +
               std::to_string(flops / 1e9) + "," +  // Convert to GFLOPS
               std::to_string(time_ms) + "," +
               (validated ? "Yes" : "No");
    }
};

class GemmBenchmark {
public:
    // Constructor
    GemmBenchmark();
    
    // Add a GEMM implementation to benchmark
    void addImplementation(std::unique_ptr<GemmImplementation> impl);
    
    // Add all registered implementations
    void addAllImplementations();
    
    // Set matrix sizes to benchmark
    // Each triplet is (M, N, K) for matrix dimensions
    void setMatrixSizes(const std::vector<std::tuple<size_t, size_t, size_t>>& sizes);
    
    // Set number of warmup runs (default: 3)
    void setWarmupRuns(size_t runs) { warmup_runs_ = runs; }
    
    // Set number of benchmark runs (default: 10)
    void setBenchmarkRuns(size_t runs) { benchmark_runs_ = runs; }
    
    // Set whether to validate results (default: true)
    void setValidateResults(bool validate) { validate_results_ = validate; }
    
    // Set reference implementation for validation (default: naive CPU)
    void setReferenceImplementation(const std::string& name) { reference_impl_ = name; }
    
    // Run all benchmarks
    std::vector<BenchmarkResult> runAll();
    
    // Save results to CSV file
    bool saveResults(const std::string& filename, const std::vector<BenchmarkResult>& results);
    
private:
    std::vector<std::unique_ptr<GemmImplementation>> implementations_;
    std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes_;
    size_t warmup_runs_ = 3;
    size_t benchmark_runs_ = 10;
    bool validate_results_ = true;
    std::string reference_impl_ = "cpu_naive";
    
    // Benchmark a single implementation with given matrix sizes
    BenchmarkResult benchmarkImplementation(
        GemmImplementation* impl,
        Matrix<float> A, Matrix<float> B, Matrix<float> C,
        Matrix<float>* reference_result = nullptr);
};

// Print a nicely formatted benchmark result table
inline void printBenchmarkResults(const std::vector<BenchmarkResult>& results) {
    // Print header
    std::cout << std::setw(20) << "Implementation" << " | "
            //   << std::setw(5) << "Type" << " | "
              << std::setw(6) << "M" << " | "
              << std::setw(6) << "N" << " | "
              << std::setw(6) << "K" << " | "
              << std::setw(10) << "GFLOPS" << " | "
              << std::setw(10) << "Time (ms)" << " | "
              << std::setw(10) << "Valid" << std::endl;
    
    std::cout << std::string(90, '-') << std::endl;
    
    // Print results
    for (const auto& result : results) {
        std::cout << std::setw(20) << result.implementation_name << " | "
                  << std::setw(6) << result.m << " | "
                  << std::setw(6) << result.n << " | "
                  << std::setw(6) << result.k << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << (result.flops / 1e9) << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.time_ms << " | "
                  << std::setw(10) << (result.validated ? "Yes" : "No") << std::endl;
    }
}

// Map of implementation names to factory functions
static std::unordered_map<std::string, std::function<GemmImplementation*()>> implementation_factories;

// Function to register an implementation factory
void registerImplementation(const std::string& name, std::function<GemmImplementation*()> factory);

// Factory function to create a specific GEMM implementation
GemmImplementation* createImplementation(const std::string& name);

// Register all available implementations
void registerImplementations();

} // namespace gemm