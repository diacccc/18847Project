#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include "benchmark.h"
#include "gemm.h"

using namespace gemm;

int main(int argc, char** argv) {

    // Initialize available implementations
    registerImplementations();
    
    // Setup benchmark
    GemmBenchmark benchmark;
    //benchmark.addImplementation(std::unique_ptr<GemmImplementation>(createImplementation("cpu_naive")));
    benchmark.addAllImplementations();
    
    // Set matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes;
    
    matrix_sizes = {
        // Small matrices
        {128, 128, 128},
        {256, 256, 256},
        {384, 384, 384},
        {512, 512, 512},
        {640, 640, 640},
        {768, 768, 768},
        {896, 896, 896},
        {1024, 1024, 1024},
        // // Large matrices
        // {2048, 2048, 2048},
        // // Non-square matrices
        // {1024, 2048, 512},
        // {2048, 1024, 3072}
    };
    
    benchmark.setMatrixSizes(matrix_sizes);
    
    size_t runs = 10;
    benchmark.setBenchmarkRuns(runs);
    
    size_t warmup = 3;
    benchmark.setWarmupRuns(warmup);

    benchmark.setValidateResults(true);
    
    // Run benchmarks
    std::cout << "Running benchmarks..." << std::endl;
    auto results = benchmark.runAll();
    
    // Print results
    std::cout << "\nBenchmark Results:" << std::endl;
    printBenchmarkResults(results);
    
    // // Save results to file if requested
    // if (args.hasArg("--output")) {
    //     std::string output_file = args.getArgValue("--output", "results/benchmark_results.csv");
    //     if (benchmark.saveResults(output_file, results)) {
    //         std::cout << "Results saved to: " << output_file << std::endl;
    //     } else {
    //         std::cerr << "Error saving results to file." << std::endl;
    //         return 1;
    //     }
    // }
    
    return 0;
}