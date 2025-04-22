#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "benchmark.h"
#include "gemm.h"

using namespace gemm;

class CommandLineParser
{
  private:
    std::unordered_map<std::string, std::string> args;
    std::vector<std::string> positionalArgs;

  public:
    CommandLineParser(int argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // Check if it's a flag/option (starts with -- or -)
            if (arg.substr(0, 2) == "--" || arg.substr(0, 1) == "-")
            {
                std::string key = arg;
                std::string value = "";

                // Check if the next argument is a value (not another flag)
                if (i + 1 < argc && argv[i + 1][0] != '-')
                {
                    value = argv[i + 1];
                    i++; // Skip the next argument since we've used it as a value
                }

                args[key] = value;
            }
            else
            {
                // It's a positional argument
                positionalArgs.push_back(arg);
            }
        }
    }

    bool hasArg(const std::string &key) const
    {
        return args.find(key) != args.end();
    }

    std::string getArgValue(const std::string &key, const std::string &defaultValue = "") const
    {
        auto it = args.find(key);
        if (it != args.end())
        {
            return it->second;
        }
        return defaultValue;
    }

    int getArgValueInt(const std::string &key, int defaultValue = 0) const
    {
        auto it = args.find(key);
        if (it != args.end() && !it->second.empty())
        {
            try
            {
                return std::stoi(it->second);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Failed to convert " << key << " value to integer. Using default." << std::endl;
            }
        }
        return defaultValue;
    }

    size_t getArgValueSize(const std::string &key, size_t defaultValue = 0) const
    {
        auto it = args.find(key);
        if (it != args.end() && !it->second.empty())
        {
            try
            {
                return std::stoull(it->second);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Failed to convert " << key << " value to size_t. Using default." << std::endl;
            }
        }
        return defaultValue;
    }

    bool getArgValueBool(const std::string &key, bool defaultValue = false) const
    {
        auto it = args.find(key);
        if (it != args.end())
        {
            if (it->second.empty() || it->second == "true" || it->second == "1")
            {
                return true;
            }
            else if (it->second == "false" || it->second == "0")
            {
                return false;
            }
        }
        return defaultValue;
    }

    const std::vector<std::string> &getPositionalArgs() const
    {
        return positionalArgs;
    }

    void printHelp() const
    {
        std::cout << "GEMM Benchmark Usage:" << std::endl;
        std::cout << "  --runs <n>          Number of benchmark runs (default: 10)" << std::endl;
        std::cout << "  --warmup <n>        Number of warmup runs (default: 3)" << std::endl;
        std::cout << "  --validate <bool>   Validate results (default: true)" << std::endl;
        std::cout << "  --impl <name>       Use specific implementation (default: all)" << std::endl;
        std::cout << "  --output <file>     Output file for results (default: none)" << std::endl;
        std::cout << "  --sizes <file>      Input file with matrix sizes (default: use hardcoded sizes)" << std::endl;
        std::cout << "  --help              Display this help message" << std::endl;
    }
};

int main(int argc, char **argv)
{
    CommandLineParser args(argc, argv);

    if (args.hasArg("--help") || args.hasArg("-h"))
    {
        args.printHelp();
        return 0;
    }
    // Initialize available implementations
    registerImplementations();

    // Setup benchmark
    GemmBenchmark benchmark;

    // Set benchmark parameters
    size_t runs = args.getArgValueSize("--runs", 10);
    benchmark.setBenchmarkRuns(runs);

    size_t warmup = args.getArgValueSize("--warmup", 3);
    benchmark.setWarmupRuns(warmup);

    bool validate = args.getArgValueBool("--validate", true);
    benchmark.setValidateResults(validate);
    
    // Check if specific implementation is requested
    if (args.hasArg("--impl"))
    {
        std::string impl_name = args.getArgValue("--impl");
        std::cout << "Using implementation: " << impl_name << std::endl;
        benchmark.addImplementation(std::unique_ptr<GemmImplementation>(createImplementation(impl_name)));

        if (validate)
        {
            benchmark.addImplementation(std::unique_ptr<GemmImplementation>(createImplementation("BLAS")));
        }
    }
    else
    {
        benchmark.addAllImplementations();
    }

    // Set matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes;

    matrix_sizes = {
        // Small matrices
        {128, 128, 128},    {256, 256, 256},    {384, 384, 384},    {512, 512, 512},
        {640, 640, 640},    {768, 768, 768},    {896, 896, 896},    {1024, 1024, 1024},
        {1152, 1152, 1152}, {1280, 1280, 1280}, {1408, 1408, 1408}, {1536, 1536, 1536},
        {1664, 1664, 1664}, {1792, 1792, 1792}, {1920, 1920, 1920}, {2048, 2048, 2048},
        // // Non-square matrices
        // {1024, 2048, 512},
        // {2048, 1024, 3072}
    };

    benchmark.setMatrixSizes(matrix_sizes);

    

    // Run benchmarks
    std::cout << "Running benchmarks with " << runs << " runs and " << warmup << " warmups, validation "
              << (validate ? "enabled" : "disabled") << std::endl;
    auto results = benchmark.runAll();

    // Print results
    std::cout << "\nBenchmark Results:" << std::endl;
    printBenchmarkResults(results);

    // // Save results to file if requested
    // if (args.hasArg("--output")) {
    //     std::string output_file = args.getArgValue("--output",
    //     "results/benchmark_results.csv"); if
    //     (benchmark.saveResults(output_file, results)) {
    //         std::cout << "Results saved to: " << output_file << std::endl;
    //     } else {
    //         std::cerr << "Error saving results to file." << std::endl;
    //         return 1;
    //     }
    // }

    return 0;
}