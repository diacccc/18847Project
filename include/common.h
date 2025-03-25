#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace gemm {

// Calculate number of floating point operations for a GEMM operation
inline double calculateGemmFlops(size_t m, size_t n, size_t k) {
    // Each (i,j,k) iteration: 2 operations (multiply and add)
    // Total: 2 * M * N * K
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

// Timer class for benchmarking
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_ = false;

public:
    // Start the timer
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }
    
    // Stop the timer
    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
    }
    
    // Get elapsed time in milliseconds
    double elapsedMilliseconds() const {
        if (is_running_) {
            return std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start_time_).count();
        }
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }
    
    // Get elapsed time in seconds
    double elapsedSeconds() const {
        return elapsedMilliseconds() / 1000.0;
    }
};

} // namespace gemm