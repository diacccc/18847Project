#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <cassert>

namespace gemm {

// Simple matrix class template
template<typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    size_t ld_;     // Leading dimension (for padded matrices)
    std::vector<T> data_;

public:
    // Constructor for a rows x cols matrix
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), ld_(rows), data_(rows * cols) {}
    
    // Constructor with leading dimension (useful for aligned memory)
    Matrix(size_t rows, size_t cols, size_t ld) 
        : rows_(rows), cols_(cols), ld_(ld), data_(rows * ld) {
        assert(ld >= cols && "Leading dimension must be >= cols");
    }
    
    // Destructor
    ~Matrix() {
    }
    
    // Fill with random values in range [min, max]
    void randomize(T min = 0, T max = 1) {
    #pragma omp parallel num_threads(8)
    {
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
        std::uniform_real_distribution<T> dist(min, max);

        // thread initializes memory
        #pragma omp for schedule(static)
        for (size_t j = 0; j < cols_; ++j) {
            for (size_t i = 0; i < rows_; ++i) {
                at(i, j) = dist(gen);
            }
        }
    }
}
    
    // Fill with a specific value
    void fill(T value) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                at(i, j) = value;
            }
        }
    }
    
    // Access element at (row, col)
    T& at(size_t row, size_t col) {
        assert(row < rows_ && col < cols_ && "Matrix indices out of bounds");
        return data_[row + col * ld_ ];
    }
    
    // Access element at (row, col) (const version)
    const T& at(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_ && "Matrix indices out of bounds");
        return data_[row + col * ld_];
    }
    
    // Get raw data pointer
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // Get dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t ld() const { return ld_; }
    size_t size() const { return rows_ * cols_; }
    
    
    // Print matrix elements
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << " = " << std::endl;
        }
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << at(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Compare matrices (useful for validation)
    // Todo: Normalization or double precision
    
    bool isEqual(const Matrix<T>& other, T tolerance = 1e-5) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if (std::abs(at(i, j) - other.at(i, j)) > tolerance) {
                    std::cerr << "(" << i << ", " << j << ")\t" <<  at(i, j) << " " << other.at(i, j) << std::endl;
                    return false;
                }
            }
        }
        
        return true;
    }
};

}