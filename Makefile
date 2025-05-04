# GNUmakefile for CPU GEMM Optimization Project

# Compiler settings

# Check the system architecture
ARCH := $(shell uname -m)

# Apple Silicon 
ifeq ($(ARCH),arm64)
	CXX := clang++
	CXXFLAGS := -g -std=c++17 -O3 -Wall -Wextra -march=native 
	LDFLAGS := -lm
	
	# OpenMP flags
	CXXFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include 
	LDFLAGS += -lomp -L/opt/homebrew/opt/libomp/lib 

	# OpenBLAS flags
	CXXFLAGS += -I/opt/homebrew/opt/openblas/include
	LDFLAGS += -L/opt/homebrew/opt/openblas/lib -lopenblas
	# LDFLAGS += -framework Accelerate -DACCELERATE_NEW_LAPACK 

	# Rust Metal implementation flags
	RUST_DIR := src/rust_metal_gemm
	RUST_TARGET := target/release
	RUST_LIB := $(RUST_DIR)/$(RUST_TARGET)/librust_metal_gemm.a
	METAL_FLAGS := -framework Metal -framework Foundation -framework CoreGraphics -framework QuartzCore
	RUST_INCLUDE := -I$(RUST_DIR)/include
	
	# Additional flags for Metal support
	CXXFLAGS += $(RUST_INCLUDE)
	LDFLAGS += $(METAL_FLAGS) -L$(RUST_DIR)/$(RUST_TARGET) -lrust_metal_gemm
else
    # Intel x86_64
    CXX := g++
	CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native -fopenmp
	LDFLAGS := -lm 
    # MKL LP64, sequential linking (common setup)
	MKLROOT ?= /opt/intel/oneapi/mkl/latest
	BLASFLAGS = -I$(MKLROOT)/include
	LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl


    # Add NUMA support for x86_64
    # CXXFLAGS += -D_NUMA
    # LDFLAGS += -lnuma
endif

# Directories
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src
IMPL_DIR := $(SRC_DIR)/implementations
RESULTS_DIR := results

# Create build directories
$(shell mkdir -p $(BUILD_DIR) $(BUILD_DIR)/implementations $(RESULTS_DIR))

# Source files
MAIN_SRC := $(SRC_DIR)/main.cpp
BENCHMARK_SRC := $(SRC_DIR)/benchmark.cpp
IMPL_SRCS := $(wildcard $(IMPL_DIR)/*.cpp)

FORMAT_FILES := $(MAIN_SRC) $(BENCHMARK_SRC) $(IMPL_SRCS)
# Object files
MAIN_OBJ := $(BUILD_DIR)/main.o
BENCHMARK_OBJ := $(BUILD_DIR)/benchmark.o
IMPL_OBJS := $(patsubst $(IMPL_DIR)/%.cpp,$(BUILD_DIR)/implementations/%.o,$(IMPL_SRCS))

# All objects
OBJS := $(MAIN_OBJ) $(BENCHMARK_OBJ) $(IMPL_OBJS)

# Main executable
TARGET := gemm

# OpenMP environment variables for specific core binding
OMP_ENV := OMP_PLACES="{0,4},{1,5},{2,6},{3,7}" OMP_PROC_BIND="spread"

# Phony targets
.PHONY: all clean run help rust format format-check run-single run-numa

# Default target
all: rust $(TARGET)

# Help message
help:
	@echo "GEMM Optimization Project"
	@echo "Usage:"
	@echo "  make        - Build the project"
	@echo "  make run    - Run the benchmark"
	@echo "  make clean  - Remove build files"
	@echo "  make help   - Show this help message"

# Rust Metal implementation
rust:
ifeq ($(ARCH),arm64)
	@echo "Building Rust Metal implementation..."
	cd $(RUST_DIR) && cargo build --release
endif

# Build main executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(BLASFLAGS)

# Compile main source
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@
# Compile benchmark source
$(BENCHMARK_OBJ): $(BENCHMARK_SRC)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile implementation sources
$(BUILD_DIR)/implementations/%.o: $(IMPL_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/implementations/gemm_blas.o: $(IMPL_DIR)/gemm_blas.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@ $(BLASFLAGS)

format:
	clang-format -i -style=Microsoft $(FORMAT_FILES)

format-check:
	@echo "Checking code formatting..."
	@clang-format -style=Microsoft --dry-run --Werror $(FORMAT_FILES) \
		|| (echo "Code formatting check failed. Run 'make format' to fix." && exit 1)

# Run the benchmark
run-single: $(TARGET)
	OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 ./$(TARGET) --output results_single.csv

# Run the benchmark
run: $(TARGET)
	OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 MKL_NUM_THREADS=8 $(OMP_ENV) ./$(TARGET) --output results.csv


# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/*.csv
ifeq ($(ARCH),arm64)
	cd $(RUST_DIR) && cargo clean
endif