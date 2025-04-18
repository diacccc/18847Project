# GNUmakefile for CPU GEMM Optimization Project

# Compiler settings




# Check the system architecture
ARCH := $(shell uname -m)

# Apple Silicon 
ifeq ($(ARCH),arm64)
	CXX := clang++
	CXXFLAGS := -g -std=c++17 -O3 -Wall -Wextra -march=native -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/Cellar/simde/0.8.2/include
	LDFLAGS := -lm -Xpreprocessor -fopenmp -lomp -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/Cellar/simde/0.8.2/lib
	BLASFLAGS = -L/opt/homebrew/opt/openblas/lib -I/opt/homebrew/opt/openblas/include -lopenblas
	# BLASFLAGS = -framework Accelerate -DACCELERATE_NEW_LAPACK 
else
    # Intel x86_64
    CXX := g++
	CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native 
	LDFLAGS := -lm 
    BLASFLAGS = -lopenblas
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

# Phony targets
.PHONY: all clean run help

# Default target
all: $(TARGET)

# Help message
help:
	@echo "CPU GEMM Optimization Project"
	@echo "Usage:"
	@echo "  make        - Build the project"
	@echo "  make run    - Run the benchmark"
	@echo "  make clean  - Remove build files"
	@echo "  make help   - Show this help message"

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

.PHONY: format format-check

# Run the benchmark
run: $(TARGET)
	USE_GPU=0 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/*.csv