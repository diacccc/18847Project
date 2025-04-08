# GNUmakefile for CPU GEMM Optimization Project

# Compiler settings
CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/Cellar/simde/0.8.2/include
LDFLAGS := -lm -Xpreprocessor -fopenmp -lomp -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/Cellar/simde/0.8.2/lib



# Check the system architecture
ARCH := $(shell uname -m)

# Apple Silicon 
ifeq ($(ARCH),arm64)
    BLASFLAGS = -framework Accelerate -DACCELERATE_NEW_LAPACK
else
    # Intel x86_64
    # TODO
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
	$(CXX) $(CXXFLAGS) $(BLASFLAGS) $^ -o $@ $(LDFLAGS)

# Compile main source
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(BLASFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile benchmark source
$(BENCHMARK_OBJ): $(BENCHMARK_SRC)
	$(CXX) $(CXXFLAGS) $(BLASFLAGS)  -I$(INCLUDE_DIR) -c $< -o $@

# Compile implementation sources
$(BUILD_DIR)/implementations/%.o: $(IMPL_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(BLASFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Run the benchmark
run: $(TARGET)
	VECLIB_MAXIMUM_THREADS=1 ./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/*.csv