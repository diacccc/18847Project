# GNUmakefile for GEMM Optimization Project

OMP_PATH := /opt/homebrew/opt/libomp
MKL_PATH := /opt/intel/oneapi/mkl/latest
BLAS_PATH := /opt/homebrew/opt/openblas

# Check the system architecture
ARCH := $(shell uname -m)

# Apple Silicon 
ifeq ($(ARCH),arm64)
	CXX := clang++
	CXXFLAGS := -g -std=c++17 -O3 -Wall -Wextra -march=native 
	LDFLAGS := -lm
	
	# OpenMP flags
	CXXFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PATH)/include 
	LDFLAGS += -lomp -L$(OMP_PATH)/lib 

	# OpenBLAS flags
	CXXFLAGS += -I$(BLAS_PATH)/include
	LDFLAGS += -L$(BLAS_PATH)/lib -lopenblas
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
	LDFLAGS += -liomp5
    # MKL LP64, sequential linking (common setup)
	BLASFLAGS = -I$(MKL_PATH)/include
	LDFLAGS += -L$(MKL_PATH)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl


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
.PHONY: all clean run help rust format format-check run-single doxygen-config doxygen docs

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


# Doxygen configuration
DOXYGEN := doxygen
DOXYFILE := Doxyfile
DOXYGEN_OUTPUT_DIR := docs

# Source directories to document
SRC_DIRS := include src/implementations src/rust_metal_gemm/src

# Doxygen targets
doxygen-config:
	@echo "Generating Doxygen configuration file..."
	@$(DOXYGEN) -g $(DOXYFILE)
	@sed -i.bak 's|OUTPUT_DIRECTORY *=.*|OUTPUT_DIRECTORY = $(DOXYGEN_OUTPUT_DIR)|' $(DOXYFILE)
	@sed -i.bak 's|INPUT *=.*|INPUT = $(SRC_DIRS)|' $(DOXYFILE)
	@sed -i.bak 's|RECURSIVE *=.*|RECURSIVE = YES|' $(DOXYFILE)
	@sed -i.bak 's|EXTRACT_ALL *=.*|EXTRACT_ALL = YES|' $(DOXYFILE)
	@sed -i.bak 's|PROJECT_NAME *=.*|PROJECT_NAME = "GEMM Optimization Project"|' $(DOXYFILE)
	@echo "Created and configured $(DOXYFILE)."

doxygen: 
	@echo "Checking for Doxyfile..."
	@if [ ! -f $(DOXYFILE) ]; then \
		echo "Doxyfile not found. Creating default configuration..."; \
		$(MAKE) doxygen-config; \
	fi
	@echo "Generating documentation with Doxygen..."
	@mkdir -p $(DOXYGEN_OUTPUT_DIR)
	@$(DOXYGEN) $(DOXYFILE)
	@echo "Documentation generated in $(DOXYGEN_OUTPUT_DIR) directory."

docs: doxygen-config doxygen

# Run the benchmark
run-single: $(TARGET)
	LD_LIBRARY_PATH=$(MKL_PATH)/lib/intel64:$$LD_LIBRARY_PATH OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 ./$(TARGET) --output

# Run the benchmark
run: $(TARGET)
	LD_LIBRARY_PATH=$(MKL_PATH)/lib/intel64:$$LD_LIBRARY_PATH MKL_DYNAMIC=FALSE OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 MKL_NUM_THREADS=8 $(OMP_ENV) ./$(TARGET) --output


# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/*.csv
ifeq ($(ARCH),arm64)
	cd $(RUST_DIR) && cargo clean
endif