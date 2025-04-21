# 18847 Project - Optimizing (DC)? GEMM on single CPU and multiple CPUs

## File Structure

```
18847Project/
├── Makefile                # Main GNU makefile
├── README.md               # Project documentation
├── include/                # Header files
│   ├── gemm.h              # GEMM interface definition
│   ├── types.h             # Matrix class definition
│   └── benchmark.h         # Benchmarking utilities
│   └── common.h            # Common utilities
├── src/                    # Source files
│   ├── main.cpp            # Main application entry
│   ├── benchmark.cpp       # Benchmarking implementation
│   └── implementations/    # GEMM implementations
│       ├── gemm_naive.cpp  # Naive implementation
│       └── TODO 
```

## Requirements

For macOS, 
```shell
brew install clang-format

brew install libomp

brew install simde

brew install openblas
```

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

## Test
```
make
./gemm
```