# =============================================================================
# uSpmmProject Makefile Helper
# =============================================================================

BUILD_DIR = build
CMAKE_FLAGS =

.PHONY: all help config build clean test run-all-benchmarks run-all-examples \
        run-bench-basic run-bench-mtx run-bench-synth run-bench-shuffled \
        run-ex-mtx run-ex-checkers run-ex-uspmm

# Default target
all: build

# ---------------------------------------------------------
# Build System
# ---------------------------------------------------------
config:
	@echo ">>> Configuring CMake..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

build: config
	@echo ">>> Building all targets..."
	cmake --build $(BUILD_DIR) -j

clean:
	@echo ">>> Cleaning build directory..."
	rm -rf $(BUILD_DIR)

test: build
	@echo ">>> Running Tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure

# ---------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------
run-all-benchmarks: run-bench-basic run-bench-synth run-bench-shuffled run-bench-mtx

run-bench-basic: config
	@echo ">>> Building and running basic benchmark..."
	cmake --build $(BUILD_DIR) --target benchmark_basic -j
	./$(BUILD_DIR)/benchmarks/benchmark_basic

run-bench-mtx: config
	@echo ">>> Building and running MTX file benchmark..."
	cmake --build $(BUILD_DIR) --target benchmark_mtx_files -j
	./$(BUILD_DIR)/benchmarks/benchmark_mtx_files

run-bench-synth: config
	@echo ">>> Building and running synthetic benchmark..."
	cmake --build $(BUILD_DIR) --target benchmark_synthetic_matrices -j
	./$(BUILD_DIR)/benchmarks/benchmark_synthetic_matrices

run-bench-shuffled: config
	@echo ">>> Building and running shuffled benchmark..."
	cmake --build $(BUILD_DIR) --target benchmark_shuffled_synthetic_matrices -j
	./$(BUILD_DIR)/benchmarks/benchmark_shuffled_synthetic_matrices

# ---------------------------------------------------------
# Examples
# ---------------------------------------------------------
run-all-examples: run-ex-mtx run-ex-checkers run-ex-uspmm

run-ex-mtx: config
	@echo ">>> Building and running MTX file reading example..."
	cmake --build $(BUILD_DIR) --target mtx_file_reading -j
	./$(BUILD_DIR)/examples/mtx_file_reading

run-ex-checkers: config
	@echo ">>> Building and running Checkers reordering example..."
	cmake --build $(BUILD_DIR) --target checkers_reordering -j
	./$(BUILD_DIR)/examples/checkers_reordering

run-ex-uspmm: config
	@echo ">>> Building and running uSPMM checkers HIP example..."
	cmake --build $(BUILD_DIR) --target uspmm_checkers -j
	./$(BUILD_DIR)/examples/uspmm_checkers

# ---------------------------------------------------------
# Help Menu
# ---------------------------------------------------------
help:
	@echo "========================================================"
	@echo "                uSpmmProject Makefile                   "
	@echo "========================================================"
	@echo "Build Commands:"
	@echo "  make build               - Configure and build everything"
	@echo "  make clean               - Remove the build directory"
	@echo "  make test                - Run all unit tests via ctest"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make run-bench-basic     - Run benchmark_basic"
	@echo "  make run-bench-mtx       - Run benchmark_mtx_files"
	@echo "  make run-bench-synth     - Run benchmark_synthetic_matrices"
	@echo "  make run-bench-shuffled  - Run benchmark_shuffled_synthetic_matrices"
	@echo "  make run-all-benchmarks  - Run all benchmarks sequentially"
	@echo ""
	@echo "Examples:"
	@echo "  make run-ex-mtx          - Run mtx_file_reading example"
	@echo "  make run-ex-checkers     - Run checkers_reordering example"
	@echo "  make run-ex-uspmm        - Run uspmm_checkers HIP example"
	@echo "  make run-all-examples    - Run all examples sequentially"
	@echo "========================================================"