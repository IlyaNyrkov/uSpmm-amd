Here is the `README.md` file formatted to clearly introduce your project, explain the core concepts, and guide users through the repository structure and commands.

---

# Unstructured Sparse Matrix Multiplication on AMD (uSpmm)

### Overview

This project implements optimized pipeline for performing Unstructured Sparse Matrix-Dense Matrix Multiplication (SpMM) on modern AMD Datacenter GPUs using matrix cores.

Standard sparse matrix formats (like CSR) typically suffer from poor memory locality and cannot natively utilize hardware Matrix Cores, which expect dense blocks of data. This project solves that problem by:

1. **Clustering:** Utilizing a 1D Jaccard-similarity clustering algorithm on the CPU to permute rows of an unstructured sparse matrix, forcing scattered non-zero elements into dense macro-blocks.
2. **Format Conversion:** Converting the optimized CSR matrix into a Block-CSR (BCSR) format.
3. **Hardware Acceleration:** Executing a custom, fully generic `rocWMMA` kernel that feeds these dense blocks directly into the GPU's Matrix Cores, achieving massive speedups over standard sparse math libraries (like `hipSPARSE`).

### Requirements

* **ROCm Environment:** Requires ROCm version **6.3.3** or higher (there are problems for version starting from 7).
* *Note for HPC clusters:* If you are using the Spack package manager, ensure you load the environment via: `module load rocm/6.3.3`

* **Hardware Target:** This code is heavily optimized and compiled explicitly for the **AMD MI210** accelerator (`gfx90a` architecture).

---

### Getting Started

This repository includes a master `makefile` that handles all CMake configuration, linking, and execution commands.

**1. Build the Project:**
From the root directory, simply run:

```bash
make build
```

*(This will automatically create a `build/` directory, configure CMake with strict `-O3` and `-ffast-math` optimization flags, and compile all libraries and executables).*

**2. Run your first benchmark:**
To see the hardware acceleration in action, run the synthetic matrix suite:

```bash
make run-bench-synth
```

Expected:
```shell
================================================================================
                              SYNTHETIC SUITE SUMMARY                           
================================================================================
Matrix Type                               rocSPARSE (GF)  rocWMMA (GF)   Speedup
--------------------------------------------------------------------------------
Banded (BW=16)                                   1612.05       5992.56     3.72x
Banded (BW=64)                                   2209.20      12030.30     5.45x
Upper Triangular (Density=0.05)                   982.80        359.78     0.37x
Lower Triangular (Density=0.20)                   891.36       1410.33     1.58x
Upper Triangular (Density=0.50)                  1044.57       3596.45     3.44x
Hessenberg Upper (Density=0.05)                   982.98        362.03     0.37x
Hessenberg Lower (Density=0.20)                   892.18       1399.81     1.57x
Arrowhead (Width=128)                            1308.40      12556.98     9.60x
Block Diagonal (Block=64, Dens=0.8)              2044.06       7965.33     3.90x
Block Diagonal (Block=256, Dens=0.9)             2194.49      12812.87     5.84x
Near Symmetric (Dens=0.05, DropRate=0.1)          840.23        271.04     0.32x
Power Law / Scale-Free (Alpha=2.5)                118.59         32.98     0.28x
================================================================================
```


**3. Explore Commands:**
To see a full list of available build, benchmark, and testing commands, run:

```bash
make help
```

---

### Project Structure

The project is organized into modular directories separating the mathematical utilities from the hardware kernels.

```text
uSpmm-amd/
├── benchmarks/      # Performance measurement suites
├── examples/        # Educational usage and correctness tests
├── include/         # Header definitions
│   ├── matrix_utils/    # Data structures, generation, and formatting
│   └── uspmm/           # Clustering logic and GPU kernel signatures
├── src/             # Core implementation files
│   └── uspmm/           # Device code (.hip)
├── tests/           # Gtest files
├── test_data_mtx/   # .mtx files for benchmarking
└── Makefile         # Master command hub
```

#### Key Components & Methods

* **`include/matrix_utils/`**: Contains utility headers for defining types (`CSRMatrix`, `BCSRMatrix`), generating synthetic test matrices (banded, block-diagonal, power-law, etc.), converting between formats, and reading `.mtx` (Matrix Market) files.
* **`include/uspmm/clustering.hpp`**: Contains the row-reordering algorithms to optimize the matrix structure. Key methods include:
```cpp
// Computes the optimal row permutation to group non-zero elements into blocks
std::vector<int> computeIterativeClustering(const matrix_utils::CSRMatrix<T>& matrix, float dist_thresh, int block_width);

// Applies the computed permutation to physically reorder the matrix rows
matrix_utils::CSRMatrix<T> apply_permutation(const matrix_utils::CSRMatrix<T>& mat, const std::vector<int>& perm);
```


* **`src/uspmm/spmm_rocwmma.hip`**: The generic, heavily optimized device kernel utilizing AMD Local Data Share (LDS) and `rocWMMA` cooperative matrix core math. The wrapper signatures adapt to various hardware precisions:
```cpp
// Example wrapper utilizing FP16 Matrix Cores accumulating into FP32
void run_spmm_rocwmma_f16(const int* d_bcsr_row_ptr, const int* d_bcsr_col_ind, const __half* d_bcsr_values,
                          int num_block_rows_a, const __half* d_B, float* d_C,
                          int M, int N, int K);

```



---

### Benchmarks

The `benchmarks/` directory contains suites designed to push the GPU to its limits, comparing the custom Matrix Core implementation directly against the native `roc::hipSPARSE` baseline.

* `benchmark_basic`: A quick, single-run diagnostic test.
* `benchmark_synthetic_matrices`: Tests performance across various optimal topological shapes (Triangular, Banded, Hessenberg, Arrowhead).
* `benchmark_shuffled_synthetic_matrices`: Empirically proves the value of the clustering algorithm by scrambling optimal matrices, running them unoptimized, clustering them, and comparing the regained performance.
* `benchmark_mtx_files`: Automatically parses and benchmarks real-world sparse matrices in .mtx from the `test_data_mtx/` folder.

---

### Examples

The `examples/` directory demonstrates how to safely interact with the API and validate data integrity.

* `mtx_file_reading.cpp`: Shows how to ingest Matrix Market files and calculate sparsity metrics.
* `checkers_reordering.cpp`: Demonstrates the CPU-side clustering pipeline and visualizes block reduction.
* **`uspmm_checkers.hip` (Numerical Correctness):** This is the core validation file. It runs the custom `rocWMMA` SpMM kernel, calculates the exact same result using a standard FP32 math library, and compares every single element. It guarantees that the mixed-precision hardware accumulation is numerically correct and mathematically sound before running benchmarks.