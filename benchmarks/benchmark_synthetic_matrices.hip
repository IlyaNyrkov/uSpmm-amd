#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipsparse/hipsparse.h>

#include "matrix_utils/generation.hpp"
#include "matrix_utils/conversion.hpp"
#include "uspmm/clustering.hpp"
#include "uspmm/spmm_rocwmma.hpp"

#define HIP_CHECK(expr) { hipError_t err = (expr); if (err != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __LINE__ << "\n"; exit(1); } }
#define HIPSPARSE_CHECK(expr) { hipsparseStatus_t err = (expr); if (err != HIPSPARSE_STATUS_SUCCESS) { std::cerr << "hipSPARSE Error\n"; exit(1); } }

// Struct to store results for the final summary table
struct BenchmarkResult {
    std::string name;
    double hipsparse_gflops;
    double custom_gflops;
    double speedup;
};

// Helper function now returns the BenchmarkResult
BenchmarkResult run_benchmark_case(const std::string& case_name,
                                   matrix_utils::CSRMatrix<__half>& mat_A_csr,
                                   bool use_clustering,
                                   int N)
{
    std::cout << "\n################################################################\n";
    std::cout << ">>> RUNNING CASE: " << case_name << "\n";
    std::cout << "################################################################\n";

    int M = mat_A_csr.num_rows;
    int K = mat_A_csr.num_cols;
    int r_block = 16, c_block = 16;

    // Safety padding for Matrix Cores
    int padded_M = ((M + r_block - 1) / r_block) * r_block;
    int padded_K = ((K + c_block - 1) / c_block) * c_block;

    std::cout << "--- Initialization ---\n";
    auto h_B = matrix_utils::generation::generate_dense<__half>(padded_K, N, 123);

    std::cout << "--- Benchmarking CPU Clustering ---\n";
    double avg_clustering_ms = 0.0;

    auto bcsr_before = matrix_utils::format::csr_to_bcsr(mat_A_csr, r_block, c_block);
    int blocks_before = bcsr_before.bcsr_values.size() / (r_block * c_block);
    std::cout << "Matrix A has " << blocks_before << " blocks before reordering\n";

    matrix_utils::CSRMatrix<__half> mat_A_reordered = mat_A_csr;

    if (use_clustering) {
        int warmup_runs = 1;
        int timed_runs = 3;

        auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 16);

        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed_runs; ++i) {
            perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 16);
        }
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_ms = stop_cpu - start_cpu;
        avg_clustering_ms = cpu_ms.count() / timed_runs;

        mat_A_reordered = uspmm::apply_permutation(mat_A_csr, perm);
    } else {
        std::cout << "(Clustering skipped for structurally optimal matrix)\n";
    }

    auto mat_A_bcsr = matrix_utils::format::csr_to_bcsr(mat_A_reordered, r_block, c_block);
    int blocks_after = mat_A_bcsr.bcsr_values.size() / (r_block * c_block);

    std::cout << "Matrix A has " << blocks_after << " blocks after reordering\n";
    long long total_dense_elements = static_cast<long long>(M) * K;
    long long reduction_ratio = total_dense_elements - mat_A_bcsr.bcsr_values.size();

    std::cout << "Matrix A had " << total_dense_elements << " elements (if fully dense)\n";
    std::cout << "Matrix A val count reduced by " << reduction_ratio << " elements (Sparse footprint)\n";

    // --- GPU Setup & Memory Allocations ---
    int* d_bcsr_row_ptr, *d_bcsr_col_ind, *d_csr_row_ptr, *d_csr_col_ind;
    __half* d_bcsr_values, *d_B;
    float* d_C_custom, *d_C_hipsparse, *d_csr_values_fp32, *d_B_fp32;

    HIP_CHECK(hipMalloc(&d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_values,  mat_A_bcsr.bcsr_values.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B,            h_B.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C_custom,     padded_M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.data(), mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.data(), mat_A_bcsr.bcsr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_values,  mat_A_bcsr.bcsr_values.data(), mat_A_bcsr.bcsr_values.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice));

    std::vector<float> h_csr_values_fp32(mat_A_reordered.values.size());
    std::vector<float> h_B_fp32(h_B.size());
    for(size_t i=0; i<mat_A_reordered.values.size(); ++i) h_csr_values_fp32[i] = (float)mat_A_reordered.values[i];
    for(size_t i=0; i<h_B.size(); ++i) h_B_fp32[i] = (float)h_B[i];

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, mat_A_reordered.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, mat_A_reordered.col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_values_fp32, h_csr_values_fp32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B_fp32, h_B_fp32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C_hipsparse,  padded_M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_csr_row_ptr, mat_A_reordered.row_ptr.data(), mat_A_reordered.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, mat_A_reordered.col_ind.data(), mat_A_reordered.col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_values_fp32, h_csr_values_fp32.data(), h_csr_values_fp32.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_fp32, h_B_fp32.data(), h_B_fp32.size() * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t handle; HIPSPARSE_CHECK(hipsparseCreate(&handle));
    hipsparseSpMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&descrA, M, K, h_csr_values_fp32.size(), d_csr_row_ptr, d_csr_col_ind, d_csr_values_fp32, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    hipsparseDnMatDescr_t descrB, descrC;
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrB, K, N, N, d_B_fp32, HIP_R_32F, HIPSPARSE_ORDER_ROW));
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrC, M, N, N, d_C_hipsparse, HIP_R_32F, HIPSPARSE_ORDER_ROW));

    float alpha = 1.0f, beta = 0.0f;
    size_t buffer_size = 0; void* d_temp_buffer = nullptr;
    HIPSPARSE_CHECK(hipsparseSpMM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

    // --- GPU Timing Utilities ---
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    float total_ms = 0.0f;
    int warmup_gpu = 3;
    int timed_gpu = 10;

    double total_flops = 2.0 * mat_A_csr.values.size() * N;
    double bytes_A_bcsr = (mat_A_bcsr.bcsr_values.size() * 2.0) + (mat_A_bcsr.bcsr_row_ptr.size() * 4.0) + (mat_A_bcsr.bcsr_col_ind.size() * 4.0);
    double bytes_B = padded_K * N * 2.0;
    double bytes_C = padded_M * N * 4.0;
    double total_bytes = bytes_A_bcsr + bytes_B + bytes_C;

    // --- 3. Custom rocWMMA ---
    std::cout << "--- Benchmarking Custom rocWMMA Kernel ---\n";
    for (int i = 0; i < warmup_gpu; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, padded_M, N, padded_K);
    HIP_CHECK(hipDeviceSynchronize());

    hipEventRecord(start);
    for (int i = 0; i < timed_gpu; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, padded_M, N, padded_K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_ms, start, stop);

    double avg_custom_ms = total_ms / timed_gpu;
    double custom_gflops = (total_flops / (avg_custom_ms * 1e6));
    double custom_gbps = (total_bytes / (avg_custom_ms * 1e6));

    // --- 4. hipSPARSE Baseline ---
    std::cout << "--- Benchmarking hipSPARSE Baseline ---\n";
    for (int i = 0; i < warmup_gpu; ++i) HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
    HIP_CHECK(hipDeviceSynchronize());

    hipEventRecord(start);
    for (int i = 0; i < timed_gpu; ++i) HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_ms, start, stop);

    double avg_hipsparse_ms = total_ms / timed_gpu;
    double hipsparse_gflops = (total_flops / (avg_hipsparse_ms * 1e6));
    double hipsparse_gbps = (total_bytes / (avg_hipsparse_ms * 1e6));

    // --- 5. Output ---
    std::cout << "\n========================================================\n";
    std::cout << "                   BENCHMARK RESULTS                    \n";
    std::cout << "   Matrix Dimensions : " << M << " x " << K << " x " << N << "\n";
    std::cout << "   Runs per Kernel   : " << timed_gpu << "\n";
    std::cout << "========================================================\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. CPU Iterative Clustering Time : " << avg_clustering_ms << " ms\n\n";

    std::cout << "2. hipSPARSE (FP32 Compute Units)\n";
    std::cout << "   Time      : " << avg_hipsparse_ms << " ms\n";
    std::cout << "   Perf      : " << hipsparse_gflops << " GFLOPS\n";
    std::cout << "   Bandwidth : " << hipsparse_gbps << " GB/s\n\n";

    std::cout << "3. Custom rocWMMA (FP16 Matrix Cores)\n";
    std::cout << "   Time      : " << avg_custom_ms << " ms\n";
    std::cout << "   Perf      : " << custom_gflops << " GFLOPS\n";
    std::cout << "   Bandwidth : " << custom_gbps << " GB/s\n\n";

    double speedup = avg_hipsparse_ms / avg_custom_ms;
    std::cout << ">>> ROCWMMA SPEEDUP VS HIPSPARSE: " << speedup << "x <<<\n";
    std::cout << "========================================================\n";

    // --- 6. Cleanup Memory! ---
    hipsparseDestroy(handle);
    hipEventDestroy(start); hipEventDestroy(stop);
    hipFree(d_bcsr_row_ptr); hipFree(d_bcsr_col_ind); hipFree(d_bcsr_values);
    hipFree(d_csr_row_ptr); hipFree(d_csr_col_ind); hipFree(d_csr_values_fp32);
    hipFree(d_B); hipFree(d_B_fp32); hipFree(d_C_custom); hipFree(d_C_hipsparse);
    hipFree(d_temp_buffer);

    // Return results for the summary table
    return {case_name, hipsparse_gflops, custom_gflops, speedup};
}

int main() {
    int size = 8192;   // 8K x 8K Matrix A
    int N = 2048;      // 2K cols for Matrix B

    std::cout << "Beginning Synthetic Matrix Suite...\n";
    matrix_utils::CSRMatrix<__half> mat;

    // Vector to collect all results
    std::vector<BenchmarkResult> results;

    // 1. Banded Matrices
    mat = matrix_utils::generation::generate_banded<__half>(size, 16);
    results.push_back(run_benchmark_case("Banded (BW=16)", mat, false, N));

    mat = matrix_utils::generation::generate_banded<__half>(size, 64);
    results.push_back(run_benchmark_case("Banded (BW=64)", mat, false, N));

    // 2. Triangular Matrices
    mat = matrix_utils::generation::generate_triangular<__half>(size, true, 0.05f);
    results.push_back(run_benchmark_case("Upper Triangular (Density=0.05)", mat, false, N));

    mat = matrix_utils::generation::generate_triangular<__half>(size, false, 0.20f);
    results.push_back(run_benchmark_case("Lower Triangular (Density=0.20)", mat, false, N));

    mat = matrix_utils::generation::generate_triangular<__half>(size, true, 0.50f);
    results.push_back(run_benchmark_case("Upper Triangular (Density=0.50)", mat, false, N));

    // 3. Hessenberg Matrices
    mat = matrix_utils::generation::generate_hessenberg<__half>(size, true, 0.05f);
    results.push_back(run_benchmark_case("Hessenberg Upper (Density=0.05)", mat, false, N));

    mat = matrix_utils::generation::generate_hessenberg<__half>(size, false, 0.20f);
    results.push_back(run_benchmark_case("Hessenberg Lower (Density=0.20)", mat, false, N));

    // 4. Arrowhead
    mat = matrix_utils::generation::generate_arrowhead<__half>(size, 128);
    results.push_back(run_benchmark_case("Arrowhead (Width=128)", mat, false, N));

    // 5. Block Diagonal
    mat = matrix_utils::generation::generate_block_diagonal<__half>(size, 64, 0.8f);
    results.push_back(run_benchmark_case("Block Diagonal (Block=64, Dens=0.8)", mat, false, N));

    mat = matrix_utils::generation::generate_block_diagonal<__half>(size, 256, 0.9f);
    results.push_back(run_benchmark_case("Block Diagonal (Block=256, Dens=0.9)", mat, false, N));

    // 6. Near Symmetric
    mat = matrix_utils::generation::generate_near_symmetric<__half>(size, 0.05f, 0.1f);
    results.push_back(run_benchmark_case("Near Symmetric (Dens=0.05, DropRate=0.1)", mat, true, N));

    // 7. Power Law
    mat = matrix_utils::generation::generate_power_law<__half>(size, size, 2.5f);
    results.push_back(run_benchmark_case("Power Law / Scale-Free (Alpha=2.5)", mat, true, N));

    // ---------------------------------------------------------
    // SUMMARY TABLE PRINTOUT
    // ---------------------------------------------------------
    std::cout << "\n\n================================================================================\n";
    std::cout << "                              SYNTHETIC SUITE SUMMARY                           \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(42) << "Matrix Type"
              << std::right << std::setw(14) << "rocSPARSE (GF)"
              << std::setw(14) << "rocWMMA (GF)"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(42) << r.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2) << r.hipsparse_gflops
                  << std::setw(14) << r.custom_gflops
                  << std::setw(9) << r.speedup << "x\n";
    }
    std::cout << "================================================================================\n\n";

    return 0;
}