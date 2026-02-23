#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipsparse/hipsparse.h>

#include "matrix_utils/generation.hpp"
#include "matrix_utils/conversion.hpp"
#include "uspmm/clustering.hpp"
#include "uspmm/spmm_rocwmma.hpp"
#include <rocwmma/rocwmma.hpp>

#define HIP_CHECK(expr) { hipError_t err = (expr); if (err != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n"; exit(1); } }
#define HIPSPARSE_CHECK(expr) { hipsparseStatus_t err = (expr); if (err != HIPSPARSE_STATUS_SUCCESS) { std::cerr << "hipSPARSE Error\n"; exit(1); } }


int main() {
    int M = 16384;
    int K = 16384;
    int N = 16384;

    std::cout << "--- Initialization ---\n";
    //auto mat_A_csr = matrix_utils::generation::generate_checkers<__half>(M, K, 4);
    auto mat_A_csr = matrix_utils::generation::generate_triangular<__half>(M, false, 1.0, 4);

    auto h_B = matrix_utils::generation::generate_dense<__half>(K, N, 123);

    // ---------------------------------------------------------
    // 1. Benchmark: CPU Clustering
    // ---------------------------------------------------------
    std::cout << "--- Benchmarking CPU Clustering ---\n";
    int warmup_runs = 0;
    int timed_runs = 1;

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 4);
    }

    // Timed
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 4);
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_ms = stop_cpu - start_cpu;
    double avg_clustering_ms = cpu_ms.count() / timed_runs;

    // count blocks after permutation
    auto mat_A_bcsr = matrix_utils::format::csr_to_bcsr(mat_A_csr, 16, 16);
    std::cout << "Matrix A has " << mat_A_bcsr.bcsr_values.size() / (16 * 16) << " blocks before reordering" << std::endl;

    // Apply permutation and convert for the GPU benchmarks
    auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 4);
    auto mat_A_reordered = uspmm::apply_permutation(mat_A_csr, perm);
    mat_A_bcsr = matrix_utils::format::csr_to_bcsr(mat_A_reordered, 16, 16);

    std::cout << "Matrix A has " << mat_A_bcsr.bcsr_values.size() / (16 * 16) << " blocks after reordering" << std::endl;
    long int reduction_ratio = M * K - mat_A_bcsr.bcsr_values.size();
    std::cout << "Matrix A had " <<  M * K << " elements" << std::endl;
    std::cout << "Matrix A val count reduced by " << reduction_ratio << " elements" << std::endl;

    // ---------------------------------------------------------
    // 2. GPU Setup & Memory Allocations
    // ---------------------------------------------------------
    int* d_bcsr_row_ptr, *d_bcsr_col_ind, *d_csr_row_ptr, *d_csr_col_ind;
    __half* d_bcsr_values, *d_B;
    float* d_C_custom, *d_C_hipsparse, *d_csr_values_fp32, *d_B_fp32;

    HIP_CHECK(hipMalloc(&d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_values,  mat_A_bcsr.bcsr_values.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B,            h_B.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C_custom,     M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.data(), mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.data(), mat_A_bcsr.bcsr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_values,  mat_A_bcsr.bcsr_values.data(), mat_A_bcsr.bcsr_values.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Upcast to FP32 for hipSPARSE baseline
    std::vector<float> h_csr_values_fp32(mat_A_reordered.values.size());
    std::vector<float> h_B_fp32(h_B.size());
    for(size_t i=0; i<mat_A_reordered.values.size(); ++i) h_csr_values_fp32[i] = (float)mat_A_reordered.values[i];
    for(size_t i=0; i<h_B.size(); ++i) h_B_fp32[i] = (float)h_B[i];

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, mat_A_reordered.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, mat_A_reordered.col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_values_fp32, h_csr_values_fp32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B_fp32, h_B_fp32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C_hipsparse,  M * N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_csr_row_ptr, mat_A_reordered.row_ptr.data(), mat_A_reordered.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, mat_A_reordered.col_ind.data(), mat_A_reordered.col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_values_fp32, h_csr_values_fp32.data(), h_csr_values_fp32.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_fp32, h_B_fp32.data(), h_B_fp32.size() * sizeof(float), hipMemcpyHostToDevice));

    // Setup hipSPARSE
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

    // ---------------------------------------------------------
    // GPU Timing Utilities
    // ---------------------------------------------------------
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    float total_ms = 0.0f;

    // --- Math for Metrics ---
    // FLOPs: 2 * NNZ * N
    double total_flops = 2.0 * mat_A_csr.values.size() * N;

    // Bytes (Minimum): Size A + Size B + Size C
    double bytes_A_bcsr = (mat_A_bcsr.bcsr_values.size() * 2.0) + (mat_A_bcsr.bcsr_row_ptr.size() * 4.0) + (mat_A_bcsr.bcsr_col_ind.size() * 4.0);
    double bytes_B = K * N * 2.0; // FP16
    double bytes_C = M * N * 4.0; // FP32
    double total_bytes = bytes_A_bcsr + bytes_B + bytes_C;

    // ---------------------------------------------------------
    // 3. Benchmark: Custom rocWMMA (FP16/FP32 Matrix Cores)
    // ---------------------------------------------------------
    std::cout << "--- Benchmarking Custom rocWMMA Kernel ---\n";
    for (int i = 0; i < warmup_runs; ++i) {
        uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, M, N, K);
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEventRecord(start);
    for (int i = 0; i < timed_runs; ++i) {
        uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, M, N, K);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_ms, start, stop);

    double avg_custom_ms = total_ms / timed_runs;
    double custom_gflops = (total_flops / (avg_custom_ms * 1e6));
    double custom_gbps = (total_bytes / (avg_custom_ms * 1e6));

    // ---------------------------------------------------------
    // 4. Benchmark: hipSPARSE (FP32 baseline)
    // ---------------------------------------------------------
    std::cout << "--- Benchmarking hipSPARSE Baseline ---\n";
    for (int i = 0; i < warmup_runs; ++i) {
        HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEventRecord(start);
    for (int i = 0; i < timed_runs; ++i) {
        HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_ms, start, stop);

    double avg_hipsparse_ms = total_ms / timed_runs;
    double hipsparse_gflops = (total_flops / (avg_hipsparse_ms * 1e6));
    double hipsparse_gbps = (total_bytes / (avg_hipsparse_ms * 1e6)); // Rough estimate for comparison

    // ---------------------------------------------------------
    // 5. Results Printout
    // ---------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << "                   BENCHMARK RESULTS                    \n";
    std::cout << "   Matrix Dimensions : " << M << " x " << K << " x " << N << "\n";
    std::cout << "   Runs per Kernel   : " << timed_runs << "\n";
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

    // Cleanup...
    hipsparseDestroy(handle);
    hipEventDestroy(start); hipEventDestroy(stop);
    return 0;
}