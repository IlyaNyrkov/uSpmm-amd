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

struct ShuffledBenchmarkResult {
    std::string name;
    double hipsparse_gflops;
    double unclustered_gflops;
    double clustered_gflops;
    double speedup_vs_sparse;
    double speedup_vs_unclustered;
};

ShuffledBenchmarkResult run_shuffled_benchmark(const std::string& case_name,
                                               matrix_utils::CSRMatrix<__half>& base_mat,
                                               int N)
{
    std::cout << "\n################################################################\n";
    std::cout << ">>> RUNNING SHUFFLED CASE: " << case_name << "\n";
    std::cout << "################################################################\n";

    int M = base_mat.num_rows;
    int K = base_mat.num_cols;
    int r_block = 16, c_block = 16;

    int padded_M = ((M + r_block - 1) / r_block) * r_block;
    int padded_K = ((K + c_block - 1) / c_block) * c_block;

    auto h_B = matrix_utils::generation::generate_dense<__half>(padded_K, N, 123);
    double total_flops = 2.0 * base_mat.values.size() * N;

    // ---------------------------------------------------------
    // 1. Scramble the Matrix
    // ---------------------------------------------------------
    matrix_utils::CSRMatrix<__half> mat_shuffled = base_mat;
    matrix_utils::generation::shuffle_rows(mat_shuffled, 42); // Scramble it!

    auto bcsr_shuffled = matrix_utils::format::csr_to_bcsr(mat_shuffled, r_block, c_block);
    int blocks_shuffled = bcsr_shuffled.bcsr_values.size() / (r_block * c_block);

    // ---------------------------------------------------------
    // 2. Measure UNCLUSTERED rocWMMA
    // ---------------------------------------------------------
    double unclustered_gflops = 0.0;
    {
        int* d_bcsr_row_ptr, *d_bcsr_col_ind;
        __half* d_bcsr_values, *d_B;
        float* d_C;

        HIP_CHECK(hipMalloc(&d_bcsr_row_ptr, bcsr_shuffled.bcsr_row_ptr.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bcsr_col_ind, bcsr_shuffled.bcsr_col_ind.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bcsr_values,  bcsr_shuffled.bcsr_values.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_B,            h_B.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_C,            padded_M * N * sizeof(float)));

        HIP_CHECK(hipMemcpy(d_bcsr_row_ptr, bcsr_shuffled.bcsr_row_ptr.data(), bcsr_shuffled.bcsr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bcsr_col_ind, bcsr_shuffled.bcsr_col_ind.data(), bcsr_shuffled.bcsr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bcsr_values,  bcsr_shuffled.bcsr_values.data(), bcsr_shuffled.bcsr_values.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice));

        hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop); float total_ms = 0;

        for (int i = 0; i < 3; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, bcsr_shuffled.num_block_rows, d_B, d_C, padded_M, N, padded_K);
        HIP_CHECK(hipDeviceSynchronize());

        hipEventRecord(start);
        for (int i = 0; i < 10; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, bcsr_shuffled.num_block_rows, d_B, d_C, padded_M, N, padded_K);
        hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&total_ms, start, stop);

        unclustered_gflops = total_flops / ((total_ms / 10.0) * 1e6);

        hipEventDestroy(start); hipEventDestroy(stop);
        hipFree(d_bcsr_row_ptr); hipFree(d_bcsr_col_ind); hipFree(d_bcsr_values); hipFree(d_B); hipFree(d_C);
    }

    // ---------------------------------------------------------
    // 3. Cluster and Fix the Matrix
    // ---------------------------------------------------------
    std::cout << "--- Applying Iterative Clustering ---\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto perm = uspmm::computeIterativeClustering(mat_shuffled, 0.0f, 16);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = stop_cpu - start_cpu;

    auto mat_clustered = uspmm::apply_permutation(mat_shuffled, perm);
    auto bcsr_clustered = matrix_utils::format::csr_to_bcsr(mat_clustered, r_block, c_block);
    int blocks_clustered = bcsr_clustered.bcsr_values.size() / (r_block * c_block);

    std::cout << "    Clustering Time : " << cpu_ms.count() << " ms\n";
    std::cout << "    Blocks Shuffled : " << blocks_shuffled << "\n";
    std::cout << "    Blocks Recovered: " << blocks_clustered << " (" << (blocks_shuffled - blocks_clustered) << " blocks eliminated!)\n";

    // ---------------------------------------------------------
    // 4. Measure CLUSTERED rocWMMA
    // ---------------------------------------------------------
    double clustered_gflops = 0.0;
    {
        int* d_bcsr_row_ptr, *d_bcsr_col_ind;
        __half* d_bcsr_values, *d_B;
        float* d_C;

        HIP_CHECK(hipMalloc(&d_bcsr_row_ptr, bcsr_clustered.bcsr_row_ptr.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bcsr_col_ind, bcsr_clustered.bcsr_col_ind.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bcsr_values,  bcsr_clustered.bcsr_values.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_B,            h_B.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_C,            padded_M * N * sizeof(float)));

        HIP_CHECK(hipMemcpy(d_bcsr_row_ptr, bcsr_clustered.bcsr_row_ptr.data(), bcsr_clustered.bcsr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bcsr_col_ind, bcsr_clustered.bcsr_col_ind.data(), bcsr_clustered.bcsr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bcsr_values,  bcsr_clustered.bcsr_values.data(), bcsr_clustered.bcsr_values.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice));

        hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop); float total_ms = 0;

        for (int i = 0; i < 3; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, bcsr_clustered.num_block_rows, d_B, d_C, padded_M, N, padded_K);
        HIP_CHECK(hipDeviceSynchronize());

        hipEventRecord(start);
        for (int i = 0; i < 10; ++i) uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, bcsr_clustered.num_block_rows, d_B, d_C, padded_M, N, padded_K);
        hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&total_ms, start, stop);

        clustered_gflops = total_flops / ((total_ms / 10.0) * 1e6);

        hipEventDestroy(start); hipEventDestroy(stop);
        hipFree(d_bcsr_row_ptr); hipFree(d_bcsr_col_ind); hipFree(d_bcsr_values); hipFree(d_B); hipFree(d_C);
    }

    // ---------------------------------------------------------
    // 5. Measure hipSPARSE Baseline (on the clustered matrix)
    // ---------------------------------------------------------
    double hipsparse_gflops = 0.0;
    {
        std::vector<float> h_csr_values_fp32(mat_clustered.values.size());
        std::vector<float> h_B_fp32(h_B.size());
        for(size_t i=0; i<mat_clustered.values.size(); ++i) h_csr_values_fp32[i] = (float)mat_clustered.values[i];
        for(size_t i=0; i<h_B.size(); ++i) h_B_fp32[i] = (float)h_B[i];

        int* d_csr_row_ptr, *d_csr_col_ind;
        float* d_csr_values_fp32, *d_B_fp32, *d_C_hipsparse;

        HIP_CHECK(hipMalloc(&d_csr_row_ptr, mat_clustered.row_ptr.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_csr_col_ind, mat_clustered.col_ind.size() * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_csr_values_fp32, h_csr_values_fp32.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B_fp32, h_B_fp32.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_C_hipsparse,  padded_M * N * sizeof(float)));

        HIP_CHECK(hipMemcpy(d_csr_row_ptr, mat_clustered.row_ptr.data(), mat_clustered.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_csr_col_ind, mat_clustered.col_ind.data(), mat_clustered.col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
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

        hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop); float total_ms = 0;

        for (int i = 0; i < 3; ++i) HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
        HIP_CHECK(hipDeviceSynchronize());

        hipEventRecord(start);
        for (int i = 0; i < 10; ++i) HIPSPARSE_CHECK(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));
        hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&total_ms, start, stop);

        hipsparse_gflops = total_flops / ((total_ms / 10.0) * 1e6);

        hipsparseDestroy(handle); hipEventDestroy(start); hipEventDestroy(stop);
        hipFree(d_csr_row_ptr); hipFree(d_csr_col_ind); hipFree(d_csr_values_fp32); hipFree(d_B_fp32); hipFree(d_C_hipsparse); hipFree(d_temp_buffer);
    }

    // ---------------------------------------------------------
    // 6. Output Summary for Case
    // ---------------------------------------------------------
    double speedup_sparse = clustered_gflops / hipsparse_gflops;
    double speedup_unclustered = clustered_gflops / unclustered_gflops;

    std::cout << "\n    [ Performance Metrics ]\n";
    std::cout << "      hipSPARSE (FP32)         : " << hipsparse_gflops << " GFLOPS\n";
    std::cout << "      rocWMMA Scrambled (FP16) : " << unclustered_gflops << " GFLOPS\n";
    std::cout << "      rocWMMA Clustered (FP16) : " << clustered_gflops << " GFLOPS\n";
    std::cout << "      Clustering Gain          : " << speedup_unclustered << "x faster than scrambled!\n";

    return {case_name, hipsparse_gflops, unclustered_gflops, clustered_gflops, speedup_sparse, speedup_unclustered};
}

int main() {
    int size = 8192;   // 8K x 8K Matrix A
    int N = 2048;      // 2K cols for Matrix B

    std::cout << "Beginning Scrambled Matrix Clustering Suite...\n";
    matrix_utils::CSRMatrix<__half> mat;
    std::vector<ShuffledBenchmarkResult> results;

    // 1. Banded Matrices
    mat = matrix_utils::generation::generate_banded<__half>(size, 16);
    results.push_back(run_shuffled_benchmark("Banded (BW=16)", mat, N));

    mat = matrix_utils::generation::generate_banded<__half>(size, 64);
    results.push_back(run_shuffled_benchmark("Banded (BW=64)", mat, N));

    // 2. Triangular Matrices
    mat = matrix_utils::generation::generate_triangular<__half>(size, true, 0.05f);
    results.push_back(run_shuffled_benchmark("Upper Triangular (Density=0.05)", mat, N));

    mat = matrix_utils::generation::generate_triangular<__half>(size, false, 0.20f);
    results.push_back(run_shuffled_benchmark("Lower Triangular (Density=0.20)", mat, N));

    // 3. Hessenberg Matrices
    mat = matrix_utils::generation::generate_hessenberg<__half>(size, true, 0.05f);
    results.push_back(run_shuffled_benchmark("Hessenberg Upper (Density=0.05)", mat, N));

    // 4. Arrowhead
    mat = matrix_utils::generation::generate_arrowhead<__half>(size, 128);
    results.push_back(run_shuffled_benchmark("Arrowhead (Width=128)", mat, N));

    // 5. Block Diagonal
    mat = matrix_utils::generation::generate_block_diagonal<__half>(size, 64, 0.8f);
    results.push_back(run_shuffled_benchmark("Block Diagonal (Block=64)", mat, N));

    mat = matrix_utils::generation::generate_block_diagonal<__half>(size, 256, 0.9f);
    results.push_back(run_shuffled_benchmark("Block Diagonal (Block=256)", mat, N));

    // 6. Near Symmetric & Power Law (These are inherently unstructured, but scrambling makes them worst-case)
    mat = matrix_utils::generation::generate_near_symmetric<__half>(size, 0.05f, 0.1f);
    results.push_back(run_shuffled_benchmark("Near Symmetric", mat, N));

    mat = matrix_utils::generation::generate_power_law<__half>(size, size, 2.5f);
    results.push_back(run_shuffled_benchmark("Power Law (Alpha=2.5)", mat, N));

    // ---------------------------------------------------------
    // FINAL SUMMARY TABLE PRINTOUT
    // ---------------------------------------------------------
    std::cout << "\n\n=================================================================================================================\n";
    std::cout << "                                  SHUFFLED CLUSTERING SUITE SUMMARY                                              \n";
    std::cout << "=================================================================================================================\n";
    std::cout << std::left << std::setw(34) << "Matrix Type (Scrambled)"
              << std::right << std::setw(15) << "rocSPARSE GF"
              << std::setw(20) << "Shuffled rocWMMA GF"
              << std::setw(21) << "Clustered rocWMMA GF"
              << std::setw(11) << "vs Sparse"
              << std::setw(11) << " vs Shuffled\n";
    std::cout << "-----------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(34) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(15) << r.hipsparse_gflops
                  << std::setw(20) << r.unclustered_gflops
                  << std::setw(21) << r.clustered_gflops
                  << std::setw(10) << r.speedup_vs_sparse << "x"
                  << std::setw(10) << r.speedup_vs_unclustered << "x\n";
    }
    std::cout << "=================================================================================================================\n\n";

    return 0;
}