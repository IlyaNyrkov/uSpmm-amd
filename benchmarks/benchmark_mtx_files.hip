#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipsparse/hipsparse.h>

#include "matrix_utils/generation.hpp"
#include "matrix_utils/conversion.hpp"
#include "matrix_utils/io.hpp"
#include "uspmm/clustering.hpp"
#include "uspmm/spmm_rocwmma.hpp"

#define HIP_CHECK(expr) { hipError_t err = (expr); if (err != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(err) << "\n"; exit(1); } }
#define HIPSPARSE_CHECK(expr) { hipsparseStatus_t err = (expr); if (err != HIPSPARSE_STATUS_SUCCESS) { std::cerr << "hipSPARSE Error\n"; exit(1); } }

namespace fs = std::filesystem;

int main() {
    std::string base_path = "../test_data_mtx/";

    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        std::cerr << "Error: Directory '" << base_path << "' does not exist.\n";
        return 1;
    }

    std::cout << "========================================================\n";
    std::cout << "         ADVANCED SPMM BENCHMARK SUITE (MI210)          \n";
    std::cout << "========================================================\n\n";

    // Hardware Tile Size
    int r_block = 16, c_block = 16;
    int N = 1024; // Fixed column width for Dense Matrix B

    int warmup_runs = 3;
    int timed_runs = 10;

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            try {
                std::cout << "--------------------------------------------------------\n";
                std::cout << ">>> Processing: " << filename << "\n";

                // ---------------------------------------------------------
                // 1. Matrix Loading & Stats
                // ---------------------------------------------------------
                auto mat_A_csr = matrix_utils::io::read_mtx<__half>(filepath);

                int M = mat_A_csr.num_rows;
                int K = mat_A_csr.num_cols;
                long long total_elements = static_cast<long long>(M) * K;
                long long nnz = mat_A_csr.values.size();
                double sparsity = 100.0 * (1.0 - (static_cast<double>(nnz) / total_elements));

                std::cout << "    Dimensions : " << M << " x " << K << " x " << N << " (M x K x N)\n";
                std::cout << "    NNZ        : " << nnz << "\n";
                std::cout << "    Sparsity   : " << std::fixed << std::setprecision(4) << sparsity << "%\n";

                // Safety padding for hardware blocks
                int padded_M = ((M + r_block - 1) / r_block) * r_block;
                int padded_K = ((K + c_block - 1) / c_block) * c_block;

                auto h_B = matrix_utils::generation::generate_dense<__half>(padded_K, N, 123);

                // ---------------------------------------------------------
                // 2. CPU Clustering & BCSR Conversion
                // ---------------------------------------------------------
                auto start_cpu = std::chrono::high_resolution_clock::now();
                auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 16);
                auto mat_A_reordered = uspmm::apply_permutation(mat_A_csr, perm);
                auto stop_cpu = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> cpu_ms = stop_cpu - start_cpu;

                auto bcsr_before = matrix_utils::format::csr_to_bcsr(mat_A_csr, r_block, c_block);
                auto mat_A_bcsr = matrix_utils::format::csr_to_bcsr(mat_A_reordered, r_block, c_block);

                int blocks_before = bcsr_before.bcsr_values.size() / (r_block * c_block);
                int blocks_after = mat_A_bcsr.bcsr_values.size() / (r_block * c_block);

                std::cout << "    Clustering : " << cpu_ms.count() << " ms\n";
                std::cout << "    Blocks     : " << blocks_before << " -> " << blocks_after
                          << " (" << (blocks_before - blocks_after) << " blocks saved)\n";

                // ---------------------------------------------------------
                // 3. GPU Memory Allocations
                // ---------------------------------------------------------
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
                HIP_CHECK(hipMemset(d_C_custom, 0, padded_M * N * sizeof(float)));

                // Upcast to FP32 for hipSPARSE baseline
                std::vector<float> h_csr_values_fp32(mat_A_reordered.values.size());
                std::vector<float> h_B_fp32(h_B.size());
                for(size_t i=0; i<mat_A_reordered.values.size(); ++i) h_csr_values_fp32[i] = (float)mat_A_reordered.values[i];
                for(size_t i=0; i<h_B.size(); ++i) h_B_fp32[i] = (float)h_B[i];

                HIP_CHECK(hipMalloc(&d_csr_row_ptr, mat_A_reordered.row_ptr.size() * sizeof(int)));
                HIP_CHECK(hipMalloc(&d_csr_col_ind, mat_A_reordered.col_ind.size() * sizeof(int)));
                HIP_CHECK(hipMalloc(&d_csr_values_fp32, h_csr_values_fp32.size() * sizeof(float)));
                HIP_CHECK(hipMalloc(&d_B_fp32, h_B_fp32.size() * sizeof(float)));
                HIP_CHECK(hipMalloc(&d_C_hipsparse, padded_M * N * sizeof(float)));

                HIP_CHECK(hipMemcpy(d_csr_row_ptr, mat_A_reordered.row_ptr.data(), mat_A_reordered.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_csr_col_ind, mat_A_reordered.col_ind.data(), mat_A_reordered.col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_csr_values_fp32, h_csr_values_fp32.data(), h_csr_values_fp32.size() * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_B_fp32, h_B_fp32.data(), h_B_fp32.size() * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_C_hipsparse, 0, padded_M * N * sizeof(float)));

                // Setup hipSPARSE
                hipsparseHandle_t handle;
                HIPSPARSE_CHECK(hipsparseCreate(&handle));

                hipsparseSpMatDescr_t descrA;
                HIPSPARSE_CHECK(hipsparseCreateCsr(&descrA, M, K, h_csr_values_fp32.size(), d_csr_row_ptr, d_csr_col_ind, d_csr_values_fp32, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));

                hipsparseDnMatDescr_t descrB, descrC;
                // Note: We tell hipSPARSE the exact M and K sizes, even though our underlying buffers are safely padded
                HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrB, K, N, N, d_B_fp32, HIP_R_32F, HIPSPARSE_ORDER_ROW));
                HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrC, M, N, N, d_C_hipsparse, HIP_R_32F, HIPSPARSE_ORDER_ROW));

                float alpha = 1.0f, beta = 0.0f;
                size_t buffer_size = 0;
                void* d_temp_buffer = nullptr;

                HIPSPARSE_CHECK(hipsparseSpMM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
                HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

                // ---------------------------------------------------------
                // 4. Benchmarking
                // ---------------------------------------------------------
                hipEvent_t start, stop;
                hipEventCreate(&start); hipEventCreate(&stop);
                float total_ms = 0.0f;

                double total_flops = 2.0 * mat_A_csr.values.size() * N;
                double bytes_A_bcsr = (mat_A_bcsr.bcsr_values.size() * 2.0) + (mat_A_bcsr.bcsr_row_ptr.size() * 4.0) + (mat_A_bcsr.bcsr_col_ind.size() * 4.0);
                double bytes_B = K * N * 2.0;
                double bytes_C = M * N * 4.0;
                double total_bytes = bytes_A_bcsr + bytes_B + bytes_C;

                // --- Custom rocWMMA ---
                for (int i = 0; i < warmup_runs; ++i) {
                    uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, padded_M, N, padded_K);
                }
                HIP_CHECK(hipDeviceSynchronize());

                hipEventRecord(start);
                for (int i = 0; i < timed_runs; ++i) {
                    uspmm::kernel::run_spmm_rocwmma_f16(d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values, mat_A_bcsr.num_block_rows, d_B, d_C_custom, padded_M, N, padded_K);
                }
                hipEventRecord(stop);
                hipEventSynchronize(stop);
                hipEventElapsedTime(&total_ms, start, stop);

                double avg_custom_ms = total_ms / timed_runs;
                double custom_gflops = (total_flops / (avg_custom_ms * 1e6));
                double custom_gbps = (total_bytes / (avg_custom_ms * 1e6));

                // --- hipSPARSE Baseline ---
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
                double hipsparse_gbps = (total_bytes / (avg_hipsparse_ms * 1e6));

                // ---------------------------------------------------------
                // 5. Output
                // ---------------------------------------------------------
                std::cout << "\n    [ Performance Metrics ]\n";
                std::cout << "      hipSPARSE (FP32) : " << avg_hipsparse_ms << " ms | "
                          << hipsparse_gflops << " GFLOPS | " << hipsparse_gbps << " GB/s\n";
                std::cout << "      rocWMMA   (FP16) : " << avg_custom_ms << " ms | "
                          << custom_gflops << " GFLOPS | " << custom_gbps << " GB/s\n";
                std::cout << "      SPEEDUP          : " << (avg_hipsparse_ms / avg_custom_ms) << "x\n\n";

                // ---------------------------------------------------------
                // 6. Memory Cleanup
                // ---------------------------------------------------------
                hipsparseDestroy(handle);
                hipEventDestroy(start);
                hipEventDestroy(stop);

                hipFree(d_bcsr_row_ptr); hipFree(d_bcsr_col_ind); hipFree(d_bcsr_values);
                hipFree(d_csr_row_ptr); hipFree(d_csr_col_ind); hipFree(d_csr_values_fp32);
                hipFree(d_B); hipFree(d_B_fp32);
                hipFree(d_C_custom); hipFree(d_C_hipsparse);
                hipFree(d_temp_buffer);

            } catch (const std::exception& e) {
                std::cerr << ">>> Failed to process " << filename << ": " << e.what() << "\n\n";
            }
        }
    }

    return 0;
}