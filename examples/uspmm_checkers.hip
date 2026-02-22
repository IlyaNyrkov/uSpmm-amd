#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipsparse/hipsparse.h>

#include "matrix_utils/generation.hpp"
#include "matrix_utils/conversion.hpp"
#include "uspmm/clustering.hpp"
#include "uspmm/spmm_rocwmma.hpp"

#define HIP_CHECK(expr) { \
    hipError_t err = (expr); \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << "\n"; \
        exit(1); \
    } \
}

#define HIPSPARSE_CHECK(expr) { \
    hipsparseStatus_t err = (expr); \
    if (err != HIPSPARSE_STATUS_SUCCESS) { \
        std::cerr << "hipSPARSE Error at line " << __LINE__ << "\n"; \
        exit(1); \
    } \
}

int main() {
    int M = 1024; // Rows of A
    int K = 1024; // Cols of A / Rows of B
    int N = 1024; // Cols of B

    // 1. Generate Checkers Matrix A and Dense Matrix B
    std::cout << "1. Generating Matrices...\n";
    auto mat_A_csr = matrix_utils::generation::generate_checkers<__half>(M, K, 2);
    auto h_B = matrix_utils::generation::generate_dense<__half>(K, N, 123);

    // 2. Rearrange and convert A to BCSR
    std::cout << "2. Clustering and Converting A to BCSR...\n";
    auto perm = uspmm::computeIterativeClustering(mat_A_csr, 0.0f, 2);
    auto mat_A_reordered = uspmm::apply_permutation(mat_A_csr, perm);

    // Hardware macro-blocks for MI210 are 16x16
    int r_block = 16, c_block = 16;
    auto mat_A_bcsr = matrix_utils::format::csr_to_bcsr(mat_A_reordered, r_block, c_block);

    // 3. Allocate Device Memory
    std::cout << "3. Allocating Device Memory...\n";
    int* d_bcsr_row_ptr, *d_bcsr_col_ind;
    __half* d_bcsr_values, *d_B;
    float* d_C_custom, *d_C_hipsparse;

    HIP_CHECK(hipMalloc(&d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bcsr_values,  mat_A_bcsr.bcsr_values.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B,            h_B.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C_custom,     M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C_hipsparse,  M * N * sizeof(float)));

    // Copy BCSR structures
    HIP_CHECK(hipMemcpy(d_bcsr_row_ptr, mat_A_bcsr.bcsr_row_ptr.data(), mat_A_bcsr.bcsr_row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_col_ind, mat_A_bcsr.bcsr_col_ind.data(), mat_A_bcsr.bcsr_col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bcsr_values,  mat_A_bcsr.bcsr_values.data(), mat_A_bcsr.bcsr_values.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Copy B and initialize Cs
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C_custom, 0, M * N * sizeof(float)));
    HIP_CHECK(hipMemset(d_C_hipsparse, 0, M * N * sizeof(float)));

    // 4. Run Custom rocWMMA Kernel
    std::cout << "4. Running Custom rocWMMA BCSR Kernel...\n";
    uspmm::kernel::run_spmm_rocwmma_f16(
        d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values,
        mat_A_bcsr.num_block_rows, d_B, d_C_custom, M, N, K
    );
    HIP_CHECK(hipDeviceSynchronize());

// 5. Run hipSPARSE Baseline (Upcasted to FP32 to avoid rocSPARSE FP16 CSR bug)
    std::cout << "5. Running hipSPARSE Baseline (Upcasted to FP32)...\n";

    // Upcast CSR A to FP32 for the baseline
    std::vector<float> h_csr_values_fp32(mat_A_reordered.values.size());
    for(size_t i = 0; i < mat_A_reordered.values.size(); ++i) {
        h_csr_values_fp32[i] = static_cast<float>(mat_A_reordered.values[i]);
    }

    // Upcast Dense B to FP32 for the baseline
    std::vector<float> h_B_fp32(h_B.size());
    for(size_t i = 0; i < h_B.size(); ++i) {
        h_B_fp32[i] = static_cast<float>(h_B[i]);
    }

    // Allocate Device Memory for FP32 baseline
    int* d_csr_row_ptr, *d_csr_col_ind;
    float* d_csr_values_fp32, *d_B_fp32;

    HIP_CHECK(hipMalloc(&d_csr_row_ptr, mat_A_reordered.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_col_ind, mat_A_reordered.col_ind.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_csr_values_fp32, h_csr_values_fp32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B_fp32, h_B_fp32.size() * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_csr_row_ptr, mat_A_reordered.row_ptr.data(), mat_A_reordered.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_col_ind, mat_A_reordered.col_ind.data(), mat_A_reordered.col_ind.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_values_fp32, h_csr_values_fp32.data(), h_csr_values_fp32.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_fp32, h_B_fp32.data(), h_B_fp32.size() * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Notice we are passing HIP_R_32F (FP32) to everything now
    hipsparseSpMatDescr_t descrA;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&descrA, M, K, h_csr_values_fp32.size(),
                                       d_csr_row_ptr, d_csr_col_ind, d_csr_values_fp32,
                                       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));

    hipsparseDnMatDescr_t descrB, descrC;
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrB, K, N, N, d_B_fp32, HIP_R_32F, HIPSPARSE_ORDER_ROW));
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&descrC, M, N, N, d_C_hipsparse, HIP_R_32F, HIPSPARSE_ORDER_ROW));

    float alpha = 1.0f, beta = 0.0f;
    size_t buffer_size = 0;
    void* d_temp_buffer = nullptr;

    HIPSPARSE_CHECK(hipsparseSpMM_bufferSize(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

    HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

    HIPSPARSE_CHECK(hipsparseSpMM(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrA, descrB, &beta, descrC, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, d_temp_buffer));

    HIP_CHECK(hipDeviceSynchronize());

    // 6. Compare Results
    std::cout << "6. Validating Results...\n";
    std::vector<float> h_C_custom(M * N);
    std::vector<float> h_C_hipsparse(M * N);
    HIP_CHECK(hipMemcpy(h_C_custom.data(), d_C_custom, M * N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_C_hipsparse.data(), d_C_hipsparse, M * N * sizeof(float), hipMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float val_custom = h_C_custom[i * N + j];
            float val_hipsparse = h_C_hipsparse[i * N + j];

            // Compare FP16 hardware accumulation vs FP32 software accumulation
            if (std::abs(val_custom - val_hipsparse) > 1e-2) {
                std::cerr << "Mismatch at [Row " << i << ", Col " << j << "]: "
                          << "Custom rocWMMA (FP16) = " << val_custom << ", "
                          << "hipSPARSE (FP32) = " << val_hipsparse << "\n";
                match = false;
            }
        }
    }

    if (match) {
        std::cout << ">>> SUCCESS: Custom rocWMMA kernel matches hipSPARSE perfectly!\n";
    } else {
        std::cout << ">>> FAILED: Discrepancies found.\n";
    }



    // Cleanup
    hipsparseDestroy(handle);
    hipFree(d_bcsr_row_ptr); hipFree(d_bcsr_col_ind); hipFree(d_bcsr_values);
    hipFree(d_csr_row_ptr); hipFree(d_csr_col_ind); hipFree(d_csr_values_fp32);
    hipFree(d_B); hipFree(d_B_fp32); hipFree(d_C_custom); hipFree(d_C_hipsparse);
    hipFree(d_temp_buffer);

    return 0;
}