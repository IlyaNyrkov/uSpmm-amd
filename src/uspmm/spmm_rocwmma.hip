#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include "uspmm/spmm_rocwmma.hpp"

using namespace rocwmma;

const int BLOCK_M = 16;
const int BLOCK_N = 16;
const int BLOCK_K = 16;

namespace uspmm {
namespace kernel {

__global__ void bcsr_spmm_kernel(
    const int* bcsr_row_ptr, const int* bcsr_col_ind, const __half* bcsr_values,
    const __half* B, float* C, int num_block_rows_a, int num_block_cols_b, int ldb, int ldc)
{
    int lane_id = threadIdx.x;

    int block_row_A = blockIdx.y;
    int block_col_B = blockIdx.x;

    if (block_row_A >= num_block_rows_a || block_col_B >= num_block_cols_b) return;

    __shared__ __half shared_A[BLOCK_M * BLOCK_K];
    __shared__ __half shared_B[BLOCK_K * BLOCK_N];

    fragment<matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, __half, row_major> fragA;
    fragment<matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, __half, row_major> fragB;
    fragment<accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> fragC;
    fill_fragment(fragC, 0.0f);

    int start_idx = bcsr_row_ptr[block_row_A];
    int end_idx   = bcsr_row_ptr[block_row_A + 1];

    for (int i = start_idx; i < end_idx; ++i) {
        int col_A = bcsr_col_ind[i];

        const __half* ptr_A = bcsr_values + (i * BLOCK_M * BLOCK_K);
        const __half* ptr_B = B + (col_A * BLOCK_K * ldb) + (block_col_B * BLOCK_N);

        for(int elem = 0; elem < 4; ++elem) {
            int flat_idx = lane_id + (elem * 64);
            int row = flat_idx / 16;
            int col = flat_idx % 16;

            shared_A[flat_idx] = ptr_A[flat_idx];
            shared_B[flat_idx] = ptr_B[row * ldb + col];
        }

        __syncthreads();

        load_matrix_sync(fragA, shared_A, 16);
        load_matrix_sync(fragB, shared_B, 16);

        mma_sync(fragC, fragA, fragB, fragC);

        __syncthreads();
    }

    float* ptr_C = C + (block_row_A * BLOCK_M * ldc) + (block_col_B * BLOCK_N);
    store_matrix_sync(ptr_C, fragC, ldc, mem_row_major);
}

void run_spmm_rocwmma_f16(const int* d_bcsr_row_ptr,
                          const int* d_bcsr_col_ind,
                          const __half* d_bcsr_values,
                          int num_block_rows_a,
                          const __half* d_B,
                          float* d_C,
                          int M, int N, int K)
{
    int num_block_cols_b = (N + BLOCK_N - 1) / BLOCK_N;

    // EXACTLY 1 wavefront per block.
    int threads_per_block = 64;

    dim3 block(threads_per_block);
    // Grid maps 1:1 to output tiles
    dim3 grid(num_block_cols_b, num_block_rows_a);

    bcsr_spmm_kernel<<<grid, block>>>(
        d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values,
        d_B, d_C, num_block_rows_a, num_block_cols_b, N, N
    );
}

} // namespace kernel
} // namespace uspmm