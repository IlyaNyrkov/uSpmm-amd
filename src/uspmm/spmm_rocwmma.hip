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
    const int* bcsr_row_ptr,
    const int* bcsr_col_ind,
    const __half* bcsr_values,
    const __half* B,
    float* C,
    int num_block_rows_a,
    int num_block_cols_b,
    int ldb,
    int ldc
) {
    int warp_id = threadIdx.x / 64;
    int block_row_A = blockIdx.y * (blockDim.x / 64) + warp_id;
    int block_col_B = blockIdx.x;

    if (block_row_A >= num_block_rows_a || block_col_B >= num_block_cols_b) {
        return;
    }

    // 1. Double-Buffered Hardware Fragments
    // We allocate arrays of size 2 for the inputs to act as our pipeline buffers
    fragment<matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, __half, row_major> fragA[2];
    fragment<matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, __half, row_major> fragB[2];
    fragment<accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> fragC;

    fill_fragment(fragC, 0.0f);

    int start_idx = bcsr_row_ptr[block_row_A];
    int end_idx   = bcsr_row_ptr[block_row_A + 1];
    int num_blocks = end_idx - start_idx;

    if (num_blocks > 0) {
        // --- PROLOGUE ---
        // Pre-load the very first block into Buffer 0
        int col_A = bcsr_col_ind[start_idx];
        const __half* ptr_A = bcsr_values + (start_idx * BLOCK_M * BLOCK_K);
        const __half* ptr_B = B + (col_A * BLOCK_K * ldb) + (block_col_B * BLOCK_N);

        load_matrix_sync(fragA[0], ptr_A, BLOCK_K);
        load_matrix_sync(fragB[0], ptr_B, ldb);

        // --- MAIN PIPELINE LOOP ---
        for (int i = 0; i < num_blocks - 1; ++i) {
            int curr = i % 2;        // Buffer holding data for the current math
            int next = (i + 1) % 2;  // Buffer we will load the NEXT data into

            // Calculate pointers for the NEXT block
            int next_col_A = bcsr_col_ind[start_idx + i + 1];
            const __half* next_ptr_A = bcsr_values + ((start_idx + i + 1) * BLOCK_M * BLOCK_K);
            const __half* next_ptr_B = B + (next_col_A * BLOCK_K * ldb) + (block_col_B * BLOCK_N);

            // Overlap Memory fetch and Math!
            // The compiler sees these target different registers and schedules them concurrently
            load_matrix_sync(fragA[next], next_ptr_A, BLOCK_K);
            load_matrix_sync(fragB[next], next_ptr_B, ldb);

            mma_sync(fragC, fragA[curr], fragB[curr], fragC);
        }

        // --- EPILOGUE ---
        // Drain the pipeline by computing the math on the very last loaded block
        int last = (num_blocks - 1) % 2;
        mma_sync(fragC, fragA[last], fragB[last], fragC);
    }

    // Store Result
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
    int threads_per_block = 256;
    int waves_per_block = threads_per_block / 64;

    dim3 block(threads_per_block);
    dim3 grid(num_block_cols_b, (num_block_rows_a + waves_per_block - 1) / waves_per_block);

    bcsr_spmm_kernel<<<grid, block>>>(
        d_bcsr_row_ptr, d_bcsr_col_ind, d_bcsr_values,
        d_B, d_C, num_block_rows_a, num_block_cols_b, N, N
    );
}

} // namespace kernel
} // namespace uspmm