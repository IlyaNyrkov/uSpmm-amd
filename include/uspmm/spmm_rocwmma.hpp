#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include "matrix_utils/types.hpp"

namespace uspmm {
    namespace kernel {

        void run_spmm_rocwmma_f16(const int* d_bcsr_row_ptr, const int* d_bcsr_col_ind, const __half* d_bcsr_values,
                                  int num_block_rows_a, const __half* d_B, float* d_C,
                                  int M, int N, int K);


    } // namespace kernel
} // namespace uspmm