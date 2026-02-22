#pragma once
#include <vector>
#include <algorithm>
#include "matrix_utils/types.hpp"

namespace matrix_utils {
namespace format {

    template <typename T>
    BCSRMatrix<T> csr_to_bcsr(const CSRMatrix<T>& csr, int r_block, int c_block) {
        BCSRMatrix<T> bcsr;
        bcsr.r_block = r_block;
        bcsr.c_block = c_block;

        bcsr.num_block_rows = (csr.num_rows + r_block - 1) / r_block;
        bcsr.num_block_cols = (csr.num_cols + c_block - 1) / c_block;

        bcsr.bcsr_row_ptr.reserve(bcsr.num_block_rows + 1);
        bcsr.bcsr_row_ptr.push_back(0);

        // Fast lookup to see if a block column is already active for the current block row.
        // block_row index is used as the marker to avoid re-initializing the array.
        std::vector<int> block_map(bcsr.num_block_cols, -1);

        int current_block_count = 0;

        for (int br = 0; br < bcsr.num_block_rows; ++br) {
            std::vector<int> active_block_cols;

            int r_start = br * r_block;
            int r_end = std::min(csr.num_rows, r_start + r_block);

            // Phase 1: Identify which block columns have non-zeros in this block row
            for (int r = r_start; r < r_end; ++r) {
                for (int idx = csr.row_ptr[r]; idx < csr.row_ptr[r+1]; ++idx) {
                    int c = csr.col_ind[idx];
                    int bc = c / c_block;

                    if (block_map[bc] != br) {
                        block_map[bc] = br;
                        active_block_cols.push_back(bc);
                    }
                }
            }

            // BCSR requires column indices to be sorted
            std::sort(active_block_cols.begin(), active_block_cols.end());

            for (int bc : active_block_cols) {
                bcsr.bcsr_col_ind.push_back(bc);
            }

            int num_blocks_in_row = active_block_cols.size();
            current_block_count += num_blocks_in_row;
            bcsr.bcsr_row_ptr.push_back(current_block_count);

            // Phase 2: Allocate zeros for these blocks and scatter the values
            int values_start_offset = bcsr.bcsr_values.size();
            int elements_to_add = num_blocks_in_row * r_block * c_block;
            bcsr.bcsr_values.resize(values_start_offset + elements_to_add, static_cast<T>(0));

            for (int r = r_start; r < r_end; ++r) {
                for (int idx = csr.row_ptr[r]; idx < csr.row_ptr[r+1]; ++idx) {
                    int c = csr.col_ind[idx];
                    T val = csr.values[idx];

                    int bc = c / c_block;
                    int intra_block_r = r % r_block;
                    int intra_block_c = c % c_block;

                    // Find the relative index of this block column within the active blocks
                    auto it = std::lower_bound(active_block_cols.begin(), active_block_cols.end(), bc);
                    int block_offset_in_row = std::distance(active_block_cols.begin(), it);

                    // Calculate 1D index inside bcsr_values
                    int global_block_idx = bcsr.bcsr_row_ptr[br] + block_offset_in_row;
                    int flat_idx = (global_block_idx * r_block * c_block) +
                                   (intra_block_r * c_block) +
                                   intra_block_c;

                    bcsr.bcsr_values[flat_idx] = val;
                }
            }
        }

        return bcsr;
    }

} // namespace format
} // namespace matrix_utils