#pragma once
#include <vector>

namespace matrix_utils {

    template <typename T>
    struct CSRMatrix {
        int num_rows;
        int num_cols;
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<T> values;
    };

    template <typename T>
    struct BCSRMatrix {
        int num_block_rows;
        int num_block_cols;
        int r_block; // Row size of each dense block (e.g., 16)
        int c_block; // Col size of each dense block (e.g., 16)

        std::vector<int> bcsr_row_ptr;
        std::vector<int> bcsr_col_ind;
        std::vector<T> bcsr_values; // Stored as contiguous dense blocks (row-major inside)
    };

} // namespace matrix_utils