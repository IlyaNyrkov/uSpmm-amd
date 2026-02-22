#include <gtest/gtest.h>
#include "matrix_utils/types.hpp"
#include "matrix_utils/conversion.hpp"

TEST(FormatConversionTest, CSRtoBCSR) {
    matrix_utils::CSRMatrix<float> csr;
    csr.num_rows = 4;
    csr.num_cols = 4;

    // Matrix visual representation:
    // [ 1.0  0.0 | 0.0  0.0 ]
    // [ 0.0  2.0 | 3.0  0.0 ]
    // -----------------------
    // [ 0.0  0.0 | 0.0  4.0 ]
    // [ 0.0  0.0 | 0.0  0.0 ]

    csr.row_ptr = {0, 1, 3, 4, 4};
    csr.col_ind = {0, 1, 2, 3};
    csr.values  = {1.0f, 2.0f, 3.0f, 4.0f};

    // Convert using 2x2 blocks
    auto bcsr = matrix_utils::format::csr_to_bcsr(csr, 2, 2);

    // Block dimensions should be 2x2 (since original is 4x4)
    EXPECT_EQ(bcsr.num_block_rows, 2);
    EXPECT_EQ(bcsr.num_block_cols, 2);
    EXPECT_EQ(bcsr.r_block, 2);
    EXPECT_EQ(bcsr.c_block, 2);

    // Block row 0 has non-zeros in block col 0 and block col 1
    // Block row 1 has non-zeros only in block col 1
    std::vector<int> expected_bcsr_row_ptr = {0, 2, 3};
    EXPECT_EQ(bcsr.bcsr_row_ptr, expected_bcsr_row_ptr);

    // The active block columns are:
    // BR 0: BC 0, BC 1
    // BR 1: BC 1
    std::vector<int> expected_bcsr_col_ind = {0, 1, 1};
    EXPECT_EQ(bcsr.bcsr_col_ind, expected_bcsr_col_ind);

    // The values are stored as dense 2x2 blocks (row-major).
    // Block 0 (BR 0, BC 0): [1.0, 0.0, 0.0, 2.0]
    // Block 1 (BR 0, BC 1): [0.0, 0.0, 3.0, 0.0]
    // Block 2 (BR 1, BC 1): [0.0, 4.0, 0.0, 0.0]
    std::vector<float> expected_bcsr_values = {
        1.0f, 0.0f, 0.0f, 2.0f,  // Block 0
        0.0f, 0.0f, 3.0f, 0.0f,  // Block 1
        0.0f, 4.0f, 0.0f, 0.0f   // Block 2
    };

    EXPECT_EQ(bcsr.bcsr_values, expected_bcsr_values);
}