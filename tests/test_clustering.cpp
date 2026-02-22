#include <gtest/gtest.h>
#include "uspmm/clustering.hpp"
#include "matrix_utils/types.hpp"

// Test the Jaccard Distance calculation
TEST(ClusteringTest, JaccardDistance) {
    uspmm::BitVector v1 = {0b101}; // popcount = 2
    uspmm::BitVector v2 = {0b001}; // popcount = 1

    // Union = 0b101 (2 bits), Intersection = 0b001 (1 bit)
    // Distance = 1.0 - (1.0 / 2.0) = 0.5
    EXPECT_FLOAT_EQ(uspmm::computeJaccardDistance(v1, v2), 0.5f);

    uspmm::BitVector empty1 = {0};
    uspmm::BitVector empty2 = {0};
    // Expected distance for empty vectors is 0.0
    EXPECT_FLOAT_EQ(uspmm::computeJaccardDistance(empty1, empty2), 0.0f);
}

// Test the Clustering logic on a small matrix
TEST(ClusteringTest, IterativeClustering) {
    matrix_utils::CSRMatrix<float> mat;
    mat.num_rows = 3;
    mat.num_cols = 128;

    // Row 0: elements at col 0, 1
    // Row 1: elements at col 100, 101 (completely different)
    // Row 2: elements at col 0, 2 (very similar to Row 0)
    mat.row_ptr = {0, 2, 4, 6};
    mat.col_ind = {0, 1, 100, 101, 0, 2};
    mat.values  = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    // Use a block width of 1 and a threshold of 0.8
    // Row 0 and Row 2 should be clustered together.
    auto permutation = uspmm::computeIterativeClustering(mat, 0.8f, 1);

    // Ensure all rows are present in the final permutation
    EXPECT_EQ(permutation.size(), 3);

    // Sort and check that indices 0, 1, 2 exist
    std::vector<int> sorted_perm = permutation;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    EXPECT_EQ(sorted_perm[0], 0);
    EXPECT_EQ(sorted_perm[1], 1);
    EXPECT_EQ(sorted_perm[2], 2);
}

// Test applying a permutation array to a CSR matrix
TEST(ClusteringTest, ApplyPermutation) {
    matrix_utils::CSRMatrix<float> mat;
    mat.num_rows = 3;
    mat.num_cols = 3;

    // Constructing a simple matrix:
    // Row 0: [1.0, 0.0, 0.0]  (col 0)
    // Row 1: [0.0, 2.0, 0.0]  (col 1)
    // Row 2: [0.0, 0.0, 3.0]  (col 2)
    mat.row_ptr = {0, 1, 2, 3};
    mat.col_ind = {0, 1, 2};
    mat.values  = {1.0f, 2.0f, 3.0f};

    // Permutation: new row 0 is old row 2, new row 1 is old row 0, new row 2 is old row 1
    std::vector<int> perm = {2, 0, 1};

    // Apply the permutation
    auto permuted_mat = uspmm::apply_permutation(mat, perm);

    // Check dimensions
    EXPECT_EQ(permuted_mat.num_rows, 3);
    EXPECT_EQ(permuted_mat.num_cols, 3);

    // Expected new matrix:
    // Row 0 (old 2): [0.0, 0.0, 3.0] -> col 2
    // Row 1 (old 0): [1.0, 0.0, 0.0] -> col 0
    // Row 2 (old 1): [0.0, 2.0, 0.0] -> col 1

    // Check row pointers
    std::vector<int> expected_row_ptr = {0, 1, 2, 3};
    EXPECT_EQ(permuted_mat.row_ptr, expected_row_ptr);

    // Check column indices
    std::vector<int> expected_col_ind = {2, 0, 1};
    EXPECT_EQ(permuted_mat.col_ind, expected_col_ind);

    // Check values
    std::vector<float> expected_values = {3.0f, 1.0f, 2.0f};
    EXPECT_EQ(permuted_mat.values, expected_values);
}