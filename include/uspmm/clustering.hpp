#pragma once
#include <vector>
#include <cstdint>
#include <bit>
#include "matrix_utils/types.hpp"

namespace uspmm {

    using BitVector = std::vector<uint64_t>;

    template <typename T>
    std::vector<BitVector> computeQuotientRows(const matrix_utils::CSRMatrix<T>& matrix, int block_width) {
        int num_blocks = (matrix.num_cols + block_width - 1) / block_width;
        int num_words = (num_blocks + 63) / 64;

        std::vector<BitVector> quotient_rows(matrix.num_rows, BitVector(num_words, 0ULL));

        for (int i = 0; i < matrix.num_rows; ++i) {
            for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i+1]; ++j) {
                int col = matrix.col_ind[j];
                int block_idx = col / block_width;
                int word_idx = block_idx / 64;
                int bit_idx = block_idx % 64;
                quotient_rows[i][word_idx] |= (1ULL << bit_idx);
            }
        }
        return quotient_rows;
    }

    inline float computeJaccardDistance(const BitVector& v, const BitVector& w) {
        int intersection_pop = 0;
        int union_pop = 0;

        for (size_t i = 0; i < v.size(); ++i) {
            intersection_pop += std::popcount(v[i] & w[i]);
            union_pop += std::popcount(v[i] | w[i]);
        }

        if (union_pop == 0) return 0.0f;
        return 1.0f - (static_cast<float>(intersection_pop) / union_pop);
    }

    template <typename T>
    std::vector<int> computeIterativeClustering(const matrix_utils::CSRMatrix<T>& matrix, float dist_thresh, int block_width) {
        std::vector<BitVector> V_quotients = computeQuotientRows(matrix, block_width);

        std::vector<int> unclustered;
        unclustered.reserve(matrix.num_rows);
        for (int i = 0; i < matrix.num_rows; ++i) {
            unclustered.push_back(i);
        }

        std::vector<std::vector<int>> clusters;

        while (!unclustered.empty()) {
            int seed_idx = unclustered.back();
            unclustered.pop_back();

            std::vector<int> c = {seed_idx};
            BitVector p_c = V_quotients[seed_idx];

            for (int i = static_cast<int>(unclustered.size()) - 1; i >= 0; --i) {
                int candidate_idx = unclustered[i];
                float dist = computeJaccardDistance(p_c, V_quotients[candidate_idx]);

                if (dist <= dist_thresh) {
                    c.push_back(candidate_idx);
                    for (size_t word = 0; word < p_c.size(); ++word) {
                        p_c[word] |= V_quotients[candidate_idx][word];
                    }
                    unclustered[i] = unclustered.back();
                    unclustered.pop_back();
                }
            }
            clusters.push_back(c);
        }

        std::vector<int> final_permutation;
        final_permutation.reserve(matrix.num_rows);
        for (const auto& cluster : clusters) {
            for (int row_idx : cluster) {
                final_permutation.push_back(row_idx);
            }
        }
        return final_permutation;
    }

    // -----------------------------------------------------------------
    // Apply a row permutation array to a CSR Matrix
    // -----------------------------------------------------------------
    template <typename T>
    matrix_utils::CSRMatrix<T> apply_permutation(const matrix_utils::CSRMatrix<T>& mat, const std::vector<int>& perm) {
        if (perm.size() != static_cast<size_t>(mat.num_rows)) {
            throw std::invalid_argument("Permutation array size must match the number of matrix rows.");
        }

        matrix_utils::CSRMatrix<T> permuted_mat;
        permuted_mat.num_rows = mat.num_rows;
        permuted_mat.num_cols = mat.num_cols;

        // Pre-allocate memory to avoid reallocations
        permuted_mat.row_ptr.reserve(mat.num_rows + 1);
        permuted_mat.col_ind.reserve(mat.col_ind.size());
        permuted_mat.values.reserve(mat.values.size());

        permuted_mat.row_ptr.push_back(0);

        for (int i = 0; i < mat.num_rows; ++i) {
            int old_row = perm[i];
            int start = mat.row_ptr[old_row];
            int end   = mat.row_ptr[old_row + 1];

            // Copy the column indices and values for this row
            for (int j = start; j < end; ++j) {
                permuted_mat.col_ind.push_back(mat.col_ind[j]);
                permuted_mat.values.push_back(mat.values[j]);
            }

            // Record where this row ends in the new matrix
            permuted_mat.row_ptr.push_back(permuted_mat.col_ind.size());
        }

        return permuted_mat;
    }

} // namespace uspmm