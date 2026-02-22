#include <iostream>
#include <vector>
#include "matrix_utils/generation.hpp"
#include "matrix_utils/io.hpp"
#include "uspmm/clustering.hpp"

int main() {
    std::cout << "1. Generating 16x16 Checkers Matrix (2x2 blocks)...\n";
    auto mat = matrix_utils::generation::generate_checkers<float>(16, 16, 2);

    std::cout << "\n2. Original Matrix:\n";
    matrix_utils::io::print_csr(mat);

    std::cout << "3. Running Iterative Clustering (dist_thresh = 0.0, block_width = 2)...\n";
    // A distance threshold of 0.0 perfectly groups identical rows.
    auto perm = uspmm::computeIterativeClustering(mat, 0.0f, 2);

    std::cout << "   Resulting Permutation Vector: [ ";
    for (int p : perm) {
        std::cout << p << " ";
    }
    std::cout << "]\n";

    std::cout << "\n4. Applying Permutation to Original Matrix...\n";
    auto reordered_mat = uspmm::apply_permutation(mat, perm);

    std::cout << "\n5. Rearranged Matrix (Notice the perfectly grouped dense blocks!):\n";
    matrix_utils::io::print_csr(reordered_mat);

    return 0;
}