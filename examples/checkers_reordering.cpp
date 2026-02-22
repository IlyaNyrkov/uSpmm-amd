#include <iostream>
#include <vector>
#include "matrix_utils/generation.hpp"
#include "matrix_utils/io.hpp"
#include "matrix_utils/conversion.hpp"
#include "uspmm/clustering.hpp"

int main() {
    std::cout << "1. Generating 16x16 Checkers Matrix (2x2 checkers pattern)...\n";
    auto mat = matrix_utils::generation::generate_checkers<float>(16, 16, 2);

    std::cout << "\n2. Original Matrix:\n";
    matrix_utils::io::print_csr(mat);

    std::cout << "3. Running Iterative Clustering (dist_thresh = 0.0, block_width = 2)...\n";
    auto perm = uspmm::computeIterativeClustering(mat, 0.0f, 2);

    std::cout << "   Resulting Permutation Vector: [ ";
    for (int p : perm) {
        std::cout << p << " ";
    }
    std::cout << "]\n";

    std::cout << "\n4. Applying Permutation to Original Matrix...\n";
    auto reordered_mat = uspmm::apply_permutation(mat, perm);

    std::cout << "\n5. Rearranged Matrix:\n";
    matrix_utils::io::print_csr(reordered_mat);

    std::cout << "6. Converting to BCSR format (using 4x4 hardware macro-blocks)...\n";
    int r_block = 2;
    int c_block = 2;

    auto bcsr_orig = matrix_utils::format::csr_to_bcsr(mat, r_block, c_block);
    std::cout << "BCSR Original Matrix (RAW):" << std::endl;
    matrix_utils::io::printBCSRRaw(bcsr_orig);

    auto bcsr_reordered = matrix_utils::format::csr_to_bcsr(reordered_mat, r_block * 4, c_block);
    std::cout << "BCSR Reordered Matrix:" << std::endl;
    matrix_utils::io::printBCSRRaw(bcsr_reordered);

    // Calculate the amount of dense blocks allocated
    int orig_blocks = bcsr_orig.bcsr_values.size() / (r_block * c_block);
    int reordered_blocks = bcsr_reordered.bcsr_values.size() / (r_block * 4 * c_block);

    std::cout << "\n7. Hardware Tile Allocation Results:\n";
    std::cout << "   Original BCSR blocks allocated   : " << orig_blocks << "\n";
    std::cout << "   Rearranged BCSR blocks allocated : " << reordered_blocks << "\n";

    int saved_blocks = orig_blocks - reordered_blocks;
    std::cout << "   -> Iterative clustering eliminated " << saved_blocks
              << " padded memory blocks!\n\n";

    return 0;
}