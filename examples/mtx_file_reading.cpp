#include <iostream>
#include <string>
#include <filesystem>
#include <iomanip>
#include "matrix_utils/io.hpp"

namespace fs = std::filesystem;

int main() {
    // Target directory (relative to the build folder)
    std::string base_path = "../test_data_mtx/";

    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        std::cerr << "Error: Directory '" << base_path << "' does not exist or is not a folder.\n";
        return 1;
    }

    std::cout << "Scanning directory: " << base_path << "\n\n";

    for (const auto& entry : fs::directory_iterator(base_path)) {

        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();

            try {
                std::cout << "=================================================\n";
                std::cout << ">>> Loading: " << filename << "\n";

                auto mat = matrix_utils::io::read_mtx<float>(filepath);

                long long total_elements = static_cast<long long>(mat.num_rows) * mat.num_cols;
                long long nnz = mat.values.size();
                double sparsity = 100.0 * (1.0 - (static_cast<double>(nnz) / total_elements));

                std::cout << "    Dimensions : " << mat.num_rows << " x " << mat.num_cols << "\n"
                          << "    NNZ        : " << nnz << "\n"
                          << "    Sparsity   : " << std::fixed << std::setprecision(4) << sparsity << "%\n";

                matrix_utils::io::print_csr_submatrix(mat, 0, 10, 0, 10);

            } catch (const std::exception& e) {
                std::cerr << "Error reading " << filename << ": " << e.what() << "\n\n";
            }
        }
    }

    return 0;
}