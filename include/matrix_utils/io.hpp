#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>
#include "matrix_utils/types.hpp"

namespace matrix_utils {
namespace io {

    template <typename T>
    void printBCSRRaw(BCSRMatrix<T>& mat) {
        std::cout << "row ptrs: [";
        for (const auto& row : mat.bcsr_row_ptr) {
            std::cout << " " << row;
        }
        std::cout << "]" << std::endl;

        std::cout << "col indices: [";
        for (const auto& col : mat.bcsr_col_ind) {
            std::cout << " " << col;
        }
        std::cout << "]" << std::endl;

        std::cout << "vals : [" << std::endl;
        for (const auto& val : mat.bcsr_values) {
            std::cout << " " << val;
        }
        std::cout << "]" << std::endl;
    }

    // -----------------------------------------------------------------
    // Helper to print a single value safely (handles int8_t chars and __half)
    // -----------------------------------------------------------------
    template <typename T>
    inline void print_val(const T& val, int width = 6) {
        // Casting to float ensures __half and int8_t print as actual numbers
        std::cout << std::setw(width) << std::setprecision(3) << static_cast<float>(val) << " ";
    }

    // -----------------------------------------------------------------
    // 1. Print full CSR Matrix (Standard Algebraic Notation)
    // -----------------------------------------------------------------
    template <typename T>
    void print_csr(const CSRMatrix<T>& mat) {
        std::cout << "--- CSR Matrix (" << mat.num_rows << "x" << mat.num_cols << ") ---" << std::endl;
        for (int i = 0; i < mat.num_rows; ++i) {
            int row_start = mat.row_ptr[i];
            int row_end   = mat.row_ptr[i + 1];
            int current_idx = row_start;

            for (int j = 0; j < mat.num_cols; ++j) {
                if (current_idx < row_end && mat.col_ind[current_idx] == j) {
                    print_val(mat.values[current_idx]);
                    current_idx++;
                } else {
                    print_val(0.0f);
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // 2. Print Principal Submatrix of a CSR Matrix
    // -----------------------------------------------------------------
    template <typename T>
    void print_csr_submatrix(const CSRMatrix<T>& mat, int start_row, int end_row, int start_col, int end_col) {
        // Bound checks
        start_row = std::max(0, start_row);
        end_row   = std::min(mat.num_rows, end_row);
        start_col = std::max(0, start_col);
        end_col   = std::min(mat.num_cols, end_col);

        std::cout << "--- CSR Submatrix Rows[" << start_row << ":" << end_row
                  << "] Cols[" << start_col << ":" << end_col << "] ---" << std::endl;

        for (int i = start_row; i < end_row; ++i) {
            int row_start = mat.row_ptr[i];
            int row_end   = mat.row_ptr[i + 1];
            int current_idx = row_start;

            // Fast-forward to the starting column
            while (current_idx < row_end && mat.col_ind[current_idx] < start_col) {
                current_idx++;
            }

            for (int j = start_col; j < end_col; ++j) {
                if (current_idx < row_end && mat.col_ind[current_idx] == j) {
                    print_val(mat.values[current_idx]);
                    current_idx++;
                } else {
                    print_val(0.0f);
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // 3A. Print 1D Dense Matrix (Assuming Row-Major)
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense(const std::vector<T>& mat, int rows, int cols) {
        std::cout << "--- Dense Matrix (" << rows << "x" << cols << ") ---" << std::endl;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                print_val(mat[i * cols + j]);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // 3B. Print Principal Submatrix of a 1D Dense Matrix
    // -----------------------------------------------------------------
    template <typename T>
    void print_dense_submatrix(const std::vector<T>& mat, int total_cols,
                               int start_row, int end_row, int start_col, int end_col) {
        std::cout << "--- Dense Submatrix Rows[" << start_row << ":" << end_row
                  << "] Cols[" << start_col << ":" << end_col << "] ---" << std::endl;

        for (int i = start_row; i < end_row; ++i) {
            for (int j = start_col; j < end_col; ++j) {
                print_val(mat[i * total_cols + j]);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // 4. Read .mtx (Matrix Market) file into CSR Format
    // -----------------------------------------------------------------
    template <typename T>
    CSRMatrix<T> read_mtx(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open MTX file: " + filename);
        }

        std::string line;
        int rows = 0, cols = 0, nnz = 0;

        // Skip comments and parse dimensions
        while (std::getline(file, line)) {
            // Skip MatrixMarket header/comments
            if (line.empty() || line[0] == '%') continue;

            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nnz)) {
                throw std::runtime_error("Failed to parse MTX dimensions.");
            }
            break;
        }

        // Temporary structure to hold COO (Coordinate) format
        struct Element {
            int r, c;
            T v;
            bool operator<(const Element& other) const {
                if (r != other.r) return r < other.r;
                return c < other.c;
            }
        };

        std::vector<Element> elements;
        elements.reserve(nnz);

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;

            std::istringstream iss(line);
            int r, c;
            float v = 1.0f;

            iss >> r >> c;
            if (!(iss >> v)) { v = 1.0f; }

            elements.push_back({r - 1, c - 1, static_cast<T>(v)});
        }

        // Sort by row, then by column
        std::sort(elements.begin(), elements.end());

        // Convert sorted COO to CSR
        CSRMatrix<T> csr;
        csr.num_rows = rows;
        csr.num_cols = cols;
        csr.row_ptr.assign(rows + 1, 0);
        csr.col_ind.reserve(nnz);
        csr.values.reserve(nnz);

        for (const auto& el : elements) {
            csr.col_ind.push_back(el.c);
            csr.values.push_back(el.v);
            csr.row_ptr[el.r + 1]++;
        }

        // Prefix sum for row_ptr
        for (int i = 0; i < rows; ++i) {
            csr.row_ptr[i + 1] += csr.row_ptr[i];
        }

        return csr;
    }

} // namespace io
} // namespace matrix_utils