#include <iostream>
#include <memory>
#include <chrono>
#include <cstring>

// Represents a 2D matrix (similar to a tensor in ML frameworks)
class Matrix {
private:
    float* data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructor
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_ = new float[rows * cols];
        std::cout << "Creating Matrix " << rows << "x" << cols
                  << " (" << rows * cols << " elements)" << std::endl;
    }

    // TODO: Implement destructor
    ~Matrix() {
        // TODO: Clean up data_
        std::cout << "Destroying Matrix " << rows_ << "x" << cols_ << std::endl;
    }

    // Copy constructor - performs deep copy (expensive!)
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_) {
        std::cout << "Copying Matrix " << rows_ << "x" << cols_ << " (expensive!)" << std::endl;
        data_ = new float[rows_ * cols_];
        std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(float));
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            std::cout << "Copy-assigning Matrix " << other.rows_ << "x" << other.cols_ << std::endl;
            delete[] data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = new float[rows_ * cols_];
            std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(float));
        }
        return *this;
    }

    // TODO: Implement move constructor
    // Should transfer ownership of data_ from other to this
    // and leave other in a valid but empty state
    // Matrix(Matrix&& other) noexcept {
    //     // TODO: Implement efficient move
    // }

    // TODO: Implement move assignment operator
    // Should transfer ownership and handle self-assignment
    // Matrix& operator=(Matrix&& other) noexcept {
    //     // TODO: Implement efficient move assignment
    // }

    // Get element at (row, col)
    float& at(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& at(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    void fill(float value) {
        for (size_t i = 0; i < rows_ * cols_; ++i) {
            data_[i] = value;
        }
    }
};

// TODO: Implement a function that returns a Matrix by value
// This should trigger move semantics (RVO or NRVO)
Matrix createMatrix(size_t rows, size_t cols, float fillValue) {
    // TODO: Create a matrix, fill it, and return it
    // Question: Should you use std::move here?
    Matrix m(rows, cols);
    m.fill(fillValue);
    return m;  // TODO: Is std::move needed here?
}

// TODO: Implement a function that takes an rvalue reference
// This explicitly accepts only rvalues (temporaries)
// void processTemporary(Matrix&& matrix) {
//     // TODO: What can we do with matrix here?
//     std::cout << "Processing temporary matrix" << std::endl;
// }

// TODO: Implement a function with perfect forwarding
// template<typename T>
// void forwardToProcess(T&& matrix) {
//     // TODO: Use std::forward to preserve value category
//     processTemporary(std::forward<T>(matrix));
// }

// Benchmark copy vs move
void benchmarkCopyVsMove() {
    std::cout << "\n=== Benchmark: Copy vs Move ===" << std::endl;

    const size_t SIZE = 1000;

    // Test copy
    auto start = std::chrono::high_resolution_clock::now();
    {
        Matrix m1(SIZE, SIZE);
        m1.fill(1.0f);
        Matrix m2 = m1;  // Copy constructor
        Matrix m3(SIZE, SIZE);
        m3 = m1;  // Copy assignment
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Copy operations took: " << copy_duration.count() << " μs" << std::endl;

    // TODO: Test move
    // Uncomment when move constructor/assignment are implemented
    /*
    start = std::chrono::high_resolution_clock::now();
    {
        Matrix m1(SIZE, SIZE);
        m1.fill(1.0f);
        Matrix m2 = std::move(m1);  // Move constructor
        Matrix m3(SIZE, SIZE);
        m3 = std::move(m2);  // Move assignment
    }
    end = std::chrono::high_resolution_clock::now();
    auto move_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Move operations took: " << move_duration.count() << " μs" << std::endl;
    std::cout << "Speedup: " << (double)copy_duration.count() / move_duration.count() << "x" << std::endl;
    */
}

int main() {
    std::cout << "=== Lab 02: Move Semantics (Starter) ===" << std::endl;

    // Test 1: Copy semantics (expensive)
    {
        std::cout << "\nTest 1: Copy Semantics" << std::endl;
        Matrix m1(100, 100);
        m1.fill(1.0f);

        Matrix m2 = m1;  // Copy constructor - expensive!
        m2.at(0, 0) = 2.0f;

        std::cout << "m1(0,0) = " << m1.at(0, 0) << std::endl;
        std::cout << "m2(0,0) = " << m2.at(0, 0) << std::endl;
    }

    // Test 2: TODO - Move semantics
    // Uncomment when move constructor is implemented
    /*
    {
        std::cout << "\nTest 2: Move Semantics" << std::endl;
        Matrix m1(100, 100);
        m1.fill(3.0f);

        Matrix m2 = std::move(m1);  // Move constructor - cheap!
        // m1 is now in moved-from state - don't use it!

        std::cout << "m2(0,0) = " << m2.at(0, 0) << std::endl;
    }
    */

    // Test 3: TODO - Return value optimization
    /*
    {
        std::cout << "\nTest 3: Return Value Optimization" << std::endl;
        Matrix m = createMatrix(50, 50, 5.0f);
        std::cout << "m(0,0) = " << m.at(0, 0) << std::endl;
    }
    */

    // Benchmark
    benchmarkCopyVsMove();

    std::cout << "\n=== End of tests ===" << std::endl;
    return 0;
}
