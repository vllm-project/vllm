#include <iostream>
#include <memory>
#include <chrono>
#include <cstring>
#include <utility>  // for std::move, std::forward

// Represents a 2D matrix (similar to a tensor in ML frameworks)
class Matrix {
private:
    float* data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructor
    explicit Matrix(size_t rows = 0, size_t cols = 0) : rows_(rows), cols_(cols) {
        if (rows > 0 && cols > 0) {
            data_ = new float[rows * cols];
            std::cout << "Creating Matrix " << rows << "x" << cols
                      << " (" << rows * cols << " elements)" << std::endl;
        } else {
            data_ = nullptr;
        }
    }

    // Destructor
    ~Matrix() {
        if (data_ != nullptr) {
            std::cout << "Destroying Matrix " << rows_ << "x" << cols_ << std::endl;
            delete[] data_;
        }
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

    // Move constructor - transfers ownership (cheap!)
    // noexcept is important for performance (e.g., vector reallocation)
    Matrix(Matrix&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        std::cout << "Moving Matrix " << rows_ << "x" << cols_ << " (cheap!)" << std::endl;

        // Leave other in a valid but empty state
        // This is crucial - the moved-from object's destructor will still run!
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {  // Guard against self-assignment
            std::cout << "Move-assigning Matrix " << other.rows_ << "x" << other.cols_ << std::endl;

            // Free existing resources
            delete[] data_;

            // Transfer ownership from other
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;

            // Leave other in valid but empty state
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

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

    bool isEmpty() const {
        return data_ == nullptr;
    }
};

// Returns a Matrix by value
// The compiler will use RVO (Return Value Optimization) or move semantics
// NO std::move needed here - it would actually prevent RVO!
Matrix createMatrix(size_t rows, size_t cols, float fillValue) {
    Matrix m(rows, cols);
    m.fill(fillValue);
    return m;  // RVO or move - DON'T use std::move here!
}

// Function that accepts only rvalues (temporaries)
void processTemporary(Matrix&& matrix) {
    std::cout << "Processing temporary matrix " << matrix.rows() << "x" << matrix.cols() << std::endl;
    // We can move from matrix since it's an rvalue reference
    Matrix owned = std::move(matrix);
    std::cout << "Took ownership of matrix" << std::endl;
}

// Overload that accepts lvalues
void processTemporary(const Matrix& matrix) {
    std::cout << "Processing lvalue matrix (copy needed) " << matrix.rows() << "x" << matrix.cols() << std::endl;
}

// Perfect forwarding - preserves value category (lvalue vs rvalue)
template<typename T>
void forwardToProcess(T&& matrix) {
    // std::forward preserves whether matrix is lvalue or rvalue
    // If T is Matrix&, forwards as lvalue
    // If T is Matrix, forwards as rvalue
    std::cout << "Forwarding matrix..." << std::endl;
    processTemporary(std::forward<T>(matrix));
}

// Example of a move-only type (deleted copy operations)
class MoveOnlyResource {
private:
    std::unique_ptr<int[]> data_;
    size_t size_;

public:
    explicit MoveOnlyResource(size_t size) : size_(size) {
        data_ = std::make_unique<int[]>(size);
        std::cout << "Created MoveOnlyResource of size " << size << std::endl;
    }

    // Delete copy operations - this class can only be moved
    MoveOnlyResource(const MoveOnlyResource&) = delete;
    MoveOnlyResource& operator=(const MoveOnlyResource&) = delete;

    // Default move operations
    MoveOnlyResource(MoveOnlyResource&&) = default;
    MoveOnlyResource& operator=(MoveOnlyResource&&) = default;

    size_t size() const { return size_; }
};

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

    // Test move
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

    if (move_duration.count() > 0) {
        std::cout << "Speedup: " << (double)copy_duration.count() / move_duration.count() << "x faster!" << std::endl;
    }
}

int main() {
    std::cout << "=== Lab 02: Move Semantics (Solution) ===" << std::endl;

    // Test 1: Copy semantics (expensive)
    {
        std::cout << "\nTest 1: Copy Semantics" << std::endl;
        Matrix m1(100, 100);
        m1.fill(1.0f);

        Matrix m2 = m1;  // Copy constructor - expensive but necessary
        m2.at(0, 0) = 2.0f;

        // m1 and m2 are independent
        std::cout << "m1(0,0) = " << m1.at(0, 0) << " (unchanged)" << std::endl;
        std::cout << "m2(0,0) = " << m2.at(0, 0) << " (modified)" << std::endl;
    }

    // Test 2: Move semantics
    {
        std::cout << "\nTest 2: Move Semantics" << std::endl;
        Matrix m1(100, 100);
        m1.fill(3.0f);

        Matrix m2 = std::move(m1);  // Move constructor - cheap!
        // m1 is now in moved-from state (empty but valid)

        std::cout << "m2(0,0) = " << m2.at(0, 0) << std::endl;
        std::cout << "m1 is empty: " << (m1.isEmpty() ? "yes" : "no") << std::endl;
    }

    // Test 3: Return value optimization
    {
        std::cout << "\nTest 3: Return Value Optimization" << std::endl;
        Matrix m = createMatrix(50, 50, 5.0f);
        // Compiler uses RVO - likely zero copies!
        std::cout << "m(0,0) = " << m.at(0, 0) << std::endl;
    }

    // Test 4: Move assignment
    {
        std::cout << "\nTest 4: Move Assignment" << std::endl;
        Matrix m1(100, 100);
        m1.fill(7.0f);

        Matrix m2(50, 50);
        m2 = std::move(m1);  // Move assignment

        std::cout << "m2 now has size " << m2.rows() << "x" << m2.cols() << std::endl;
        std::cout << "m1 is empty: " << (m1.isEmpty() ? "yes" : "no") << std::endl;
    }

    // Test 5: Processing rvalues vs lvalues
    {
        std::cout << "\nTest 5: Rvalue vs Lvalue Processing" << std::endl;

        Matrix m1(10, 10);
        processTemporary(m1);  // Lvalue - calls const& overload

        processTemporary(Matrix(10, 10));  // Rvalue - calls && overload
    }

    // Test 6: Perfect forwarding
    {
        std::cout << "\nTest 6: Perfect Forwarding" << std::endl;

        Matrix m1(10, 10);
        forwardToProcess(m1);  // Forwards as lvalue

        forwardToProcess(Matrix(10, 10));  // Forwards as rvalue
    }

    // Test 7: Move-only type
    {
        std::cout << "\nTest 7: Move-Only Type" << std::endl;
        MoveOnlyResource r1(100);

        // This would NOT compile (copy deleted):
        // MoveOnlyResource r2 = r1;

        // But move works:
        MoveOnlyResource r2 = std::move(r1);
        std::cout << "Moved resource, size: " << r2.size() << std::endl;
    }

    // Benchmark
    benchmarkCopyVsMove();

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
