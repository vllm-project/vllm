#include <iostream>
#include <cassert>
#include <memory>
#include <cstring>
#include <utility>
#include <type_traits>

// Simple test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "..."; \
    test_##name(); \
    std::cout << " PASSED" << std::endl; \
} while(0)

// Copy the Matrix class for testing
class Matrix {
private:
    float* data_;
    size_t rows_;
    size_t cols_;
    static int total_instances_;
    static int total_copies_;
    static int total_moves_;

public:
    explicit Matrix(size_t rows = 0, size_t cols = 0) : rows_(rows), cols_(cols) {
        if (rows > 0 && cols > 0) {
            data_ = new float[rows * cols]();
            total_instances_++;
        } else {
            data_ = nullptr;
        }
    }

    ~Matrix() {
        if (data_ != nullptr) {
            delete[] data_;
            total_instances_--;
        }
    }

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_) {
        data_ = new float[rows_ * cols_];
        std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(float));
        total_instances_++;
        total_copies_++;
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = new float[rows_ * cols_];
            std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(float));
            total_copies_++;
        }
        return *this;
    }

    Matrix(Matrix&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        total_moves_++;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            total_moves_++;
        }
        return *this;
    }

    float& at(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& at(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    bool isEmpty() const { return data_ == nullptr; }

    void fill(float value) {
        for (size_t i = 0; i < rows_ * cols_; ++i) {
            data_[i] = value;
        }
    }

    static int instances() { return total_instances_; }
    static int copies() { return total_copies_; }
    static int moves() { return total_moves_; }
    static void resetCounters() {
        total_copies_ = 0;
        total_moves_ = 0;
    }
};

int Matrix::total_instances_ = 0;
int Matrix::total_copies_ = 0;
int Matrix::total_moves_ = 0;

// Helper function for testing
Matrix createMatrix(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    return m;
}

// Tests

TEST(move_constructor_basic) {
    Matrix::resetCounters();

    Matrix m1(10, 10);
    m1.fill(5.0f);

    Matrix m2(std::move(m1));

    // m2 should have the data
    assert(m2.rows() == 10);
    assert(m2.cols() == 10);
    assert(m2.at(0, 0) == 5.0f);

    // m1 should be empty
    assert(m1.isEmpty());
    assert(m1.rows() == 0);
    assert(m1.cols() == 0);

    // Should have been a move, not a copy
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() == 1);
}

TEST(move_assignment_basic) {
    Matrix::resetCounters();

    Matrix m1(10, 10);
    m1.fill(7.0f);

    Matrix m2(5, 5);
    m2 = std::move(m1);

    // m2 should have m1's data
    assert(m2.rows() == 10);
    assert(m2.cols() == 10);
    assert(m2.at(0, 0) == 7.0f);

    // m1 should be empty
    assert(m1.isEmpty());

    // Should have been a move, not a copy
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() == 1);
}

TEST(move_self_assignment) {
    Matrix m(10, 10);
    m.fill(3.0f);

    // Self-move-assignment should be handled gracefully
    m = std::move(m);

    // Matrix should still be valid (though state is unspecified)
    // At minimum, it shouldn't crash
    assert(true);
}

TEST(copy_constructor_creates_independent_copy) {
    Matrix m1(10, 10);
    m1.fill(1.0f);

    Matrix m2(m1);  // Copy

    // Both should have same size
    assert(m1.rows() == 10);
    assert(m2.rows() == 10);

    // Modify m2
    m2.at(0, 0) = 99.0f;

    // m1 should be unchanged
    assert(m1.at(0, 0) == 1.0f);
    assert(m2.at(0, 0) == 99.0f);
}

TEST(copy_assignment_creates_independent_copy) {
    Matrix m1(10, 10);
    m1.fill(1.0f);

    Matrix m2(5, 5);
    m2 = m1;  // Copy assignment

    assert(m2.rows() == 10);
    assert(m2.cols() == 10);

    m2.at(0, 0) = 99.0f;
    assert(m1.at(0, 0) == 1.0f);
}

TEST(return_value_uses_move_or_rvo) {
    Matrix::resetCounters();

    Matrix m = createMatrix(10, 10);

    assert(m.rows() == 10);
    assert(m.cols() == 10);

    // Should have 0 or 1 move (depending on RVO)
    // Should definitely not copy
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() <= 1);
}

TEST(move_leaves_valid_state) {
    Matrix m1(10, 10);
    m1.fill(5.0f);

    Matrix m2(std::move(m1));

    // m1 should be in valid state (can be destroyed, assigned to)
    // It should be empty but valid
    assert(m1.isEmpty());

    // We should be able to assign to m1
    m1 = Matrix(5, 5);
    assert(m1.rows() == 5);
    assert(m1.cols() == 5);
    assert(!m1.isEmpty());
}

TEST(move_chain) {
    Matrix::resetCounters();

    Matrix m1(10, 10);
    Matrix m2(std::move(m1));
    Matrix m3(std::move(m2));
    Matrix m4(std::move(m3));

    assert(m4.rows() == 10);
    assert(m1.isEmpty());
    assert(m2.isEmpty());
    assert(m3.isEmpty());

    // Should be 3 moves, no copies
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() == 3);
}

TEST(vector_of_matrices_uses_move) {
    Matrix::resetCounters();

    std::vector<Matrix> vec;
    vec.reserve(3);  // Prevent reallocation

    vec.push_back(Matrix(10, 10));
    vec.push_back(Matrix(20, 20));
    vec.push_back(Matrix(30, 30));

    assert(vec.size() == 3);
    assert(vec[0].rows() == 10);
    assert(vec[1].rows() == 20);
    assert(vec[2].rows() == 30);

    // Should use move, not copy (when adding rvalue temps)
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() == 3);
}

TEST(type_traits_check) {
    // Verify that Matrix is move constructible and assignable
    assert(std::is_move_constructible<Matrix>::value);
    assert(std::is_move_assignable<Matrix>::value);

    // Should also be copy constructible and assignable
    assert(std::is_copy_constructible<Matrix>::value);
    assert(std::is_copy_assignable<Matrix>::value);

    // Move constructor should be noexcept
    assert(std::is_nothrow_move_constructible<Matrix>::value);
    assert(std::is_nothrow_move_assignable<Matrix>::value);
}

TEST(empty_matrix_move) {
    Matrix m1;  // Empty matrix
    assert(m1.isEmpty());

    Matrix m2(std::move(m1));
    assert(m2.isEmpty());
    assert(m1.isEmpty());

    // Should not crash
}

TEST(resource_cleanup) {
    int initial = Matrix::instances();

    {
        Matrix m1(10, 10);
        assert(Matrix::instances() == initial + 1);

        Matrix m2(std::move(m1));
        assert(Matrix::instances() == initial + 1);  // Still just one allocation
    }

    assert(Matrix::instances() == initial);  // All cleaned up
}

TEST(large_matrix_move_efficiency) {
    Matrix::resetCounters();

    const size_t LARGE = 1000;
    Matrix m1(LARGE, LARGE);
    m1.fill(1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix m2(std::move(m1));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Move should be very fast (< 1 ms for sure)
    assert(duration.count() < 1000000);  // Less than 1ms

    // Should be move, not copy
    assert(Matrix::copies() == 0);
    assert(Matrix::moves() == 1);
}

int main() {
    std::cout << "=== Running Move Semantics Tests ===" << std::endl;

    RUN_TEST(move_constructor_basic);
    RUN_TEST(move_assignment_basic);
    RUN_TEST(move_self_assignment);
    RUN_TEST(copy_constructor_creates_independent_copy);
    RUN_TEST(copy_assignment_creates_independent_copy);
    RUN_TEST(return_value_uses_move_or_rvo);
    RUN_TEST(move_leaves_valid_state);
    RUN_TEST(move_chain);
    RUN_TEST(vector_of_matrices_uses_move);
    RUN_TEST(type_traits_check);
    RUN_TEST(empty_matrix_move);
    RUN_TEST(resource_cleanup);
    RUN_TEST(large_matrix_move_efficiency);

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
