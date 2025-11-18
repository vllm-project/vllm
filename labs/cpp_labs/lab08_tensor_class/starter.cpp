#include <iostream>
#include <vector>
#include <memory>

// TODO: Implement a complete Tensor class
// Features needed:
// - Template for dtype
// - Shape management
// - Smart pointer for data
// - Move semantics
// - Element-wise operations
// - Matrix multiplication

template<typename T>
class Tensor {
private:
    std::unique_ptr<T[]> data_;
    std::vector<size_t> shape_;
    size_t size_;

public:
    // TODO: Implement constructor
    Tensor(std::vector<size_t> shape) : shape_(shape) {
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }
        data_ = std::make_unique<T[]>(size_);
    }

    // TODO: Implement element access
    // TODO: Implement operations (add, mul, matmul)
    // TODO: Implement reshape
    // TODO: Implement print
};

int main() {
    std::cout << "=== Lab 08: Tensor Class (Starter) ===" << std::endl;

    // TODO: Create and test tensor
    // Tensor<float> t({2, 3});
    // t.fill(1.0f);
    // t.print();

    return 0;
}
