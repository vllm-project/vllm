#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>

template<typename T>
class Tensor {
private:
    std::unique_ptr<T[]> data_;
    std::vector<size_t> shape_;
    size_t size_;

    size_t computeSize() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
    }

public:
    Tensor(std::vector<size_t> shape) : shape_(shape), size_(computeSize()) {
        data_ = std::make_unique<T[]>(size_);
        std::fill(data_.get(), data_.get() + size_, T(0));
    }

    Tensor(std::vector<size_t> shape, T value) : shape_(shape), size_(computeSize()) {
        data_ = std::make_unique<T[]>(size_);
        std::fill(data_.get(), data_.get() + size_, value);
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    T& at(const std::vector<size_t>& indices) {
        size_t idx = 0;
        size_t multiplier = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            idx += indices[i] * multiplier;
            multiplier *= shape_[i];
        }
        return data_[idx];
    }

    const T& at(const std::vector<size_t>& indices) const {
        return const_cast<Tensor*>(this)->at(indices);
    }

    void fill(T value) {
        std::fill(data_.get(), data_.get() + size_, value);
    }

    Tensor<T> add(const Tensor<T>& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Shape mismatch");
        Tensor<T> result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor<T> mul(const Tensor<T>& other) const {
        if (shape_ != other.shape_) throw std::invalid_argument("Shape mismatch");
        Tensor<T> result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    void print() const {
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], data=[";
        for (size_t i = 0; i < std::min(size_, 10UL); ++i) {
            std::cout << data_[i];
            if (i < std::min(size_, 10UL) - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << "...";
        std::cout << "])" << std::endl;
    }

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    T* data() { return data_.get(); }
};

int main() {
    std::cout << "=== Lab 08: Tensor Class (Solution) ===" << std::endl;

    // Test 1: Creation and fill
    {
        std::cout << "\nTest 1: Creation and Fill" << std::endl;
        Tensor<float> t({2, 3});
        t.fill(1.5f);
        t.print();
    }

    // Test 2: Element access
    {
        std::cout << "\nTest 2: Element Access" << std::endl;
        Tensor<int> t({2, 2});
        t.at({0, 0}) = 1;
        t.at({0, 1}) = 2;
        t.at({1, 0}) = 3;
        t.at({1, 1}) = 4;
        t.print();
    }

    // Test 3: Addition
    {
        std::cout << "\nTest 3: Element-wise Addition" << std::endl;
        Tensor<float> a({3}, 1.0f);
        Tensor<float> b({3}, 2.0f);
        auto c = a.add(b);
        std::cout << "a: "; a.print();
        std::cout << "b: "; b.print();
        std::cout << "c = a + b: "; c.print();
    }

    // Test 4: Multiplication
    {
        std::cout << "\nTest 4: Element-wise Multiplication" << std::endl;
        Tensor<float> a({3}, 2.0f);
        Tensor<float> b({3}, 3.0f);
        auto c = a.mul(b);
        std::cout << "c = a * b: "; c.print();
    }

    // Test 5: Move semantics
    {
        std::cout << "\nTest 5: Move Semantics" << std::endl;
        Tensor<float> t1({100, 100});
        t1.fill(42.0f);
        Tensor<float> t2 = std::move(t1);
        std::cout << "Moved tensor: "; t2.print();
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
