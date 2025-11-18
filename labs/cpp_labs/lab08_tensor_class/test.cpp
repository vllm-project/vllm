#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <numeric>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

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

    void fill(T value) { std::fill(data_.get(), data_.get() + size_, value); }
    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }
    T* data() { return data_.get(); }
};

TEST(tensor_creation) {
    Tensor<float> t({2, 3});
    assert(t.size() == 6);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
}

TEST(tensor_fill) {
    Tensor<int> t({3, 3});
    t.fill(5);
    for (size_t i = 0; i < t.size(); ++i) {
        assert(t.data()[i] == 5);
    }
}

TEST(tensor_move) {
    Tensor<float> t1({10, 10});
    t1.fill(1.0f);
    Tensor<float> t2 = std::move(t1);
    assert(t2.size() == 100);
}

TEST(tensor_different_dtypes) {
    Tensor<int> ti({2, 2});
    Tensor<float> tf({2, 2});
    Tensor<double> td({2, 2});
    assert(ti.size() == 4);
    assert(tf.size() == 4);
    assert(td.size() == 4);
}

int main() {
    std::cout << "=== Running Tensor Class Tests ===" << std::endl;
    RUN_TEST(tensor_creation);
    RUN_TEST(tensor_fill);
    RUN_TEST(tensor_move);
    RUN_TEST(tensor_different_dtypes);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
