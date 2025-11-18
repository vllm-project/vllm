#include <iostream>
#include <cassert>
#include <vector>
#include <type_traits>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "..."; \
    test_##name(); \
    std::cout << " PASSED" << std::endl; \
} while(0)

// Template class for testing
template<typename T>
class Tensor {
private:
    std::vector<T> data_;
public:
    explicit Tensor(size_t size) : data_(size) {}
    Tensor(std::initializer_list<T> init) : data_(init) {}
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    size_t size() const { return data_.size(); }

    T sum() const {
        T result = T(0);
        for (const auto& val : data_) {
            result += val;
        }
        return result;
    }
};

template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
T dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
    T result = T(0);
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Tests

TEST(template_function_int) {
    assert(max(5, 10) == 10);
    assert(max(100, 50) == 100);
}

TEST(template_function_double) {
    assert(max(3.14, 2.71) == 3.14);
    assert(max(1.5, 2.5) == 2.5);
}

TEST(template_function_char) {
    assert(max('a', 'z') == 'z');
    assert(max('m', 'b') == 'm');
}

TEST(tensor_int) {
    Tensor<int> t({1, 2, 3, 4, 5});
    assert(t.size() == 5);
    assert(t[0] == 1);
    assert(t[4] == 5);
    assert(t.sum() == 15);
}

TEST(tensor_float) {
    Tensor<float> t(3);
    t[0] = 1.5f;
    t[1] = 2.5f;
    t[2] = 3.0f;
    assert(t.size() == 3);
    assert(t.sum() == 7.0f);
}

TEST(tensor_double) {
    Tensor<double> t({1.1, 2.2, 3.3});
    assert(t.size() == 3);
    assert(t[1] == 2.2);
}

TEST(dot_product_int) {
    Tensor<int> a({1, 2, 3});
    Tensor<int> b({4, 5, 6});
    assert(dotProduct(a, b) == 32);  // 1*4 + 2*5 + 3*6 = 32
}

TEST(dot_product_float) {
    Tensor<float> a({1.0f, 2.0f});
    Tensor<float> b({3.0f, 4.0f});
    assert(dotProduct(a, b) == 11.0f);  // 1*3 + 2*4 = 11
}

TEST(type_traits) {
    assert((std::is_same<decltype(max(1, 2)), int>::value));
    assert((std::is_same<decltype(max(1.0, 2.0)), double>::value));
}

TEST(tensor_empty) {
    Tensor<int> t(0);
    assert(t.size() == 0);
    assert(t.sum() == 0);
}

int main() {
    std::cout << "=== Running Templates Tests ===" << std::endl;

    RUN_TEST(template_function_int);
    RUN_TEST(template_function_double);
    RUN_TEST(template_function_char);
    RUN_TEST(tensor_int);
    RUN_TEST(tensor_float);
    RUN_TEST(tensor_double);
    RUN_TEST(dot_product_int);
    RUN_TEST(dot_product_float);
    RUN_TEST(type_traits);
    RUN_TEST(tensor_empty);

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
