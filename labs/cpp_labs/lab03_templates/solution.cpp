#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <stdexcept>

// Simple template function
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Template class for a 1D Tensor
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

    // Generic sum that works for any numeric type
    T sum() const {
        T result = T(0);  // Zero-initialize for the type
        for (const auto& val : data_) {
            result += val;
        }
        return result;
    }

    // Mean (only for floating-point types)
    template<typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, U>::type
    mean() const {
        if (data_.empty()) return U(0);
        return sum() / static_cast<U>(data_.size());
    }

    void fill(T value) {
        for (auto& val : data_) {
            val = value;
        }
    }

    void print() const {
        std::cout << "[";
        for (size_t i = 0; i < data_.size(); ++i) {
            std::cout << data_[i];
            if (i < data_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

// Template function for dot product
template<typename T>
T dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Tensors must have same size");
    }

    T result = T(0);
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Template function for element-wise operation
template<typename T, typename Func>
Tensor<T> transform(const Tensor<T>& input, Func op) {
    Tensor<T> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = op(input[i]);
    }
    return result;
}

// Template function for element-wise binary operation
template<typename T, typename Func>
Tensor<T> combine(const Tensor<T>& a, const Tensor<T>& b, Func op) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Tensors must have same size");
    }

    Tensor<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = op(a[i], b[i]);
    }
    return result;
}

// Generic print function
template<typename T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// Specialized for bool
template<>
void print<bool>(const bool& value) {
    std::cout << (value ? "true" : "false") << std::endl;
}

// Variadic template for printing multiple values
template<typename T>
void printAll(const T& value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void printAll(const T& first, const Args&... rest) {
    std::cout << first << ", ";
    printAll(rest...);
}

// Template with default type
template<typename T = float>
class DefaultTensor {
private:
    std::vector<T> data_;
public:
    explicit DefaultTensor(size_t size) : data_(size) {}
    size_t size() const { return data_.size(); }
};

// Constexpr template function (compile-time computation)
template<int N>
constexpr int factorial() {
    if constexpr (N <= 1) {
        return 1;
    } else {
        return N * factorial<N-1>();
    }
}

// SFINAE example: only enable for numeric types
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
square(T value) {
    return value * value;
}

int main() {
    std::cout << "=== Lab 03: Templates (Solution) ===" << std::endl;

    // Test 1: Template function
    {
        std::cout << "\nTest 1: Template Function (max)" << std::endl;
        auto m1 = max(10, 20);
        auto m2 = max(3.14, 2.71);
        auto m3 = max('a', 'z');

        std::cout << "max(10, 20) = " << m1 << std::endl;
        std::cout << "max(3.14, 2.71) = " << m2 << std::endl;
        std::cout << "max('a', 'z') = " << m3 << std::endl;
    }

    // Test 2: Template class with different types
    {
        std::cout << "\nTest 2: Template Class (Tensor<T>)" << std::endl;

        Tensor<float> tf(5);
        tf.fill(3.14f);
        std::cout << "Float tensor sum: " << tf.sum() << std::endl;
        std::cout << "Float tensor mean: " << tf.mean() << std::endl;

        Tensor<int> ti({1, 2, 3, 4, 5});
        std::cout << "Int tensor: ";
        ti.print();
        std::cout << "Int tensor sum: " << ti.sum() << std::endl;

        Tensor<double> td(3);
        td[0] = 1.1; td[1] = 2.2; td[2] = 3.3;
        std::cout << "Double tensor mean: " << td.mean() << std::endl;
    }

    // Test 3: Dot product
    {
        std::cout << "\nTest 3: Dot Product" << std::endl;
        Tensor<float> a({1.0f, 2.0f, 3.0f});
        Tensor<float> b({4.0f, 5.0f, 6.0f});

        float dot = dotProduct(a, b);
        std::cout << "Dot product: " << dot << std::endl;  // 1*4 + 2*5 + 3*6 = 32
    }

    // Test 4: Transform (element-wise operation)
    {
        std::cout << "\nTest 4: Transform" << std::endl;
        Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f});

        auto squared = transform(t, [](float x) { return x * x; });
        std::cout << "Squared: ";
        squared.print();

        auto doubled = transform(t, [](float x) { return x * 2; });
        std::cout << "Doubled: ";
        doubled.print();
    }

    // Test 5: Combine (element-wise binary operation)
    {
        std::cout << "\nTest 5: Combine" << std::endl;
        Tensor<int> a({1, 2, 3});
        Tensor<int> b({10, 20, 30});

        auto sum = combine(a, b, [](int x, int y) { return x + y; });
        std::cout << "Sum: ";
        sum.print();

        auto product = combine(a, b, [](int x, int y) { return x * y; });
        std::cout << "Product: ";
        product.print();
    }

    // Test 6: Template specialization
    {
        std::cout << "\nTest 6: Template Specialization" << std::endl;
        print(42);
        print(3.14);
        print(true);   // Uses specialized version
        print(false);  // Uses specialized version
    }

    // Test 7: Variadic templates
    {
        std::cout << "\nTest 7: Variadic Templates" << std::endl;
        std::cout << "Multiple values: ";
        printAll(1, 2.5, "hello", 'x', true);
    }

    // Test 8: Compile-time computation
    {
        std::cout << "\nTest 8: Compile-Time Computation" << std::endl;
        constexpr int f5 = factorial<5>();
        constexpr int f10 = factorial<10>();
        std::cout << "5! = " << f5 << std::endl;
        std::cout << "10! = " << f10 << std::endl;
    }

    // Test 9: SFINAE
    {
        std::cout << "\nTest 9: SFINAE (square for numeric types)" << std::endl;
        std::cout << "square(5) = " << square(5) << std::endl;
        std::cout << "square(3.5) = " << square(3.5) << std::endl;
        // square("hello") would not compile - SFINAE excludes non-numeric types
    }

    // Test 10: Default template parameter
    {
        std::cout << "\nTest 10: Default Template Parameter" << std::endl;
        DefaultTensor<> t1(10);  // Uses default (float)
        DefaultTensor<double> t2(10);  // Explicit type
        std::cout << "Default tensor size: " << t1.size() << std::endl;
        std::cout << "Double tensor size: " << t2.size() << std::endl;
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
