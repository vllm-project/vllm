#include <iostream>
#include <vector>
#include <cmath>

// TODO: Implement a template function for computing maximum of two values
// template<typename T>
// T max(T a, T b) {
//     return (a > b) ? a : b;
// }

// TODO: Implement a template class for a simple 1D Tensor
// template<typename T>
// class Tensor {
// private:
//     std::vector<T> data_;
// public:
//     // TODO: Implement constructor, size(), at(), etc.
// };

// Non-template version (limited to floats)
class FloatTensor {
private:
    std::vector<float> data_;

public:
    FloatTensor(size_t size) : data_(size) {}

    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    size_t size() const { return data_.size(); }

    // TODO: This only works for float - make it a template!
    float sum() const {
        float result = 0.0f;
        for (const auto& val : data_) {
            result += val;
        }
        return result;
    }
};

// TODO: Implement template function for dot product
// template<typename T>
// T dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
//     // TODO: Compute dot product
// }

// TODO: Implement template function for element-wise operation
// template<typename T, typename Func>
// Tensor<T> transform(const Tensor<T>& input, Func op) {
//     // TODO: Apply operation to each element
// }

// TODO: Implement template specialization for printing
// Generic version
// template<typename T>
// void print(const T& value) {
//     std::cout << value << std::endl;
// }

// TODO: Specialize for bool to print "true"/"false"
// template<>
// void print<bool>(const bool& value) {
//     std::cout << (value ? "true" : "false") << std::endl;
// }

int main() {
    std::cout << "=== Lab 03: Templates (Starter) ===" << std::endl;

    // Test 1: FloatTensor (non-generic)
    {
        std::cout << "\nTest 1: FloatTensor (float only)" << std::endl;
        FloatTensor t(5);
        for (size_t i = 0; i < t.size(); ++i) {
            t[i] = static_cast<float>(i + 1);
        }

        std::cout << "Sum: " << t.sum() << std::endl;

        // Problem: Can't use with double or int!
        // FloatTensor won't work with other types
    }

    // TODO: Test 2: Generic Tensor with different types
    // {
    //     std::cout << "\nTest 2: Generic Tensor<T>" << std::endl;
    //     Tensor<double> t1(5);
    //     Tensor<int> t2(5);
    //     // TODO: Test with different types
    // }

    // TODO: Test 3: Template functions
    // {
    //     std::cout << "\nTest 3: Template Functions" << std::endl;
    //     auto m1 = max(10, 20);
    //     auto m2 = max(3.14, 2.71);
    //     // TODO: Test max with different types
    // }

    std::cout << "\n=== End of tests ===" << std::endl;
    return 0;
}
