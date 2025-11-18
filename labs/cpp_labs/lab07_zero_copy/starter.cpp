#include <iostream>
#include <vector>
#include <string_view>
#include <span>

// TODO: Implement a TensorView class (non-owning reference to tensor data)
// class TensorView {
// private:
//     float* data_;
//     size_t size_;
// public:
//     TensorView(float* data, size_t size);
//     float& operator[](size_t i);
//     size_t size() const;
// };

void processCopy(std::vector<int> vec) {
    // Expensive copy!
    std::cout << "Processing copy of vector, size: " << vec.size() << std::endl;
}

void processRef(const std::vector<int>& vec) {
    // No copy - reference
    std::cout << "Processing vector by reference, size: " << vec.size() << std::endl;
}

// TODO: Use std::span for zero-copy array views
// void processSpan(std::span<int> data) {
//     std::cout << "Processing span, size: " << data.size() << std::endl;
// }

int main() {
    std::cout << "=== Lab 07: Zero-Copy (Starter) ===" << std::endl;

    std::vector<int> data(10000);

    // Copy vs reference
    processCopy(data);  // Expensive!
    processRef(data);   // Cheap!

    // TODO: Test std::span
    // processSpan(data);

    // TODO: Test string_view
    // std::string str = "hello world";
    // std::string_view view = str;

    return 0;
}
