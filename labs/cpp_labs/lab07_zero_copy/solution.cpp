#include <iostream>
#include <vector>
#include <string_view>
#include <span>
#include <cstring>

// TensorView - non-owning reference to tensor data
template<typename T>
class TensorView {
private:
    T* data_;
    size_t size_;

public:
    TensorView(T* data, size_t size) : data_(data), size_(size) {}

    // Construct from vector (no copy!)
    TensorView(std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    size_t size() const { return size_; }
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Slice view (zero-copy!)
    TensorView<T> slice(size_t start, size_t length) {
        if (start + length > size_) {
            throw std::out_of_range("Slice out of range");
        }
        return TensorView<T>(data_ + start, length);
    }
};

// Process with copy (expensive)
void processCopy(std::vector<int> vec) {
    std::cout << "Copy: " << vec.size() << " elements copied" << std::endl;
}

// Process by reference (zero-copy)
void processRef(const std::vector<int>& vec) {
    std::cout << "Ref: " << vec.size() << " elements, no copy" << std::endl;
}

// Process with span (zero-copy, more flexible)
void processSpan(std::span<const int> data) {
    std::cout << "Span: " << data.size() << " elements, no copy" << std::endl;
}

// String processing with string_view (zero-copy)
void processString(std::string_view str) {
    std::cout << "String view: " << str.size() << " chars, no copy" << std::endl;
}

int main() {
    std::cout << "=== Lab 07: Zero-Copy (Solution) ===" << std::endl;

    // Test 1: Copy vs reference
    {
        std::cout << "\nTest 1: Copy vs Reference" << std::endl;
        std::vector<int> data(10000, 42);

        processCopy(data);  // Copies all 10000 elements
        processRef(data);   // No copy!
    }

    // Test 2: std::span for array views
    {
        std::cout << "\nTest 2: std::span" << std::endl;
        std::vector<int> vec{1, 2, 3, 4, 5};
        int arr[] = {10, 20, 30};

        processSpan(vec);  // Works with vector
        processSpan(arr);  // Works with C array
        processSpan(std::span<const int>(vec).subspan(1, 3));  // Slicing!
    }

    // Test 3: string_view
    {
        std::cout << "\nTest 3: string_view" << std::endl;
        std::string str = "hello world";

        processString(str);              // No copy
        processString("literal");        // No copy
        processString(str.substr(0, 5)); // substr returns temp string, but string_view stores it
    }

    // Test 4: TensorView (custom zero-copy view)
    {
        std::cout << "\nTest 4: TensorView" << std::endl;
        std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        TensorView<float> view(data);
        std::cout << "Full view size: " << view.size() << std::endl;

        // Slice the view (zero-copy!)
        auto slice = view.slice(1, 3);
        std::cout << "Slice size: " << slice.size() << std::endl;
        std::cout << "Slice data: ";
        for (size_t i = 0; i < slice.size(); ++i) {
            std::cout << slice[i] << " ";
        }
        std::cout << std::endl;

        // Modify through slice (affects original!)
        slice[0] = 99.0f;
        std::cout << "Original data[1]: " << data[1] << " (modified!)" << std::endl;
    }

    // Test 5: Benchmark copy vs view
    {
        std::cout << "\nTest 5: Performance Comparison" << std::endl;
        std::vector<float> large(1000000, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> copy = large;
        auto end = std::chrono::high_resolution_clock::now();
        auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        start = std::chrono::high_resolution_clock::now();
        TensorView<float> view(large);
        end = std::chrono::high_resolution_clock::now();
        auto view_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        std::cout << "Copy 1M floats: " << copy_time.count() << " μs" << std::endl;
        std::cout << "View 1M floats: " << view_time.count() << " ns" << std::endl;
        std::cout << "View is ~" << (copy_time.count() * 1000) / std::max(1L, view_time.count())
                  << "x faster!" << std::endl;
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
