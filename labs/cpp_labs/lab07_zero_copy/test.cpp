#include <iostream>
#include <cassert>
#include <vector>
#include <span>
#include <string_view>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

template<typename T>
class TensorView {
private:
    T* data_;
    size_t size_;
public:
    TensorView(T* data, size_t size) : data_(data), size_(size) {}
    TensorView(std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}
    T& operator[](size_t i) { return data_[i]; }
    size_t size() const { return size_; }
    TensorView<T> slice(size_t start, size_t length) {
        return TensorView<T>(data_ + start, length);
    }
};

TEST(span_basic) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    std::span<int> sp(vec);
    assert(sp.size() == 5);
    assert(sp[0] == 1);
}

TEST(string_view_basic) {
    std::string str = "hello";
    std::string_view view = str;
    assert(view.size() == 5);
    assert(view == "hello");
}

TEST(tensor_view_basic) {
    std::vector<float> data{1.0f, 2.0f, 3.0f};
    TensorView<float> view(data);
    assert(view.size() == 3);
    assert(view[0] == 1.0f);
}

TEST(tensor_view_slice) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    TensorView<float> view(data);
    auto slice = view.slice(1, 3);
    assert(slice.size() == 3);
    assert(slice[0] == 2.0f);
}

TEST(view_modifies_original) {
    std::vector<int> data{1, 2, 3};
    TensorView<int> view(data);
    view[0] = 99;
    assert(data[0] == 99);
}

int main() {
    std::cout << "=== Running Zero-Copy Tests ===" << std::endl;
    RUN_TEST(span_basic);
    RUN_TEST(string_view_basic);
    RUN_TEST(tensor_view_basic);
    RUN_TEST(tensor_view_slice);
    RUN_TEST(view_modifies_original);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
