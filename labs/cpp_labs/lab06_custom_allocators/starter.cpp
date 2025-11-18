#include <iostream>
#include <vector>
#include <memory>

// TODO: Implement a simple MemoryPool allocator
// class MemoryPool {
// private:
//     char* pool_;
//     size_t size_;
//     size_t used_;
// public:
//     MemoryPool(size_t size);
//     void* allocate(size_t n);
//     void deallocate(void* p);
// };

int main() {
    std::cout << "=== Lab 06: Custom Allocators (Starter) ===" << std::endl;

    // Standard allocator (baseline)
    {
        std::cout << "\nUsing standard allocator" << std::endl;
        std::vector<int> vec;
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }
        std::cout << "Vector size: " << vec.size() << std::endl;
    }

    // TODO: Use custom allocator
    // {
    //     std::cout << "\nUsing custom allocator" << std::endl;
    //     std::vector<int, PoolAllocator<int>> vec;
    //     // ...
    // }

    return 0;
}
