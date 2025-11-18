#include <iostream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <chrono>

// Simple memory pool (pre-allocated block)
class MemoryPool {
private:
    char* pool_;
    size_t size_;
    size_t used_;
    size_t alloc_count_;

public:
    MemoryPool(size_t size) : size_(size), used_(0), alloc_count_(0) {
        pool_ = static_cast<char*>(std::malloc(size));
        std::cout << "Created memory pool of " << size << " bytes" << std::endl;
    }

    ~MemoryPool() {
        std::cout << "Destroying pool (allocated " << alloc_count_ << " times)" << std::endl;
        std::free(pool_);
    }

    void* allocate(size_t n) {
        if (used_ + n > size_) {
            throw std::bad_alloc();
        }
        void* ptr = pool_ + used_;
        used_ += n;
        alloc_count_++;
        return ptr;
    }

    void deallocate(void* /*p*/) {
        // Simple pool doesn't actually free - resets on destruction
    }

    void reset() {
        used_ = 0;
        alloc_count_ = 0;
    }

    size_t bytesUsed() const { return used_; }
};

// STL-compatible allocator using memory pool
template<typename T>
class PoolAllocator {
private:
    MemoryPool* pool_;

public:
    using value_type = T;

    PoolAllocator(MemoryPool* pool) : pool_(pool) {}

    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}

    T* allocate(size_t n) {
        return static_cast<T*>(pool_->allocate(n * sizeof(T)));
    }

    void deallocate(T* p, size_t /*n*/) {
        pool_->deallocate(p);
    }

    template<typename U>
    friend class PoolAllocator;
};

template<typename T, typename U>
bool operator==(const PoolAllocator<T>& a, const PoolAllocator<U>& b) {
    return a.pool_ == b.pool_;
}

template<typename T, typename U>
bool operator!=(const PoolAllocator<T>& a, const PoolAllocator<U>& b) {
    return !(a == b);
}

int main() {
    std::cout << "=== Lab 06: Custom Allocators (Solution) ===" << std::endl;

    // Test 1: Standard allocator (baseline)
    {
        std::cout << "\nTest 1: Standard Allocator" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> vec;
        for (int i = 0; i < 10000; ++i) {
            vec.push_back(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Vector size: " << vec.size() << std::endl;
        std::cout << "Time: " << duration.count() << " μs" << std::endl;
    }

    // Test 2: Custom pool allocator
    {
        std::cout << "\nTest 2: Pool Allocator" << std::endl;
        MemoryPool pool(1024 * 1024);  // 1 MB pool

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int, PoolAllocator<int>> vec((PoolAllocator<int>(&pool)));
        for (int i = 0; i < 10000; ++i) {
            vec.push_back(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Vector size: " << vec.size() << std::endl;
        std::cout << "Pool bytes used: " << pool.bytesUsed() << std::endl;
        std::cout << "Time: " << duration.count() << " μs" << std::endl;
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
