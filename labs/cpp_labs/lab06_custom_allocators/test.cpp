#include <iostream>
#include <cassert>
#include <vector>
#include <cstdlib>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

class MemoryPool {
private:
    char* pool_;
    size_t size_;
    size_t used_;
public:
    MemoryPool(size_t size) : size_(size), used_(0) {
        pool_ = static_cast<char*>(std::malloc(size));
    }
    ~MemoryPool() { std::free(pool_); }
    void* allocate(size_t n) {
        if (used_ + n > size_) throw std::bad_alloc();
        void* ptr = pool_ + used_;
        used_ += n;
        return ptr;
    }
    void deallocate(void*) {}
    void reset() { used_ = 0; }
    size_t bytesUsed() const { return used_; }
};

TEST(pool_creation) {
    MemoryPool pool(1024);
    assert(pool.bytesUsed() == 0);
}

TEST(pool_allocation) {
    MemoryPool pool(1024);
    void* p1 = pool.allocate(100);
    assert(p1 != nullptr);
    assert(pool.bytesUsed() == 100);

    void* p2 = pool.allocate(200);
    assert(p2 != nullptr);
    assert(pool.bytesUsed() == 300);
}

TEST(pool_reset) {
    MemoryPool pool(1024);
    pool.allocate(500);
    assert(pool.bytesUsed() == 500);
    pool.reset();
    assert(pool.bytesUsed() == 0);
}

int main() {
    std::cout << "=== Running Custom Allocators Tests ===" << std::endl;
    RUN_TEST(pool_creation);
    RUN_TEST(pool_allocation);
    RUN_TEST(pool_reset);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
