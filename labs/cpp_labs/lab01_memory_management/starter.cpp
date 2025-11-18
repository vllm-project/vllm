#include <iostream>
#include <vector>
#include <cstring>

// Simulates a GPU memory buffer
class GPUBuffer {
private:
    float* data_;
    size_t size_;

public:
    GPUBuffer(size_t size) : size_(size) {
        // TODO: This uses raw new - should be managed by smart pointer
        data_ = new float[size];
        std::cout << "Creating GPU buffer of size " << size << std::endl;
    }

    // TODO: Implement proper destructor
    // Currently missing - memory leak!

    // TODO: Delete copy constructor and assignment to prevent double-free
    // Currently allows copying which can cause issues

    float* data() { return data_; }
    size_t size() const { return size_; }

    void fill(float value) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }
};

// Manages multiple GPU buffers
class BufferManager {
private:
    // TODO: Replace raw pointer with smart pointer
    std::vector<GPUBuffer*> buffers_;

public:
    // TODO: This creates a raw pointer - memory leak risk
    void allocateBuffer(size_t size) {
        GPUBuffer* buffer = new GPUBuffer(size);
        buffers_.push_back(buffer);
    }

    // TODO: Implement proper cleanup in destructor
    // Currently missing - all buffers leaked!

    GPUBuffer* getBuffer(size_t index) {
        if (index < buffers_.size()) {
            return buffers_[index];
        }
        return nullptr;
    }

    size_t bufferCount() const {
        return buffers_.size();
    }
};

// TODO: Implement a SharedBuffer class that uses shared_ptr
// This should allow multiple owners of the same buffer
class SharedBuffer {
private:
    // TODO: Use std::shared_ptr<float[]> instead of raw pointer
    float* data_;
    size_t size_;

public:
    SharedBuffer(size_t size) : size_(size) {
        data_ = new float[size];
        std::cout << "Creating shared buffer of size " << size << std::endl;
    }

    // TODO: Implement destructor (or better yet, use RAII with shared_ptr)

    // TODO: Implement proper copy constructor using shared ownership

    float* data() { return data_; }
    size_t size() const { return size_; }
};

// TODO: Implement a function that demonstrates weak_ptr usage
// This function should create a shared_ptr and a weak_ptr to it,
// then show what happens when the shared_ptr goes out of scope

void demonstrateWeakPtr() {
    // TODO: Create a shared_ptr to an int
    // TODO: Create a weak_ptr from the shared_ptr
    // TODO: Show that weak_ptr.lock() works while shared_ptr exists
    // TODO: Reset the shared_ptr and show that weak_ptr.lock() returns nullptr

    std::cout << "TODO: Implement weak_ptr demonstration" << std::endl;
}

int main() {
    std::cout << "=== Lab 01: Memory Management ===" << std::endl;

    // Test 1: BufferManager (has memory leaks!)
    {
        std::cout << "\nTest 1: BufferManager" << std::endl;
        BufferManager manager;
        manager.allocateBuffer(1024);
        manager.allocateBuffer(2048);

        GPUBuffer* buf = manager.getBuffer(0);
        if (buf) {
            buf->fill(3.14f);
            std::cout << "Filled buffer 0 with 3.14" << std::endl;
        }

        std::cout << "Manager has " << manager.bufferCount() << " buffers" << std::endl;
        // TODO: Buffers are leaked here when manager goes out of scope!
    }

    // Test 2: SharedBuffer
    {
        std::cout << "\nTest 2: SharedBuffer" << std::endl;
        // TODO: This should use shared_ptr to allow safe sharing
        SharedBuffer* buf1 = new SharedBuffer(512);
        // TODO: How do we safely share this buffer?
        // TODO: Who is responsible for deletion?

        delete buf1;  // Manual delete - error prone!
    }

    // Test 3: Weak pointer demonstration
    {
        std::cout << "\nTest 3: Weak Pointer" << std::endl;
        demonstrateWeakPtr();
    }

    std::cout << "\n=== End of tests ===" << std::endl;
    std::cout << "Note: Run with valgrind to see memory leaks!" << std::endl;
    std::cout << "valgrind --leak-check=full ./starter" << std::endl;

    return 0;
}
