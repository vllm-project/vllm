#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

// Simulates a GPU memory buffer with proper RAII
class GPUBuffer {
private:
    // Using unique_ptr for automatic memory management
    // Why: Buffer owns its data exclusively, no sharing needed
    std::unique_ptr<float[]> data_;
    size_t size_;

public:
    explicit GPUBuffer(size_t size) : size_(size) {
        // make_unique is exception-safe and preferred over 'new'
        data_ = std::make_unique<float[]>(size);
        std::cout << "Creating GPU buffer of size " << size << std::endl;
    }

    // Destructor is automatically generated and will call unique_ptr's destructor
    // which will delete[] the array - RAII in action!
    ~GPUBuffer() {
        std::cout << "Destroying GPU buffer of size " << size_ << std::endl;
    }

    // Delete copy operations to prevent accidental copying
    // Why: Copying large GPU buffers is expensive and usually unintended
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    // Enable move operations for efficient transfer of ownership
    // Why: Allows returning GPUBuffer from functions without copying
    GPUBuffer(GPUBuffer&&) = default;
    GPUBuffer& operator=(GPUBuffer&&) = default;

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }

    void fill(float value) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }

    float get(size_t index) const {
        if (index < size_) {
            return data_[index];
        }
        throw std::out_of_range("Index out of bounds");
    }
};

// Manages multiple GPU buffers using smart pointers
class BufferManager {
private:
    // Using vector of unique_ptr for automatic cleanup
    // Why: Manager owns the buffers, they should be deleted when manager is destroyed
    std::vector<std::unique_ptr<GPUBuffer>> buffers_;

public:
    // Default constructor and destructor work perfectly due to RAII
    // No manual cleanup needed!
    BufferManager() = default;
    ~BufferManager() = default;

    // Delete copy, enable move
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    BufferManager(BufferManager&&) = default;
    BufferManager& operator=(BufferManager&&) = default;

    void allocateBuffer(size_t size) {
        // make_unique creates the buffer and transfers ownership to the vector
        buffers_.push_back(std::make_unique<GPUBuffer>(size));
    }

    // Returns raw pointer for access (non-owning)
    // Why: Caller doesn't need ownership, just access
    GPUBuffer* getBuffer(size_t index) {
        if (index < buffers_.size()) {
            return buffers_[index].get();
        }
        return nullptr;
    }

    size_t bufferCount() const {
        return buffers_.size();
    }

    // Remove a buffer (demonstrates unique_ptr move semantics)
    std::unique_ptr<GPUBuffer> releaseBuffer(size_t index) {
        if (index < buffers_.size()) {
            auto buffer = std::move(buffers_[index]);
            buffers_.erase(buffers_.begin() + index);
            return buffer;
        }
        return nullptr;
    }
};

// Demonstrates shared ownership using shared_ptr
// Why: Multiple components might need to keep a buffer alive
class SharedBuffer {
private:
    // Using shared_ptr for reference-counted ownership
    std::shared_ptr<float[]> data_;
    size_t size_;

public:
    explicit SharedBuffer(size_t size) : size_(size) {
        // make_shared is more efficient than shared_ptr<T>(new T)
        // It allocates control block and data in one allocation
        data_ = std::shared_ptr<float[]>(new float[size]);
        std::cout << "Creating shared buffer of size " << size << std::endl;
    }

    ~SharedBuffer() {
        std::cout << "Destroying shared buffer (ref count: "
                  << data_.use_count() << ")" << std::endl;
    }

    // Copy constructor creates a new SharedBuffer sharing the same data
    // Why: Multiple SharedBuffer objects can safely share the underlying array
    SharedBuffer(const SharedBuffer&) = default;
    SharedBuffer& operator=(const SharedBuffer&) = default;

    // Move operations
    SharedBuffer(SharedBuffer&&) = default;
    SharedBuffer& operator=(SharedBuffer&&) = default;

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }

    // Get current reference count
    long useCount() const { return data_.use_count(); }
};

// Demonstrates weak_ptr to prevent circular references
void demonstrateWeakPtr() {
    std::cout << "\nDemonstrating weak_ptr:" << std::endl;

    // Create a shared_ptr
    auto shared = std::make_shared<int>(42);
    std::cout << "Created shared_ptr, use_count: " << shared.use_count() << std::endl;

    // Create a weak_ptr from the shared_ptr
    // Why: weak_ptr doesn't increase reference count, prevents cycles
    std::weak_ptr<int> weak = shared;
    std::cout << "Created weak_ptr, shared use_count: " << shared.use_count() << std::endl;

    // Access through weak_ptr using lock()
    // Why: lock() safely checks if object still exists
    if (auto locked = weak.lock()) {
        std::cout << "Weak ptr locked successfully, value: " << *locked << std::endl;
        std::cout << "Locked use_count: " << locked.use_count() << std::endl;
    }

    // Reset the shared_ptr
    std::cout << "Resetting shared_ptr..." << std::endl;
    shared.reset();

    // Try to lock again - should fail
    if (auto locked = weak.lock()) {
        std::cout << "Weak ptr still valid (unexpected!)" << std::endl;
    } else {
        std::cout << "Weak ptr expired (expected - object was deleted)" << std::endl;
    }
}

// Custom deleter example - useful for GPU resources
void demonstrateCustomDeleter() {
    std::cout << "\nDemonstrating custom deleter:" << std::endl;

    // Custom deleter that logs when resource is freed
    auto deleter = [](float* ptr) {
        std::cout << "Custom deleter called for buffer" << std::endl;
        delete[] ptr;
    };

    // Create unique_ptr with custom deleter
    std::unique_ptr<float[], decltype(deleter)> buffer(new float[100], deleter);
    buffer[0] = 1.23f;
    std::cout << "Buffer[0] = " << buffer[0] << std::endl;

    // Deleter will be called automatically when buffer goes out of scope
}

int main() {
    std::cout << "=== Lab 01: Memory Management (Solution) ===" << std::endl;

    // Test 1: BufferManager with unique_ptr (no leaks!)
    {
        std::cout << "\nTest 1: BufferManager with RAII" << std::endl;
        BufferManager manager;
        manager.allocateBuffer(1024);
        manager.allocateBuffer(2048);

        GPUBuffer* buf = manager.getBuffer(0);
        if (buf) {
            buf->fill(3.14f);
            std::cout << "Filled buffer 0 with 3.14" << std::endl;
            std::cout << "Buffer[0] = " << buf->get(0) << std::endl;
        }

        std::cout << "Manager has " << manager.bufferCount() << " buffers" << std::endl;

        // All buffers automatically cleaned up when manager goes out of scope!
        std::cout << "Exiting scope - automatic cleanup..." << std::endl;
    }

    // Test 2: SharedBuffer with shared_ptr
    {
        std::cout << "\nTest 2: SharedBuffer with shared ownership" << std::endl;

        SharedBuffer buf1(512);
        std::cout << "buf1 use_count: " << buf1.useCount() << std::endl;

        {
            // Create a copy - shares the same underlying data
            SharedBuffer buf2 = buf1;
            std::cout << "Created buf2 (copy), buf1 use_count: " << buf1.useCount() << std::endl;
            std::cout << "buf2 use_count: " << buf2.useCount() << std::endl;

            // Both point to same memory
            buf1.data()[0] = 99.9f;
            std::cout << "Set buf1[0] to 99.9" << std::endl;
            std::cout << "buf2[0] = " << buf2.data()[0] << " (same memory!)" << std::endl;

            std::cout << "buf2 going out of scope..." << std::endl;
        }

        std::cout << "After buf2 destroyed, buf1 use_count: " << buf1.useCount() << std::endl;
        std::cout << "buf1 going out of scope..." << std::endl;
    }

    // Test 3: Weak pointer demonstration
    {
        std::cout << "\nTest 3: Weak Pointer to prevent cycles" << std::endl;
        demonstrateWeakPtr();
    }

    // Test 4: Custom deleter
    {
        demonstrateCustomDeleter();
    }

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    std::cout << "No memory leaks! Verify with:" << std::endl;
    std::cout << "valgrind --leak-check=full ./solution" << std::endl;

    return 0;
}
