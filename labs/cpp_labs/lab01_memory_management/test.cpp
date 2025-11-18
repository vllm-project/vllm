#include <iostream>
#include <cassert>
#include <memory>
#include <vector>

// Minimal test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "..."; \
    test_##name(); \
    std::cout << " PASSED" << std::endl; \
} while(0)

// Copy the solution classes here for testing
// In a real project, these would be in a header file

class GPUBuffer {
private:
    std::unique_ptr<float[]> data_;
    size_t size_;
    static int instance_count_;  // For testing

public:
    explicit GPUBuffer(size_t size) : size_(size) {
        data_ = std::make_unique<float[]>(size);
        instance_count_++;
    }

    ~GPUBuffer() {
        instance_count_--;
    }

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
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

    static int instanceCount() { return instance_count_; }
    static void resetInstanceCount() { instance_count_ = 0; }
};

int GPUBuffer::instance_count_ = 0;

class BufferManager {
private:
    std::vector<std::unique_ptr<GPUBuffer>> buffers_;

public:
    BufferManager() = default;
    ~BufferManager() = default;

    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;
    BufferManager(BufferManager&&) = default;
    BufferManager& operator=(BufferManager&&) = default;

    void allocateBuffer(size_t size) {
        buffers_.push_back(std::make_unique<GPUBuffer>(size));
    }

    GPUBuffer* getBuffer(size_t index) {
        if (index < buffers_.size()) {
            return buffers_[index].get();
        }
        return nullptr;
    }

    size_t bufferCount() const {
        return buffers_.size();
    }

    std::unique_ptr<GPUBuffer> releaseBuffer(size_t index) {
        if (index < buffers_.size()) {
            auto buffer = std::move(buffers_[index]);
            buffers_.erase(buffers_.begin() + index);
            return buffer;
        }
        return nullptr;
    }
};

class SharedBuffer {
private:
    std::shared_ptr<float[]> data_;
    size_t size_;

public:
    explicit SharedBuffer(size_t size) : size_(size) {
        data_ = std::shared_ptr<float[]>(new float[size]);
    }

    SharedBuffer(const SharedBuffer&) = default;
    SharedBuffer& operator=(const SharedBuffer&) = default;
    SharedBuffer(SharedBuffer&&) = default;
    SharedBuffer& operator=(SharedBuffer&&) = default;

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }
    long useCount() const { return data_.use_count(); }
};

// Test cases

TEST(unique_ptr_basic) {
    auto ptr = std::make_unique<int>(42);
    assert(*ptr == 42);
    assert(ptr.get() != nullptr);
}

TEST(unique_ptr_array) {
    auto arr = std::make_unique<int[]>(10);
    arr[0] = 1;
    arr[9] = 10;
    assert(arr[0] == 1);
    assert(arr[9] == 10);
}

TEST(unique_ptr_move) {
    auto ptr1 = std::make_unique<int>(100);
    auto ptr2 = std::move(ptr1);

    assert(ptr1.get() == nullptr);  // ptr1 no longer owns the object
    assert(ptr2.get() != nullptr);
    assert(*ptr2 == 100);
}

TEST(shared_ptr_basic) {
    auto ptr = std::make_shared<int>(42);
    assert(*ptr == 42);
    assert(ptr.use_count() == 1);
}

TEST(shared_ptr_sharing) {
    auto ptr1 = std::make_shared<int>(100);
    assert(ptr1.use_count() == 1);

    {
        auto ptr2 = ptr1;  // Share ownership
        assert(ptr1.use_count() == 2);
        assert(ptr2.use_count() == 2);
        assert(*ptr1 == *ptr2);
    }

    // ptr2 went out of scope
    assert(ptr1.use_count() == 1);
    assert(*ptr1 == 100);  // Still valid
}

TEST(weak_ptr_basic) {
    std::weak_ptr<int> weak;

    {
        auto shared = std::make_shared<int>(42);
        weak = shared;

        assert(shared.use_count() == 1);  // weak_ptr doesn't increase count

        auto locked = weak.lock();
        assert(locked != nullptr);
        assert(*locked == 42);
    }

    // shared went out of scope
    auto locked = weak.lock();
    assert(locked == nullptr);  // Object is gone
}

TEST(gpu_buffer_creation) {
    GPUBuffer::resetInstanceCount();
    {
        GPUBuffer buf(100);
        assert(buf.size() == 100);
        assert(buf.data() != nullptr);
        assert(GPUBuffer::instanceCount() == 1);
    }
    assert(GPUBuffer::instanceCount() == 0);  // Cleaned up
}

TEST(gpu_buffer_fill) {
    GPUBuffer buf(10);
    buf.fill(3.14f);

    for (size_t i = 0; i < buf.size(); ++i) {
        assert(buf.get(i) == 3.14f);
    }
}

TEST(gpu_buffer_move) {
    GPUBuffer::resetInstanceCount();

    GPUBuffer buf1(100);
    assert(GPUBuffer::instanceCount() == 1);

    GPUBuffer buf2(std::move(buf1));
    assert(GPUBuffer::instanceCount() == 1);  // Moved, not copied
    assert(buf2.size() == 100);
}

TEST(buffer_manager_basic) {
    GPUBuffer::resetInstanceCount();

    {
        BufferManager manager;
        manager.allocateBuffer(100);
        manager.allocateBuffer(200);

        assert(manager.bufferCount() == 2);
        assert(GPUBuffer::instanceCount() == 2);

        GPUBuffer* buf = manager.getBuffer(0);
        assert(buf != nullptr);
        assert(buf->size() == 100);
    }

    // Manager destroyed, all buffers should be cleaned up
    assert(GPUBuffer::instanceCount() == 0);
}

TEST(buffer_manager_release) {
    GPUBuffer::resetInstanceCount();

    BufferManager manager;
    manager.allocateBuffer(100);
    manager.allocateBuffer(200);

    assert(manager.bufferCount() == 2);
    assert(GPUBuffer::instanceCount() == 2);

    // Release first buffer
    auto released = manager.releaseBuffer(0);
    assert(released != nullptr);
    assert(released->size() == 100);
    assert(manager.bufferCount() == 1);
    assert(GPUBuffer::instanceCount() == 2);  // Still 2, we own the released one

    released.reset();  // Delete the released buffer
    assert(GPUBuffer::instanceCount() == 1);
}

TEST(shared_buffer_basic) {
    SharedBuffer buf(100);
    assert(buf.size() == 100);
    assert(buf.useCount() == 1);
}

TEST(shared_buffer_sharing) {
    SharedBuffer buf1(100);
    assert(buf1.useCount() == 1);

    {
        SharedBuffer buf2 = buf1;
        assert(buf1.useCount() == 2);
        assert(buf2.useCount() == 2);

        // They share the same underlying data
        buf1.data()[0] = 99.9f;
        assert(buf2.data()[0] == 99.9f);
    }

    assert(buf1.useCount() == 1);  // buf2 destroyed
}

TEST(shared_buffer_independence) {
    SharedBuffer buf1(100);
    SharedBuffer buf2(100);

    buf1.data()[0] = 1.0f;
    buf2.data()[0] = 2.0f;

    // Different buffers, different data
    assert(buf1.data()[0] == 1.0f);
    assert(buf2.data()[0] == 2.0f);
    assert(buf1.useCount() == 1);
    assert(buf2.useCount() == 1);
}

TEST(custom_deleter) {
    bool deleted = false;

    {
        auto deleter = [&deleted](int* ptr) {
            deleted = true;
            delete ptr;
        };

        std::unique_ptr<int, decltype(deleter)> ptr(new int(42), deleter);
        assert(!deleted);
    }

    assert(deleted);  // Deleter was called
}

TEST(multiple_buffer_managers) {
    GPUBuffer::resetInstanceCount();

    {
        BufferManager mgr1, mgr2;
        mgr1.allocateBuffer(100);
        mgr2.allocateBuffer(200);

        assert(GPUBuffer::instanceCount() == 2);
    }

    assert(GPUBuffer::instanceCount() == 0);
}

TEST(vector_of_unique_ptrs) {
    GPUBuffer::resetInstanceCount();

    {
        std::vector<std::unique_ptr<GPUBuffer>> buffers;
        buffers.push_back(std::make_unique<GPUBuffer>(100));
        buffers.push_back(std::make_unique<GPUBuffer>(200));

        assert(buffers.size() == 2);
        assert(GPUBuffer::instanceCount() == 2);

        // Remove one
        buffers.erase(buffers.begin());
        assert(buffers.size() == 1);
        assert(GPUBuffer::instanceCount() == 1);
    }

    assert(GPUBuffer::instanceCount() == 0);
}

int main() {
    std::cout << "=== Running Memory Management Tests ===" << std::endl;

    RUN_TEST(unique_ptr_basic);
    RUN_TEST(unique_ptr_array);
    RUN_TEST(unique_ptr_move);
    RUN_TEST(shared_ptr_basic);
    RUN_TEST(shared_ptr_sharing);
    RUN_TEST(weak_ptr_basic);
    RUN_TEST(gpu_buffer_creation);
    RUN_TEST(gpu_buffer_fill);
    RUN_TEST(gpu_buffer_move);
    RUN_TEST(buffer_manager_basic);
    RUN_TEST(buffer_manager_release);
    RUN_TEST(shared_buffer_basic);
    RUN_TEST(shared_buffer_sharing);
    RUN_TEST(shared_buffer_independence);
    RUN_TEST(custom_deleter);
    RUN_TEST(multiple_buffer_managers);
    RUN_TEST(vector_of_unique_ptrs);

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
