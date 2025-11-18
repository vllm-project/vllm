#include <iostream>
#include <cassert>
#include <thread>
#include <mutex>
#include <vector>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

class SafeCounter {
private:
    int value_ = 0;
    mutable std::mutex mutex_;
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mutex_);
        value_++;
    }
    int get() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }
};

TEST(thread_creation) {
    bool executed = false;
    std::thread t([&executed]() { executed = true; });
    t.join();
    assert(executed);
}

TEST(safe_counter) {
    SafeCounter counter;
    std::vector<std::thread> threads;
    const int NUM_THREADS = 10;
    const int ITERATIONS = 1000;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < ITERATIONS; ++j) {
                counter.increment();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    assert(counter.get() == NUM_THREADS * ITERATIONS);
}

TEST(mutex_protection) {
    std::mutex mtx;
    int value = 0;

    std::thread t1([&mtx, &value]() {
        std::lock_guard<std::mutex> lock(mtx);
        value = 1;
    });

    std::thread t2([&mtx, &value]() {
        std::lock_guard<std::mutex> lock(mtx);
        value = 2;
    });

    t1.join();
    t2.join();

    assert(value == 1 || value == 2);
}

int main() {
    std::cout << "=== Running Threading Tests ===" << std::endl;
    RUN_TEST(thread_creation);
    RUN_TEST(safe_counter);
    RUN_TEST(mutex_protection);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
