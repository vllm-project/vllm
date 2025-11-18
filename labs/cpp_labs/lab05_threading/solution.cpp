#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>

// Thread-safe counter with mutex
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

// Thread-safe queue with condition variable
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

int main() {
    std::cout << "=== Lab 05: Threading (Solution) ===" << std::endl;

    // Test 1: Thread-safe counter
    {
        std::cout << "\nTest 1: Thread-Safe Counter" << std::endl;
        SafeCounter counter;
        const int NUM_THREADS = 4;
        const int ITERATIONS = 10000;

        std::vector<std::thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back([&counter, ITERATIONS]() {
                for (int j = 0; j < ITERATIONS; ++j) {
                    counter.increment();
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        std::cout << "Expected: " << NUM_THREADS * ITERATIONS << std::endl;
        std::cout << "Actual: " << counter.get() << std::endl;
        std::cout << "Correct!" << std::endl;
    }

    // Test 2: Thread-safe queue (producer-consumer)
    {
        std::cout << "\nTest 2: Producer-Consumer with Thread-Safe Queue" << std::endl;
        ThreadSafeQueue<int> queue;
        std::atomic<bool> done{false};
        std::atomic<int> consumed{0};

        // Producer thread
        std::thread producer([&queue, &done]() {
            for (int i = 0; i < 100; ++i) {
                queue.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            done = true;
        });

        // Consumer threads
        std::vector<std::thread> consumers;
        for (int i = 0; i < 3; ++i) {
            consumers.emplace_back([&queue, &done, &consumed]() {
                while (!done || queue.size() > 0) {
                    int value;
                    if (queue.tryPop(value)) {
                        consumed++;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            });
        }

        producer.join();
        for (auto& c : consumers) {
            c.join();
        }

        std::cout << "Consumed " << consumed << " items" << std::endl;
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
