#include <iostream>
#include <future>
#include <chrono>
#include <thread>
#include <vector>

// Simulates expensive computation (like GPU kernel launch)
int expensiveComputation(int n) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return n * n;
}

// Async version using std::async
std::future<int> computeAsync(int n) {
    return std::async(std::launch::async, expensiveComputation, n);
}

// Promise-based version
void computeWithPromise(std::promise<int> promise, int n) {
    try {
        int result = expensiveComputation(n);
        promise.set_value(result);
    } catch (...) {
        promise.set_exception(std::current_exception());
    }
}

// Simulate async inference request
struct InferenceResult {
    int request_id;
    std::vector<float> output;
    std::chrono::milliseconds latency;
};

std::future<InferenceResult> processRequestAsync(int request_id) {
    return std::async(std::launch::async, [request_id]() {
        auto start = std::chrono::high_resolution_clock::now();

        // Simulate inference
        std::this_thread::sleep_for(std::chrono::milliseconds(50 + request_id * 10));
        std::vector<float> output(10, static_cast<float>(request_id));

        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        return InferenceResult{request_id, output, latency};
    });
}

int main() {
    std::cout << "=== Lab 04: Async Operations (Solution) ===" << std::endl;

    // Test 1: Synchronous vs Asynchronous
    {
        std::cout << "\nTest 1: Sync vs Async Comparison" << std::endl;

        // Synchronous
        auto start = std::chrono::high_resolution_clock::now();
        int r1 = expensiveComputation(5);
        int r2 = expensiveComputation(10);
        int r3 = expensiveComputation(15);
        auto end = std::chrono::high_resolution_clock::now();
        auto sync_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Sync - Results: " << r1 << ", " << r2 << ", " << r3 << std::endl;
        std::cout << "Sync - Time: " << sync_duration.count() << " ms" << std::endl;

        // Asynchronous
        start = std::chrono::high_resolution_clock::now();
        auto f1 = std::async(std::launch::async, expensiveComputation, 5);
        auto f2 = std::async(std::launch::async, expensiveComputation, 10);
        auto f3 = std::async(std::launch::async, expensiveComputation, 15);

        r1 = f1.get();
        r2 = f2.get();
        r3 = f3.get();
        end = std::chrono::high_resolution_clock::now();
        auto async_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Async - Results: " << r1 << ", " << r2 << ", " << r3 << std::endl;
        std::cout << "Async - Time: " << async_duration.count() << " ms" << std::endl;
        std::cout << "Speedup: " << (double)sync_duration.count() / async_duration.count() << "x" << std::endl;
    }

    // Test 2: Promise and Future
    {
        std::cout << "\nTest 2: Promise and Future" << std::endl;

        std::promise<int> promise;
        std::future<int> future = promise.get_future();

        std::thread worker(computeWithPromise, std::move(promise), 7);

        std::cout << "Waiting for result..." << std::endl;
        int result = future.get();
        std::cout << "Result: " << result << std::endl;

        worker.join();
    }

    // Test 3: Multiple async inference requests
    {
        std::cout << "\nTest 3: Async Inference Batch" << std::endl;

        std::vector<std::future<InferenceResult>> futures;
        const int NUM_REQUESTS = 5;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_REQUESTS; ++i) {
            futures.push_back(processRequestAsync(i));
        }

        std::cout << "Processing " << NUM_REQUESTS << " requests..." << std::endl;

        for (auto& future : futures) {
            auto result = future.get();
            std::cout << "Request " << result.request_id
                      << " completed in " << result.latency.count() << " ms" << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Total batch time: " << total_duration.count() << " ms" << std::endl;
    }

    // Test 4: wait_for with timeout
    {
        std::cout << "\nTest 4: Future with Timeout" << std::endl;

        auto future = std::async(std::launch::async, []() {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            return 42;
        });

        std::cout << "Checking if ready..." << std::endl;
        auto status = future.wait_for(std::chrono::milliseconds(100));

        if (status == std::future_status::ready) {
            std::cout << "Result ready: " << future.get() << std::endl;
        } else if (status == std::future_status::timeout) {
            std::cout << "Timeout! Still waiting..." << std::endl;
            int result = future.get();  // Wait for completion
            std::cout << "Eventually got: " << result << std::endl;
        }
    }

    std::cout << "\n=== All tests completed! ===" << std::endl;
    return 0;
}
