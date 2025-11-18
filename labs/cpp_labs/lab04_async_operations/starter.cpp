#include <iostream>
#include <future>
#include <chrono>
#include <thread>

// Simulates a long-running computation
int expensiveComputation(int n) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return n * n;
}

// TODO: Implement async version using std::async
// std::future<int> computeAsync(int n) {
//     return std::async(std::launch::async, expensiveComputation, n);
// }

// TODO: Implement promise-based version
// void computeWithPromise(std::promise<int>&& promise, int n) {
//     // Compute and set the promise value
// }

int main() {
    std::cout << "=== Lab 04: Async Operations (Starter) ===" << std::endl;

    // Synchronous version (slow!)
    {
        std::cout << "\nSynchronous execution:" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        int r1 = expensiveComputation(5);
        int r2 = expensiveComputation(10);
        int r3 = expensiveComputation(15);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Results: " << r1 << ", " << r2 << ", " << r3 << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
    }

    // TODO: Asynchronous version using std::async
    // {
    //     std::cout << "\nAsynchronous execution:" << std::endl;
    //     auto start = std::chrono::high_resolution_clock::now();
    //
    //     auto f1 = std::async(std::launch::async, expensiveComputation, 5);
    //     auto f2 = std::async(std::launch::async, expensiveComputation, 10);
    //     auto f3 = std::async(std::launch::async, expensiveComputation, 15);
    //
    //     int r1 = f1.get();
    //     int r2 = f2.get();
    //     int r3 = f3.get();
    //
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    //     std::cout << "Results: " << r1 << ", " << r2 << ", " << r3 << std::endl;
    //     std::cout << "Time: " << duration.count() << " ms" << std::endl;
    // }

    return 0;
}
