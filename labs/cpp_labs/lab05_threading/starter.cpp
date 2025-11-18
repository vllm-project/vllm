#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

int counter = 0;  // Shared variable (NOT thread-safe!)

// TODO: Add mutex to protect counter
// std::mutex counter_mutex;

void incrementCounter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // TODO: Lock mutex before accessing counter
        counter++;
        // TODO: Unlock mutex
    }
}

int main() {
    std::cout << "=== Lab 05: Threading (Starter) ===" << std::endl;

    const int NUM_THREADS = 4;
    const int ITERATIONS = 10000;

    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(incrementCounter, ITERATIONS);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Expected: " << NUM_THREADS * ITERATIONS << std::endl;
    std::cout << "Actual: " << counter << std::endl;
    std::cout << "Race condition!" << (counter != NUM_THREADS * ITERATIONS ? " YES" : " NO") << std::endl;

    return 0;
}
