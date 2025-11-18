#include <iostream>
#include <cassert>
#include <future>
#include <thread>

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { std::cout << "Running " #name "..."; test_##name(); std::cout << " PASSED" << std::endl; } while(0)

TEST(async_basic) {
    auto future = std::async(std::launch::async, []() { return 42; });
    assert(future.get() == 42);
}

TEST(async_with_params) {
    auto future = std::async(std::launch::async, [](int a, int b) { return a + b; }, 10, 20);
    assert(future.get() == 30);
}

TEST(promise_future) {
    std::promise<int> promise;
    auto future = promise.get_future();

    std::thread t([&promise]() {
        promise.set_value(100);
    });

    assert(future.get() == 100);
    t.join();
}

TEST(multiple_futures) {
    auto f1 = std::async(std::launch::async, []() { return 1; });
    auto f2 = std::async(std::launch::async, []() { return 2; });
    auto f3 = std::async(std::launch::async, []() { return 3; });

    assert(f1.get() + f2.get() + f3.get() == 6);
}

TEST(future_wait_for) {
    auto future = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 42;
    });

    auto status = future.wait_for(std::chrono::milliseconds(1));
    assert(status == std::future_status::timeout);

    assert(future.get() == 42);
}

int main() {
    std::cout << "=== Running Async Operations Tests ===" << std::endl;
    RUN_TEST(async_basic);
    RUN_TEST(async_with_params);
    RUN_TEST(promise_future);
    RUN_TEST(multiple_futures);
    RUN_TEST(future_wait_for);
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
