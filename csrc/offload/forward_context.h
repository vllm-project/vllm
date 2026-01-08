#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <thread>
// Enumerates possible states of a worker thread
enum class ThreadStatus {
    WORKING,    // Thread is actively processing tasks
    WAITING,    // Thread is idle and waiting for work
    EXIT        // Thread should terminate
};

// Per-thread state information for task distribution
struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;  // Current thread status
    std::unique_ptr<std::atomic<int>> curr;             // Current task index
    int end;                                            // End index (exclusive)
};

// Manages a pool of worker threads for parallel task execution
// Provides work-stealing scheduling and memory buffer management
class ForwardContext {
public:
    // Constructor: creates thread pool with specified maximum threads
    explicit ForwardContext(int max_thread_num);

    // Destructor: stops all threads and frees memory
    ~ForwardContext();

    // Disallow copy and move operations
    ForwardContext(const ForwardContext&) = delete;
    ForwardContext& operator=(const ForwardContext&) = delete;
    ForwardContext(ForwardContext&&) = delete;
    ForwardContext& operator=(ForwardContext&&) = delete;

    // Get or allocate an aligned memory buffer
    void* getBuffer(const std::string& name, size_t size, size_t alignment = 64);

    // Execute tasks using work-stealing scheduling
    void do_work_stealing_job(int task_num, std::function<void(int, int)> compute_func);

private:
    // Process tasks for a specific thread (main loop for each thread)
    void process_tasks(int thread_id);

    // Worker thread entry point
    void worker_thread(int thread_id);

    // Memory buffer cache for reuse
    std::unordered_map<std::string, std::tuple<void*, size_t>> memoryMap;

    // Active number of threads for current job
    int thread_num_ = 0;

    // Maximum number of threads in pool
    int max_thread_num_ = 0;

    // Per-thread state information
    std::vector<ThreadState> thread_state_;

    // Task computation function: f(thread_id, task_id)
    std::function<void(int, int)> compute_func_;

    // Worker thread objects (index 0 is empty - main thread runs directly)
    std::vector<std::thread> workers_;
};