#include "forward_context.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

// Constructor: Initialize thread pool
ForwardContext::ForwardContext(int max_thread_num)
    : max_thread_num_(max_thread_num),
      thread_state_(max_thread_num_),
      workers_(max_thread_num_) {

        // Initialize thread state for all potential threads
        for (int i = 0; i < max_thread_num_; ++i) {
            thread_state_[i].curr = std::make_unique<std::atomic<int>>(0);
            thread_state_[i].status = std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
            thread_state_[i].end = 0;
        }

        // Launch worker threads (thread 0 is reserved for main thread)
        for (int i = 1; i < max_thread_num_; ++i) {
            workers_[i] = std::thread(&ForwardContext::worker_thread, this, i);
        }
}

// Destructor: Cleanup threads and memory
ForwardContext::~ForwardContext() {
    // Signal all threads to exit
    for (int i = 0; i < max_thread_num_; ++i) {
        thread_state_[i].status->store(ThreadStatus::EXIT, std::memory_order_release);
    }

    // Wait for worker threads to finish
    for (int i = 1; i < max_thread_num_; ++i) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }

    // Free all cached memory buffers
    for (auto& entry : memoryMap) {
        void* buffer = std::get<0>(entry.second);
        if (buffer) {
            free(buffer);  // Use free() for aligned_alloc() memory
        }
    }
    memoryMap.clear();
}

// Memory buffer management with LRU-style reuse
void* ForwardContext::getBuffer(const std::string& name, size_t size, size_t alignment) {
    if (name.empty() || size == 0) {
        return nullptr;
    }

    auto it = memoryMap.find(name);
    if (it != memoryMap.end()) {
        if (std::get<1>(it->second) >= size) {
            return std::get<0>(it->second);  // Reuse existing buffer
        } else {
            free(std::get<0>(it->second));   // Free insufficient buffer
        }
    }

    // Allocate new aligned buffer
    void* buffer = std::aligned_alloc(alignment, size);
    if (buffer == nullptr) {
        std::cerr << "Memory allocation failed for buffer: " << name
                  << " size: " << size << std::endl;
        exit(-1);
    }

    memoryMap[name] = std::make_tuple(buffer, size);
    return buffer;
}

// Main entry point for parallel task execution
void ForwardContext::do_work_stealing_job(int task_num, std::function<void(int, int)> compute_func) {
    compute_func_ = compute_func;
    thread_num_ = std::min(max_thread_num_, task_num);

    const int base = task_num / thread_num_;
    const int remain = task_num % thread_num_;

    // Configure main thread's range
    thread_state_[0].end = base + (0 < remain);
    thread_state_[0].curr->store(0, std::memory_order_relaxed);

    // Configure and activate worker threads
    for (int i = 1; i < thread_num_; ++i) {
        thread_state_[i].curr->store(thread_state_[i - 1].end, std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING, std::memory_order_release);
    }

    // Activate main thread last (after workers are ready)
    thread_state_[0].status->store(ThreadStatus::WORKING, std::memory_order_release);

    // Main thread processes its task range (thread_id = 0)
    process_tasks(0);

    // Wait for all worker threads to complete
    for (int i = 1; i < thread_num_; ++i) {
        while (thread_state_[i].status->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
            // Busy-wait for completion (consider condition variables for production)
        }
    }
}

// Process tasks for a specific thread ID
void ForwardContext::process_tasks(int thread_id) {
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        compute_func_(thread_id, task_id);  // Pass thread_id to compute function
    }

    thread_state_[thread_id].status->store(ThreadStatus::WAITING, std::memory_order_release);
}

// Worker thread main loop (for threads 1..n)
void ForwardContext::worker_thread(int thread_id) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        ThreadStatus status = thread_state_[thread_id].status->load(std::memory_order_acquire);

        if (status == ThreadStatus::WORKING) {
            // Process tasks and reset idle timer
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            // Check if we should sleep to reduce CPU usage
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}