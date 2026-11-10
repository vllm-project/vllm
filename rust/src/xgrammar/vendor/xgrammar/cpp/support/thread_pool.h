/*!
 * Copyright (c) 2023 by Contributors
 * \file xgrammar/support/thread_pool.h
 * \brief Thread pool.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_POOL_H_
#define XGRAMMAR_SUPPORT_THREAD_POOL_H_

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief A thread pool implementation for parallel task execution.
 *
 * ThreadPool manages a pool of worker threads that can execute tasks asynchronously.
 * Tasks are submitted to a queue and executed by available threads from the pool.
 * The pool automatically handles thread synchronization and task distribution.
 */
class ThreadPool {
 public:
  /*!
   * \brief Construct a new thread pool with the specified number of threads.
   * \param num_threads Number of worker threads to create. Defaults to hardware concurrency.
   * \note The pool starts the worker threads immediately upon construction.
   */
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
    // Initialize thread pool with num_threads threads
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            // Lock queue while waiting for new task
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });

            // Exit thread if shutdown and queue is empty
            if (shutdown_ && task_queue_.empty()) return;

            // Get task from queue
            task = std::move(task_queue_.front());
            task_queue_.pop();
          }
          try {
            task();
          } catch (...) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!first_exception_) {
              first_exception_ = std::current_exception();
            }
          }
          TaskComplete();
        }
      });
    }
  }

  /*!
   * \brief Add a new task to be executed by the thread pool.
   * \tparam F Type of the function to execute
   * \tparam Args Types of the arguments to pass to the function
   * \param f Function to execute
   * \param args Arguments to pass to the function
   * \return std::shared_future containing the result of the function call
   * \note Tasks are executed in FIFO order but may complete in any order.
   */
  template <class F, class... Args>
  auto Submit(F&& f, Args&&... args) -> std::shared_future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    // Package the task with its arguments into a shared pointer
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::shared_future<return_type> res = task->get_future().share();

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      XGRAMMAR_CHECK(!shutdown_) << "Cannot submit task to stopped ThreadPool";
      ++unfinished_task_count_;  // Increment task count

      // Directly add the task without wrapping
      task_queue_.emplace([task]() { (*task)(); });
    }
    queue_condition_.notify_one();
    return res;
  }

  /*!
   * \brief Add a new task to be executed by the thread pool without returning a future.
   * \tparam F Type of the function to execute
   * \tparam Args Types of the arguments to pass to the function
   * \param f Function to execute
   * \param args Arguments to pass to the function
   * \note Tasks are executed asynchronously by the worker threads.
   */
  template <class F, class... Args>
  void Execute(F&& f, Args&&... args) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      XGRAMMAR_CHECK(!shutdown_) << "Cannot execute task in stopped ThreadPool";
      ++unfinished_task_count_;  // Increment task count

      // Directly add the task without wrapping
      task_queue_.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }
    queue_condition_.notify_one();
  }

  /*!
   * \brief Wait until all submitted tasks have finished.
   * \note If a task submitted with Execute threw, the first such exception is rethrown here on the
   * calling thread. An exception escaping a worker thread would otherwise terminate the process.
   */
  void Wait() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_done_condition_.wait(lock, [this] { return unfinished_task_count_ == 0; });
    }
    RethrowTaskException();
  }

  /*!
   * \brief Join all threads in the pool.
   *
   * Sets shutdown flag and waits for all threads to complete their current tasks
   * before destroying the pool. Any remaining tasks in the queue will be executed
   * before shutdown completes.
   * \note If a task submitted with Execute threw, the first such exception is rethrown here on the
   * calling thread. An exception escaping a worker thread would otherwise terminate the process.
   */
  void Join() {
    Shutdown();
    RethrowTaskException();
  }

  /*!
   * \brief Destructor that ensures graceful shutdown of the thread pool.
   */
  ~ThreadPool() { Shutdown(); }

  // Prevent copying or moving of the thread pool
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

 private:
  void Shutdown() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (shutdown_) return;  // Already shut down
      shutdown_ = true;
    }

    queue_condition_.notify_all();  // Wake up all threads so they can exit
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();  // Wait for thread to finish
    }
  }

  void RethrowTaskException() {
    std::exception_ptr exception;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      std::swap(exception, first_exception_);
    }
    if (exception) {
      std::rethrow_exception(exception);
    }
  }

  void TaskComplete() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    --unfinished_task_count_;
    if (unfinished_task_count_ == 0) {
      tasks_done_condition_.notify_all();  // Notify waiting threads
    }
  }

  /*! \brief Thread container */
  std::vector<std::thread> workers_;
  /*! \brief Task queue */
  std::queue<std::function<void()>> task_queue_;
  /*! \brief Mutex to protect task queue */
  std::mutex queue_mutex_;
  /*! \brief Condition variable for thread synchronization */
  std::condition_variable queue_condition_;
  /*! \brief Condition variable for task completion */
  std::condition_variable tasks_done_condition_;
  /*! \brief Flag to indicate thread pool shutdown */
  bool shutdown_ = false;
  /*! \brief Number of unfinished tasks */
  int unfinished_task_count_ = 0;
  /*! \brief The first exception thrown by a task, rethrown by Wait() or Join() */
  std::exception_ptr first_exception_ = nullptr;
};

inline void ParallelFor(int low, int high, int num_threads, std::function<void(int)> f) {
  if (high - low == 1) {
    f(low);
    return;
  }

  ThreadPool pool(num_threads);

  int total = high - low;
  int chunk_size = (total + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start = low + t * chunk_size;
    int end = std::min(start + chunk_size, high);

    if (start >= end) break;  // No more iterations to process

    pool.Execute([f, start, end]() {
      for (int i = start; i < end; ++i) {
        f(i);
      }
    });
  }
  pool.Join();
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_POOL_H_
