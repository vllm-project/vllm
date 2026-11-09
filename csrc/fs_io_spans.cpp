// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Batched O_DIRECT reads of many small spans (e.g. embedding rows) from files.
//
// A call reads n spans of `length` bytes at (fd, offset) into one output
// buffer. The spans' aligned blocks are deduplicated and merged into extents
// of up to 64 KiB, read into a persistent aligned staging buffer, and copied
// out. Two backends:
//   io_uring: one ring per worker thread, the staging buffer registered with
//             every ring (READ_FIXED), many reads per io_uring_enter. Raw
//             syscalls, no liburing.
//   pread:    the same extents read with pread(2) by the worker threads.
// A read small enough for one ring runs on the calling thread only, so short
// decode steps do not wake the pool.

#include "fs_io_spans.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <linux/stat.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kMaxExtentBytes = 64 << 10;
constexpr int kMaxThreads = 64;
constexpr int kFdShift = 40;     // block key = fd << 40 | block index
constexpr int kPieceShift = 30;  // io_uring registers buffers of <= 1 GiB
constexpr size_t kMaxPieces = 64;

struct Extent {
  int fd;
  int64_t block;    // first block
  int64_t nblocks;  // consecutive blocks
  int64_t pos;      // byte position in the staging buffer
  int64_t fsize;    // file size (the last block may be partial)
};

// O_DIRECT offset alignment of an open file, 4096 when the kernel cannot say.
int64_t dio_align(int fd) {
#if defined(STATX_DIOALIGN)
  struct statx stx;
  if (syscall(__NR_statx, fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) == 0 &&
      (stx.stx_mask & STATX_DIOALIGN) && stx.stx_dio_offset_align > 0) {
    return stx.stx_dio_offset_align;
  }
#endif
  return 4096;
}

struct Ring {
  int fd = -1;
  unsigned entries = 0;
  unsigned *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
  unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
  io_uring_sqe* sqes = nullptr;
  io_uring_cqe* cqes = nullptr;

  // Returns 0 or -errno (io_uring disabled, e.g. by a seccomp profile).
  int init(unsigned depth) {
    io_uring_params p;
    memset(&p, 0, sizeof p);
    fd = static_cast<int>(syscall(__NR_io_uring_setup, depth, &p));
    if (fd < 0) return -errno;
    const size_t sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    const size_t cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    auto* sq = static_cast<char*>(mmap(nullptr, sq_size, PROT_READ | PROT_WRITE,
                                       MAP_SHARED | MAP_POPULATE, fd,
                                       IORING_OFF_SQ_RING));
    if (sq == MAP_FAILED) return -errno;
    char* cq = sq;
    if (!(p.features & IORING_FEAT_SINGLE_MMAP)) {
      cq = static_cast<char*>(mmap(nullptr, cq_size, PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_POPULATE, fd,
                                   IORING_OFF_CQ_RING));
      if (cq == MAP_FAILED) return -errno;
    }
    void* s = mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe),
                   PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                   IORING_OFF_SQES);
    if (s == MAP_FAILED) return -errno;
    sqes = static_cast<io_uring_sqe*>(s);
    sq_tail = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
    sq_mask = reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
    sq_array = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
    cq_head = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
    cq_tail = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
    cq_mask = reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
    cqes = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
    entries = p.sq_entries;
    return 0;
  }

  // Reads extents [e0, e1). Returns the number of failed or short reads.
  int64_t read(const Extent* ext, int64_t e0, int64_t e1, uint8_t* stage,
               bool fixed, int64_t blk) {
    int64_t next = e0, done = 0, inflight = 0, unsubmitted = 0, errors = 0;
    const int64_t total = e1 - e0;
    while (done < total) {
      unsigned tail = *sq_tail;
      while (next < e1 && inflight < entries) {
        const Extent& e = ext[next];
        const unsigned idx = tail & *sq_mask;
        io_uring_sqe* sqe = &sqes[idx];
        memset(sqe, 0, sizeof *sqe);
        sqe->opcode = fixed ? IORING_OP_READ_FIXED : IORING_OP_READ;
        sqe->fd = e.fd;
        sqe->addr = reinterpret_cast<uint64_t>(stage + e.pos);
        sqe->len = static_cast<uint32_t>(e.nblocks * blk);
        sqe->off = static_cast<uint64_t>(e.block * blk);
        sqe->buf_index =
            fixed ? static_cast<uint16_t>(e.pos >> kPieceShift) : 0;
        sqe->user_data = static_cast<uint64_t>(next);
        sq_array[idx] = idx;
        tail++, next++, inflight++, unsubmitted++;
      }
      __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE);
      const int got = static_cast<int>(
          syscall(__NR_io_uring_enter, fd, static_cast<unsigned>(unsubmitted),
                  1, IORING_ENTER_GETEVENTS, nullptr, 0));
      if (got < 0) {
        if (errno == EINTR) continue;
        return errors + (total - done);
      }
      unsubmitted -= got;
      unsigned head = *cq_head;
      while (head != __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) {
        const io_uring_cqe& c = cqes[head & *cq_mask];
        const Extent& e = ext[c.user_data];
        const int64_t off = e.block * blk;
        const int64_t want = std::min(e.nblocks * blk, e.fsize - off);
        if (c.res < 0 || c.res != want) errors++;
        head++, done++, inflight--;
      }
      __atomic_store_n(cq_head, head, __ATOMIC_RELEASE);
    }
    return errors;
  }
};

int64_t pread_extents(const Extent* ext, int64_t e0, int64_t e1, uint8_t* stage,
                      int64_t blk) {
  int64_t errors = 0;
  for (int64_t i = e0; i < e1; i++) {
    const Extent& e = ext[i];
    const int64_t off = e.block * blk;
    const int64_t want = std::min(e.nblocks * blk, e.fsize - off);
    ssize_t got;
    do {
      got =
          pread(e.fd, stage + e.pos, static_cast<size_t>(e.nblocks * blk), off);
    } while (got < 0 && errno == EINTR);
    if (got != want) errors++;
  }
  return errors;
}

class SpanReader {
 public:
  // Returns 0, or -errno when neither backend can be set up.
  int init(int threads, int queue_depth, bool use_io_uring) {
    threads_ = threads;
    queue_depth_ = queue_depth;
    if (use_io_uring) {
      unsigned depth = 1;
      while (depth < static_cast<unsigned>(queue_depth)) depth <<= 1;
      rings_.resize(threads);
      int rc = 0;
      for (int t = 0; t < threads && rc == 0; t++) rc = rings_[t].init(depth);
      if (rc != 0) rings_.clear();  // fall back to pread
    }
    for (int t = 1; t < threads; t++) {
      std::thread([this, t] { worker(t); }).detach();
    }
    return grow_stage(static_cast<size_t>(64) << 20);
  }

  bool uses_io_uring() const { return !rings_.empty(); }

  // Returns 0, -errno on setup errors, or the number of failed reads.
  int64_t read(const int64_t* fds, const int64_t* offsets, int64_t n,
               int64_t length, uint8_t* out) {
    std::lock_guard<std::mutex> guard(read_mu_);
    if (n == 0) return 0;
    int64_t blk = 512;
    std::unordered_map<int, int64_t> fsize;
    for (int64_t i = 0; i < n; i++) {
      const int fd = static_cast<int>(fds[i]);
      if (fsize.count(fd)) continue;
      struct stat st;
      if (fstat(fd, &st) != 0) return -errno;
      fsize[fd] = st.st_size;
      blk = std::max(blk, align_of(fd));
    }
    // 1. distinct (file, block) keys of every span, sorted. A span past the
    //    end of its file is a short read (its last block would read short
    //    without an error, leaving the tail of the span undefined).
    std::vector<int64_t> keys;
    keys.reserve(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; i++) {
      if (offsets[i] + length > fsize[static_cast<int>(fds[i])]) return 1;
      const int64_t b0 = offsets[i] / blk;
      const int64_t b1 = (offsets[i] + length - 1) / blk;
      for (int64_t b = b0; b <= b1; b++)
        keys.push_back((fds[i] << kFdShift) | b);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    // 2. extents of consecutive blocks; the k-th key lands at k * blk, so an
    //    extent never straddles a registered 1 GiB piece (blk divides 1 GiB)
    std::vector<Extent> ext;
    const int64_t max_blocks = std::max<int64_t>(1, kMaxExtentBytes / blk);
    for (size_t k = 0; k < keys.size(); k++) {
      const int64_t pos = static_cast<int64_t>(k) * blk;
      if (!ext.empty() && keys[k] == keys[k - 1] + 1 &&
          ext.back().nblocks < max_blocks &&
          (ext.back().pos >> kPieceShift) == ((pos + blk - 1) >> kPieceShift)) {
        ext.back().nblocks++;
        continue;
      }
      const int fd = static_cast<int>(keys[k] >> kFdShift);
      ext.push_back(
          {fd, keys[k] & ((int64_t{1} << kFdShift) - 1), 1, pos, fsize[fd]});
    }
    const int rc = grow_stage(keys.size() * static_cast<size_t>(blk));
    if (rc != 0) return rc;
    // 3. read the extents: one thread if one ring's queue holds them all
    const int64_t ne = static_cast<int64_t>(ext.size());
    int nt = static_cast<int>(
        std::min<int64_t>(threads_, (ne + queue_depth_ - 1) / queue_depth_));
    nt = std::max(nt, 1);
    std::atomic<int64_t> errors{0};
    run(nt, [&](int t) {
      const int64_t e0 = ne * t / nt, e1 = ne * (t + 1) / nt;
      if (e1 <= e0) return;
      errors +=
          uses_io_uring()
              ? rings_[t].read(ext.data(), e0, e1, stage_, registered_, blk)
              : pread_extents(ext.data(), e0, e1, stage_, blk);
    });
    if (errors.load() != 0) return errors.load();
    // 4. copy each span out of staging
    const int ct = n >= 4096 ? nt : 1;
    run(ct, [&](int t) {
      const int64_t i0 = n * t / ct, i1 = n * (t + 1) / ct;
      for (int64_t i = i0; i < i1; i++) {
        const int64_t key = (fds[i] << kFdShift) | (offsets[i] / blk);
        const int64_t k =
            std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
        memcpy(out + i * length, stage_ + k * blk + offsets[i] % blk,
               static_cast<size_t>(length));
      }
    });
    return 0;
  }

 private:
  int64_t align_of(int fd) {
    auto it = align_.find(fd);
    if (it == align_.end()) it = align_.emplace(fd, dio_align(fd)).first;
    return it->second;
  }

  // Grows the staging buffer and registers it with every ring.
  int grow_stage(size_t need) {
    if (need <= stage_size_) return 0;
    size_t size = stage_size_ ? stage_size_ * 2 : static_cast<size_t>(64) << 20;
    while (size < need) size *= 2;
    if (registered_) {
      for (auto& r : rings_) {
        syscall(__NR_io_uring_register, r.fd, IORING_UNREGISTER_BUFFERS,
                nullptr, 0);
      }
    }
    registered_ = false;
    free(stage_);
    stage_ = nullptr;
    stage_size_ = 0;
    void* p = nullptr;
    if (posix_memalign(&p, 4096, size) != 0) return -ENOMEM;
    stage_ = static_cast<uint8_t*>(p);
    memset(stage_, 0, size);  // fault the pages in once
    stage_size_ = size;
    const size_t pieces =
        (size + (size_t{1} << kPieceShift) - 1) >> kPieceShift;
    if (!rings_.empty() && pieces <= kMaxPieces) {
      std::vector<iovec> iov(pieces);
      for (size_t i = 0; i < pieces; i++) {
        const size_t off = i << kPieceShift;
        iov[i] = {stage_ + off, std::min(size - off, size_t{1} << kPieceShift)};
      }
      // Registration can fail (RLIMIT_MEMLOCK); plain reads still work.
      registered_ = true;
      for (auto& r : rings_) {
        if (syscall(__NR_io_uring_register, r.fd, IORING_REGISTER_BUFFERS,
                    iov.data(), static_cast<unsigned>(pieces)) < 0) {
          registered_ = false;
        }
      }
      if (!registered_) {
        for (auto& r : rings_) {
          syscall(__NR_io_uring_register, r.fd, IORING_UNREGISTER_BUFFERS,
                  nullptr, 0);
        }
      }
    }
    return 0;
  }

  // Runs fn(0..nt-1): fn(0) on the calling thread, the rest on the pool.
  void run(int nt, const std::function<void(int)>& fn) {
    if (nt <= 1) {
      fn(0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(pool_mu_);
      job_ = &fn;
      job_threads_ = nt;
      pending_ = nt - 1;
      generation_++;
    }
    pool_cv_.notify_all();
    fn(0);
    std::unique_lock<std::mutex> lock(pool_mu_);
    done_cv_.wait(lock, [&] { return pending_ == 0; });
  }

  void worker(int t) {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(pool_mu_);
    for (;;) {
      pool_cv_.wait(lock, [&] { return generation_ != seen; });
      seen = generation_;
      if (t >= job_threads_) continue;
      const std::function<void(int)>* fn = job_;
      lock.unlock();
      (*fn)(t);
      lock.lock();
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }

  int threads_ = 1, queue_depth_ = 16;
  std::vector<Ring> rings_;
  uint8_t* stage_ = nullptr;
  size_t stage_size_ = 0;
  bool registered_ = false;
  std::unordered_map<int, int64_t> align_;
  std::mutex read_mu_;
  std::mutex pool_mu_;
  std::condition_variable pool_cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  int job_threads_ = 0, pending_ = 0;
  uint64_t generation_ = 0;
};

// One reader per process, never destroyed: its pool threads live as long.
SpanReader* g_reader = nullptr;
std::mutex g_init_mu;

const char* backend_name() {
  return g_reader->uses_io_uring() ? "io_uring" : "pread";
}

}  // namespace

extern "C" {

PyObject* read_spans_init(PyObject* /*self*/, PyObject* args) {
  int threads = 16, queue_depth = 16, use_io_uring = 1;
  if (!PyArg_ParseTuple(args, "|iip", &threads, &queue_depth, &use_io_uring)) {
    return nullptr;
  }
  if (threads < 1 || threads > kMaxThreads || queue_depth < 1 ||
      queue_depth > 4096) {
    PyErr_SetString(PyExc_ValueError,
                    "threads must be in [1, 64] and queue_depth in [1, 4096]");
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(g_init_mu);
  if (g_reader == nullptr) {
    auto* reader = new SpanReader();
    const int rc = reader->init(threads, queue_depth, use_io_uring != 0);
    if (rc != 0) {
      delete reader;  // no pool thread was started without a staging buffer
      errno = -rc;
      return PyErr_SetFromErrno(PyExc_OSError);
    }
    g_reader = reader;
  }
  return PyUnicode_FromString(backend_name());
}

PyObject* batch_read_spans(PyObject* /*self*/, PyObject* args) {
  PyObject *fds_obj, *offsets_obj, *out_obj;
  Py_ssize_t length;
  if (!PyArg_ParseTuple(args, "OOnO", &fds_obj, &offsets_obj, &length,
                        &out_obj)) {
    return nullptr;
  }
  if (g_reader == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "call read_spans_init() first");
    return nullptr;
  }
  Py_buffer fds, offsets, out;
  if (PyObject_GetBuffer(fds_obj, &fds, PyBUF_C_CONTIGUOUS) != 0) {
    return nullptr;
  }
  if (PyObject_GetBuffer(offsets_obj, &offsets, PyBUF_C_CONTIGUOUS) != 0) {
    PyBuffer_Release(&fds);
    return nullptr;
  }
  if (PyObject_GetBuffer(out_obj, &out, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) !=
      0) {
    PyBuffer_Release(&fds);
    PyBuffer_Release(&offsets);
    return nullptr;
  }
  const Py_ssize_t n = offsets.len / static_cast<Py_ssize_t>(sizeof(int64_t));
  int64_t result = 0;
  const char* error = nullptr;
  if (fds.len != offsets.len || offsets.len % sizeof(int64_t) != 0) {
    error = "fds and offsets must be int64 buffers of the same length";
  } else if (length <= 0 || length > kMaxExtentBytes) {
    error = "length must be in [1, 65536]";
  } else if (out.len < n * length) {
    error = "out is smaller than len(offsets) * length";
  } else {
    const auto* f = static_cast<const int64_t*>(fds.buf);
    const auto* o = static_cast<const int64_t*>(offsets.buf);
    for (Py_ssize_t i = 0; i < n && error == nullptr; i++) {
      if (f[i] < 0 || o[i] < 0) error = "negative fd or offset";
    }
    if (error == nullptr) {
      Py_BEGIN_ALLOW_THREADS result =
          g_reader->read(f, o, n, length, static_cast<uint8_t*>(out.buf));
      Py_END_ALLOW_THREADS
    }
  }
  PyBuffer_Release(&fds);
  PyBuffer_Release(&offsets);
  PyBuffer_Release(&out);
  if (error != nullptr) {
    PyErr_SetString(PyExc_ValueError, error);
    return nullptr;
  }
  if (result < 0) {
    errno = static_cast<int>(-result);
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  if (result > 0) {
    PyErr_Format(PyExc_OSError, "%lld reads failed or were short",
                 static_cast<long long>(result));
    return nullptr;
  }
  Py_RETURN_NONE;
}

}  // extern "C"
