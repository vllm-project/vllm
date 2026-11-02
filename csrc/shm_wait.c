/* SPDX-License-Identifier: Apache-2.0
 * Experimental SPMC shared-memory control plane. Linux x86_64/aarch64 only.
 * Payloads live in vLLM's existing ring; this file owns their publication order.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAGIC 0x41535131u
#define ABI 1u
#define HEADER_BYTES 128u
#define MAX_READERS 1024u
typedef struct {
    _Atomic uint32_t magic;
    uint32_t abi, readers, capacity, stride;
    _Atomic uint64_t published;
    _Atomic uint32_t data_event, reader_waiters, writer_waiting, writer_closed;
} Header;
typedef struct {
    _Atomic uint64_t consumed;
    _Atomic uint32_t event, closed;
} Reader;
_Static_assert(sizeof(Header) <= HEADER_BYTES, "header overflow");
_Static_assert(sizeof(Reader) == 16, "unexpected atomic ABI");
_Static_assert(sizeof(_Atomic uint32_t) == 4, "futex word must be 32 bits");
typedef struct {
    PyObject_HEAD
    Py_buffer view;
    Header *h;
    int rank, acquired, busy;
    uint64_t seq;
} Endpoint;

static Reader *reader(Header *h, unsigned i) {
    return (Reader *)((char *)h + HEADER_BYTES + i * h->stride);
}
static int wake_word(_Atomic uint32_t *word, int count) {
    return syscall(SYS_futex, word, FUTEX_WAKE, count, NULL, NULL, 0) < 0 ? errno : 0;
}
static int park(_Atomic uint32_t *word, uint32_t expected,
                const struct timespec *deadline) {
    if (syscall(SYS_futex, word, FUTEX_WAIT_BITSET, expected, deadline,
                NULL, FUTEX_BITSET_MATCH_ANY) == 0) return 0;
    return errno == EAGAIN ? 0 : errno;
}
static uint64_t monotonic_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void spin_hint(void) {
#if defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
#error Unsupported architecture: qualify native shared atomics first
#endif
}

/* Return 1 if ready, 0 if blocked, or negative errno. This is the only
 * predicate implementation; both the fast path and registered slow path use it.
 */
static int readiness(Endpoint *e, unsigned *lagging) {
    Header *h = e->h;
    if (e->rank < 0) {
        if (atomic_load(&h->writer_closed)) return -ECANCELED;
        if (e->seq == UINT64_MAX) return -EOVERFLOW;
        int blocked = 0;
        for (unsigned i = 0; i < h->readers; ++i) {
            Reader *r = reader(h, i);
            if (atomic_load(&r->closed)) return -EPIPE;
            uint64_t done = atomic_load(&r->consumed);
            if (done > e->seq) return -EPROTO;
            if (!blocked && e->seq - done >= h->capacity) { *lagging = i; blocked = 1; }
        }
        /* Inspect every peer's closure, even if an earlier peer is lagging.
         * Otherwise a dead later peer could leave the writer asleep forever. */
        return !blocked;
    }
    Reader *r = reader(h, (unsigned)e->rank);
    if (atomic_load(&r->closed)) return -ECANCELED;
    uint64_t published = atomic_load(&h->published);
    if (published < e->seq || published - e->seq > h->capacity) return -EPROTO;
    if (published != e->seq) return 1; /* drain committed messages after close */
    return atomic_load(&h->writer_closed) ? -EPIPE : 0;
}

static int await_ready(Endpoint *e, const struct timespec *deadline,
                       uint64_t spin_until) {
    Header *h = e->h;
    unsigned lagging = 0;
    for (;;) {
        int ready = readiness(e, &lagging);
        if (ready) return ready > 0 ? 0 : -ready;
        uint64_t now = monotonic_ns();
        if (deadline && now >= (uint64_t)deadline->tv_sec * 1000000000ull
                                  + (uint64_t)deadline->tv_nsec) return ETIMEDOUT;
        if (now < spin_until) { spin_hint(); continue; }

        _Atomic uint32_t *waiters = e->rank < 0 ? &h->writer_waiting : &h->reader_waiters;
        _Atomic uint32_t *event = e->rank < 0 ? &reader(h, lagging)->event : &h->data_event;
        /* SC registration before final predicate check, paired with SC
         * publication before reading waiters. Neither side can miss both.
         * Snapshot the doorbell BEFORE recheck: cancellation also increments it.
         */
        atomic_fetch_add(waiters, 1);
        uint32_t expected = atomic_load(event);
        unsigned rechecked_lagging = lagging;
        ready = readiness(e, &rechecked_lagging);
        int error = 0;
        if (!ready && (e->rank >= 0 || rechecked_lagging == lagging))
            error = park(event, expected, deadline);
        atomic_fetch_sub(waiters, 1);
        if (ready) return ready > 0 ? 0 : -ready;
        if (error) return error;
    }
}

static PyObject *failure(int error) {
    if (error == ETIMEDOUT) PyErr_SetString(PyExc_TimeoutError, "queue deadline expired");
    else if (error == ECANCELED) PyErr_SetString(PyExc_RuntimeError, "cancelled");
    else if (error == EPIPE) PyErr_SetString(PyExc_BrokenPipeError, "queue peer closed");
    else if (error == EOVERFLOW) PyErr_SetString(PyExc_OverflowError, "queue sequence exhausted");
    else { errno = error; PyErr_SetFromErrno(PyExc_OSError); }
    return NULL;
}

static int endpoint_init(Endpoint *e, PyObject *args, PyObject *kwargs) {
    PyObject *buffer;
    int rank, create = 0;
    unsigned readers, capacity, stride = 128;
    static char *names[] = {"buffer", "readers", "capacity", "rank", "create", "stride", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OIIi|pI", names,
                                    &buffer, &readers, &capacity, &rank, &create, &stride)) return -1;
    if (e->h) { PyErr_SetString(PyExc_RuntimeError, "endpoint already initialized"); return -1; }
    if (!readers || readers > MAX_READERS || !capacity || capacity > INT_MAX ||
        rank < -1 || rank >= (int)readers || (stride != 16 && stride != 128) ||
        (create && rank != -1)) {
        PyErr_SetString(PyExc_ValueError, "invalid endpoint geometry/ownership"); return -1;
    }
    if (PyObject_GetBuffer(buffer, &e->view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) return -1;
    Header *h = e->view.buf;
    if ((uintptr_t)h % HEADER_BYTES || e->view.len < HEADER_BYTES + readers * stride) {
        PyErr_SetString(PyExc_ValueError, "control buffer alignment/size mismatch"); goto bad;
    }
    if (create) {
        memset(h, 0, HEADER_BYTES + readers * stride);
        atomic_init(&h->magic, 0);
        h->abi = ABI; h->readers = readers; h->capacity = capacity; h->stride = stride;
        atomic_init(&h->published, 0); atomic_init(&h->data_event, 0);
        atomic_init(&h->reader_waiters, 0); atomic_init(&h->writer_waiting, 0);
        atomic_init(&h->writer_closed, 0);
        for (unsigned i = 0; i < readers; ++i) {
            Reader *r = reader(h, i);
            atomic_init(&r->consumed, 0); atomic_init(&r->event, 0); atomic_init(&r->closed, 0);
        }
        atomic_store(&h->magic, MAGIC);
    }
    if (atomic_load(&h->magic) != MAGIC || h->abi != ABI || h->readers != readers ||
        h->capacity != capacity || h->stride != stride) {
        PyErr_SetString(PyExc_ValueError, "control protocol/geometry mismatch"); goto bad;
    }
    if (!atomic_is_lock_free(&h->published) || !atomic_is_lock_free(&h->data_event) ||
        !atomic_is_lock_free(&reader(h, 0)->consumed)) {
        PyErr_SetString(PyExc_RuntimeError, "native process-shared atomics unavailable"); goto bad;
    }
    e->h = h; e->rank = rank;
    e->seq = rank < 0 ? atomic_load(&h->published) : atomic_load(&reader(h, rank)->consumed);
    return 0;
bad:
    PyBuffer_Release(&e->view);
    memset(&e->view, 0, sizeof(e->view));
    return -1;
}

static void endpoint_dealloc(Endpoint *e) {
    PyTypeObject *type = Py_TYPE(e);
    freefunc free_object = (freefunc)PyType_GetSlot(type, Py_tp_free);
    if (e->view.obj) PyBuffer_Release(&e->view);
    free_object(e);
    Py_DECREF(type);
}
static PyObject *endpoint_acquire(Endpoint *e, PyObject *args, PyObject *kwargs) {
    PyObject *timeout = Py_None;
    unsigned long long spin_ns = 0;
    static char *names[] = {"timeout", "spin_ns", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OK", names, &timeout, &spin_ns)) return NULL;
    if (!e->h || e->busy || e->acquired) {
        PyErr_SetString(PyExc_RuntimeError, "uninitialized or concurrent/reentrant endpoint"); return NULL;
    }
    if (spin_ns > 1000000) {
        PyErr_SetString(PyExc_ValueError, "spin_ns must be <= 1000000"); return NULL;
    }
    uint64_t now = monotonic_ns();
    struct timespec deadline, *bound = NULL;
    if (timeout != Py_None) {
        double seconds = PyFloat_AsDouble(timeout);
        if (PyErr_Occurred()) return NULL;
        if (!isfinite(seconds) || seconds < 0 || seconds > 1000000000.0) {
            PyErr_SetString(PyExc_ValueError, "timeout must be finite and nonnegative"); return NULL;
        }
        uint64_t ns = now + (uint64_t)(seconds * 1e9);
        deadline.tv_sec = ns / 1000000000ull; deadline.tv_nsec = ns % 1000000000ull;
        bound = &deadline;
    }
    e->busy = 1;
    int error;
    do {
        Py_BEGIN_ALLOW_THREADS
        error = await_ready(e, bound, now + spin_ns);
        Py_END_ALLOW_THREADS
        if (error == EINTR && PyErr_CheckSignals() < 0) { e->busy = 0; return NULL; }
    } while (error == EINTR);
    e->busy = 0;
    if (error) return failure(error);
    e->acquired = 1;
    return PyLong_FromUnsignedLongLong(e->seq % e->h->capacity);
}
static PyObject *endpoint_release(Endpoint *e, PyObject *args) {
    int commit = 1;
    if (!PyArg_ParseTuple(args, "|p", &commit)) return NULL;
    if (!e->h || !e->acquired || e->busy) {
        PyErr_SetString(PyExc_RuntimeError, "release without acquired slot"); return NULL;
    }
    Header *h = e->h;
    e->acquired = 0;
    int error = 0;
    if (e->rank < 0) {
        if (!commit) Py_RETURN_NONE;
        if (atomic_load(&h->writer_closed)) return failure(ECANCELED);
        atomic_store(&h->published, ++e->seq);
        atomic_fetch_add(&h->data_event, 1);
        if (atomic_load(&h->reader_waiters)) error = wake_word(&h->data_event, INT_MAX);
    } else {
        Reader *r = reader(h, e->rank);
        atomic_store(&r->consumed, ++e->seq);
        atomic_fetch_add(&r->event, 1);
        if (atomic_load(&h->writer_waiting)) error = wake_word(&r->event, 1);
    }
    if (error) return failure(error);
    Py_RETURN_NONE;
}
static PyObject *endpoint_cancel(Endpoint *e, PyObject *unused) {
    if (!e->h) { PyErr_SetString(PyExc_RuntimeError, "uninitialized endpoint"); return NULL; }
    Header *h = e->h;
    if (e->rank < 0) atomic_store(&h->writer_closed, 1);
    else atomic_store(&reader(h, e->rank)->closed, 1);
    /* Always advance doorbells, even with no registered waiter yet. */
    atomic_fetch_add(&h->data_event, 1);
    int error = wake_word(&h->data_event, INT_MAX);
    for (unsigned i = 0; i < h->readers; ++i) {
        Reader *r = reader(h, i);
        atomic_fetch_add(&r->event, 1);
        int current = wake_word(&r->event, INT_MAX);
        if (current) error = current;
    }
    if (error) return failure(error);
    Py_RETURN_NONE;
}
static PyObject *endpoint_snapshot(Endpoint *e, PyObject *unused) {
    if (!e->h) { PyErr_SetString(PyExc_RuntimeError, "uninitialized endpoint"); return NULL; }
    return Py_BuildValue("{sK,sI,sI,sI,sI}", "published", (unsigned long long)atomic_load(&e->h->published),
        "reader_waiters", atomic_load(&e->h->reader_waiters),
        "writer_waiting", atomic_load(&e->h->writer_waiting),
        "writer_closed", atomic_load(&e->h->writer_closed), "abi", ABI);
}
static PyMethodDef methods[] = {
    {"acquire", (PyCFunction)(void(*)(void))endpoint_acquire, METH_VARARGS|METH_KEYWORDS, "Reserve next slot; wait without polling."},
    {"release", (PyCFunction)endpoint_release, METH_VARARGS, "Publish/acknowledge; false aborts writer publication."},
    {"cancel", (PyCFunction)endpoint_cancel, METH_NOARGS, "Close local participant and wake affected peers."},
    {"snapshot", (PyCFunction)endpoint_snapshot, METH_NOARGS, "Diagnostic only; no hot-path sampling."},
    {NULL, NULL, 0, NULL}
};
static PyType_Slot endpoint_slots[] = {
    {Py_tp_new, (void *)PyType_GenericNew},
    {Py_tp_init, (void *)endpoint_init},
    {Py_tp_dealloc, (void *)endpoint_dealloc},
    {Py_tp_methods, methods},
    {0, NULL}
};
static PyType_Spec endpoint_spec = {
    .name = "vllm._shm_wait.Endpoint", .basicsize = sizeof(Endpoint),
    .flags = Py_TPFLAGS_DEFAULT, .slots = endpoint_slots
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "vllm._shm_wait", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__shm_wait(void) {
    PyObject *endpoint_type = PyType_FromSpec(&endpoint_spec);
    if (!endpoint_type) return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m) { Py_DECREF(endpoint_type); return NULL; }
    if (PyModule_AddObject(m, "Endpoint", endpoint_type) < 0) { Py_DECREF(endpoint_type); Py_DECREF(m); return NULL; }
    return m;
}
