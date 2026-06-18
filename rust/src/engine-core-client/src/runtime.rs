use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::runtime::Runtime;

const ZMQ_WORKER_THREADS_ENV: &str = "VLLM_ZMQ_WORKER_THREADS";
/// The number of tasks running on the ZMQ runtime is fixed and expected to remain
/// small, and multiple engines share the same ZMQ socket. Therefore, based on
/// benchmarks, a default value of 4 is generally sufficient.
const DEFAULT_ZMQ_WORKER_THREADS: usize = 4;

static ZMQ_RUNTIME_SEQUENCE: OnceLock<AtomicUsize> = OnceLock::new();

/// Build a Tokio runtime for ZMQ tasks. Multiple calls to this function will
/// return multiple runtimes with distinct thread name suffixes.
pub(crate) fn build_zmq_runtime() -> Runtime {
    let sequence = ZMQ_RUNTIME_SEQUENCE
        .get_or_init(|| AtomicUsize::new(0))
        .fetch_add(1, Ordering::Relaxed);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(zmq_worker_threads())
        .thread_name_fn(move || format!("vllm-zmq-{sequence}"))
        .enable_all()
        .build()
        .expect("failed to build vLLM ZMQ runtime")
}

/// Get the number of worker threads to use for the ZMQ runtime. If env var
/// `VLLM_ZMQ_WORKER_THREADS` is set and a valid positive integer, it will be used.
/// Otherwise, the default value of `DEFAULT_ZMQ_WORKER_THREADS` will be used.
fn zmq_worker_threads() -> usize {
    std::env::var(ZMQ_WORKER_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_ZMQ_WORKER_THREADS)
}
