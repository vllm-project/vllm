use std::future::Future;
use std::sync::Mutex;

use tokio::runtime::{Builder, Handle, Runtime};
use tokio::task::JoinHandle;
use tracing::{info, warn};

const REQUEST_WORKER_THREADS_ENV: &str = "VLLM_RS_REQUEST_WORKER_THREADS";
const DEFAULT_MAX_REQUEST_WORKER_THREADS: usize = 64;

/// Tokio runtime used to run heavyweight request paths outside the HTTP
/// runtime.
///
/// The server middleware uses this runtime for inference and tokenization
/// routes so CPU-heavy request preparation does not monopolize the HTTP
/// runtime's worker queue. Dropping the wrapper shuts the runtime down in the
/// background.
pub(crate) struct RequestRuntime {
    runtime: Mutex<Option<Runtime>>,
    handle: Handle,
}

impl RequestRuntime {
    pub(crate) fn new() -> Self {
        let worker_threads = request_worker_threads();
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .thread_name("vllm-request")
            .worker_threads(worker_threads)
            .build()
            .expect("failed to build request runtime");
        let handle = runtime.handle().clone();
        Self {
            runtime: Mutex::new(Some(runtime)),
            handle,
        }
    }

    pub(crate) fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.handle.spawn(future)
    }
}

impl Drop for RequestRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.lock().expect("request runtime mutex poisoned").take() {
            runtime.shutdown_background();
        }
    }
}

/// Get the number of worker threads to use for the request runtime.
///
/// If `VLLM_RS_REQUEST_WORKER_THREADS` is set to a valid positive integer, it is
/// used directly. Otherwise, the runtime uses available parallelism capped by
/// `DEFAULT_MAX_REQUEST_WORKER_THREADS`.
fn request_worker_threads() -> usize {
    if let Some(value) = std::env::var_os(REQUEST_WORKER_THREADS_ENV) {
        match value.to_string_lossy().parse::<usize>() {
            Ok(worker_threads) if worker_threads > 0 => return worker_threads,
            _ => warn!(
                value = %value.to_string_lossy(),
                "ignoring invalid {REQUEST_WORKER_THREADS_ENV}"
            ),
        }
    }

    std::thread::available_parallelism()
        .map(|parallelism| {
            let available = parallelism.get();
            let worker_threads = available.min(DEFAULT_MAX_REQUEST_WORKER_THREADS);
            if worker_threads < available {
                info!(
                    available_parallelism = available,
                    capped_worker_threads = worker_threads,
                    "capping request runtime worker threads, set {REQUEST_WORKER_THREADS_ENV} to override"
                );
            }
            worker_threads
        })
        .unwrap_or(DEFAULT_MAX_REQUEST_WORKER_THREADS)
}
