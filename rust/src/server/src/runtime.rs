use tokio::runtime::Builder;
use tracing::{info, warn};
use vllm_engine_core_client::runtime::BackgroundShutdownRuntime;

const REQUEST_WORKER_THREADS_ENV: &str = "VLLM_RS_REQUEST_WORKER_THREADS";
const DEFAULT_MAX_REQUEST_WORKER_THREADS: usize = 32;

/// Build a Tokio runtime for heavyweight request paths outside the HTTP runtime.
///
/// The server middleware uses this runtime for inference and tokenization
/// routes so CPU-heavy request preparation does not monopolize the HTTP
/// runtime's worker queue. Dropping the wrapper shuts the runtime down in the
/// background.
pub(crate) fn build_request_runtime() -> BackgroundShutdownRuntime {
    Builder::new_multi_thread()
        .enable_all()
        .thread_name("vllm-request")
        .worker_threads(request_worker_threads())
        .build()
        .expect("failed to build request runtime")
        .into()
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
