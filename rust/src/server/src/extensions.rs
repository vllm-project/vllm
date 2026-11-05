// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Router;
use vllm_llm::{RequestObserver, RequestObservers};

pub(crate) type RouterExtension = Box<dyn FnOnce(Router) -> Router + Send>;

/// Hooks for embedding `vllm-server` in a custom binary, passed to
/// [`serve_with_extensions`](crate::serve_with_extensions). Not a stable API.
#[derive(Default)]
pub struct ServeExtensions {
    router: Option<RouterExtension>,
    request_observers: Vec<Arc<dyn RequestObserver>>,
}

impl ServeExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge additional routes into the finalized vLLM router.
    pub fn router(mut self, extend: impl FnOnce(Router) -> Router + Send + 'static) -> Self {
        self.router = Some(Box::new(extend));
        self
    }

    /// Register an observer, run in order after Prometheus recording.
    pub fn request_observer(mut self, observer: impl RequestObserver) -> Self {
        self.request_observers.push(Arc::new(observer));
        self
    }

    /// The router extension is `Send` but not `Sync`, so it is split off to
    /// avoid borrowing it across startup `.await`s.
    pub(crate) fn into_parts(self) -> (RouterExtension, StateExtensions) {
        let router = self.router.unwrap_or_else(|| Box::new(|router| router));
        let state = StateExtensions {
            request_observers: RequestObservers::new(self.request_observers),
        };
        (router, state)
    }
}

#[derive(Default)]
pub(crate) struct StateExtensions {
    pub(crate) request_observers: RequestObservers,
}
