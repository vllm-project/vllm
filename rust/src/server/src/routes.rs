mod cache;
mod collective_rpc;
mod health;
mod inference;
mod load;
mod metrics;
pub(crate) mod openai;
mod sleep;
mod version;

use std::sync::Arc;

use axum::Router;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{MethodRouter, get, post};
use itertools::Itertools as _;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::middleware;
use crate::state::AppState;

fn server_dev_mode_enabled() -> bool {
    std::env::var("VLLM_SERVER_DEV_MODE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .is_some_and(|value| value != 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteInfo {
    path: &'static str,
    methods: &'static [&'static str],
}

pub(crate) struct WrapperRouter<S = ()> {
    pub(crate) router: Router<S>,
    routes: Vec<RouteInfo>,
}

impl<S> WrapperRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    fn new() -> Self {
        Self {
            router: Router::new(),
            routes: Vec::new(),
        }
    }

    fn route(
        mut self,
        path: &'static str,
        method_router: MethodRouter<S>,
        methods: &'static [&'static str],
    ) -> Self {
        self.router = self.router.route(path, method_router);
        self.routes.push(RouteInfo { path, methods });
        self
    }
}

const GET_HEAD: &[&str] = &["GET", "HEAD"];
const POST: &[&str] = &["POST"];

pub(crate) fn log_available_routes(app: &WrapperRouter) {
    info!("Available routes are:");
    for route in &app.routes {
        info!(
            "Route: {}, Methods: {}",
            route.path,
            route.methods.iter().format(", ")
        );
    }
}

/// Build the minimal OpenAI-compatible router for one configured model.
#[allow(dead_code)]
pub(crate) fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_dev_mode_and_routes(state, server_dev_mode_enabled()).router
}

#[cfg(test)]
fn build_router_with_dev_mode(state: Arc<AppState>, dev_mode_enabled: bool) -> Router {
    build_router_with_dev_mode_and_routes(state, dev_mode_enabled).router
}

pub(crate) fn build_router_with_routes(state: Arc<AppState>) -> WrapperRouter {
    build_router_with_dev_mode_and_routes(state, server_dev_mode_enabled())
}

fn build_route_registry(dev_mode_enabled: bool) -> WrapperRouter<Arc<AppState>> {
    let mut tracked = WrapperRouter::<Arc<AppState>>::new()
        // Health & monitoring
        .route("/health", get(health::health), GET_HEAD)
        .route("/metrics", get(metrics::scrape), GET_HEAD)
        .route("/load", get(load::load), GET_HEAD)
        .route("/version", get(version::version), GET_HEAD)
        // OpenAI-compatible endpoints
        .route("/v1/models", get(openai::list_models), GET_HEAD)
        .route("/v1/completions", post(openai::completions), POST)
        .route("/v1/chat/completions", post(openai::chat_completions), POST)
        // vLLM specific inference endpoints
        .route("/inference/v1/generate", post(inference::generate), POST);

    if dev_mode_enabled {
        // Development-only
        tracked = tracked
            .route("/reset_prefix_cache", post(cache::reset_prefix_cache), POST)
            .route("/reset_mm_cache", post(cache::reset_mm_cache), POST)
            .route(
                "/reset_encoder_cache",
                post(cache::reset_encoder_cache),
                POST,
            )
            .route(
                "/collective_rpc",
                post(collective_rpc::collective_rpc),
                POST,
            )
            .route("/sleep", post(sleep::sleep), POST)
            .route("/wake_up", post(sleep::wake_up), POST)
            .route("/is_sleeping", get(sleep::is_sleeping), GET_HEAD)
    }
    tracked
}

fn build_router_with_dev_mode_and_routes(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
) -> WrapperRouter {
    let tracked = build_route_registry(dev_mode_enabled);
    let WrapperRouter { router, routes } = tracked;
    let router = router
        .with_state(state.clone())
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http());

    WrapperRouter { router, routes }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod http_client_tests;
