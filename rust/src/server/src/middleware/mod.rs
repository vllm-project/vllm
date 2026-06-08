// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod auth;
mod cors;
mod load;
mod metrics;
mod offload;
mod request_id;

use axum::extract::{MatchedPath, NestedPath};
use axum::http::Extensions;

pub use auth::authenticate_api_key;
pub use cors::{cors_layer, strip_cors_on_no_origin};
pub use load::track_server_load;
pub use metrics::track_http_metrics;
pub(crate) use offload::request_runtime_layer;
pub use request_id::set_request_id_header;

/// Return the route pattern without an outer [`Router::nest`] prefix.
///
/// [`Router::nest`]: axum::Router::nest
pub(crate) fn route_handler(extensions: &Extensions) -> &str {
    let matched_path = extensions.get::<MatchedPath>().map_or("none", MatchedPath::as_str);
    let Some(nested_path) = extensions.get::<NestedPath>() else {
        return matched_path;
    };
    let prefix = nested_path.as_str().trim_end_matches('/');
    match matched_path.strip_prefix(prefix) {
        Some("") => "/",
        Some(handler) => handler,
        None => matched_path,
    }
}
