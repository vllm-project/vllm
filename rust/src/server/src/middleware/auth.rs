use std::sync::Arc;

use axum::Json;
use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// Path prefixes that require a valid API key when authentication is enabled.
///
/// Matches the Python frontend's `GUARDED_PREFIX` in
/// `vllm.entrypoints.serve.utils.server_utils`. Any route whose matched path
/// starts with one of these prefixes will be checked for a valid
/// `Authorization: Bearer <key>` header.
const GUARDED_PREFIXES: &[&str] = &["/v1", "/v2", "/inference"];

/// Reject requests to guarded endpoints that do not carry a valid
/// `Authorization: Bearer <key>` header.
///
/// Requests whose matched path does not start with any of the
/// [`GUARDED_PREFIXES`] are passed through without authentication.
///
/// Original Python:
/// `vllm.entrypoints.serve.utils.server_utils.AuthenticationMiddleware`.
pub async fn api_key_authentication(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let handler = req.extensions().get::<MatchedPath>().map_or("", |path| path.as_str());

    let guarded = GUARDED_PREFIXES.iter().any(|prefix| handler.starts_with(prefix));
    if !guarded {
        return next.run(req).await;
    }

    // A matching response similar to python frontend
    let unauthorized = (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": "Unauthorized"})),
    )
        .into_response();

    let auth_header = req.headers().get(AUTHORIZATION).and_then(|header| header.to_str().ok());
    let Some(auth_header) = auth_header else {
        return unauthorized;
    };

    let Some((scheme, token)) = auth_header.split_once(" ") else {
        return unauthorized;
    };

    if !scheme.eq_ignore_ascii_case("Bearer") {
        return unauthorized;
    }

    if state.is_authorized(token) {
        next.run(req).await
    } else {
        unauthorized
    }
}
