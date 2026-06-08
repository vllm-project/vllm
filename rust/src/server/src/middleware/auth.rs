use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::state::AppState;

const GUARDED_PREFIXES: &[&str] = &["/v1", "/v2", "/inference"];

/// Authenticate guarded HTTP routes with an OpenAI-compatible bearer token.
///
/// Mirrors Python `AuthenticationMiddleware`: OPTIONS requests and non-guarded
/// helper endpoints such as `/health` are allowed through without a token.
pub async fn authenticate_api_key(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() == Method::OPTIONS || !requires_auth(req.uri().path()) {
        return next.run(req).await;
    }

    if verify_token(req.headers().get(AUTHORIZATION), state.api_keys()) {
        return next.run(req).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "Unauthorized" })),
    )
        .into_response()
}

fn requires_auth(path: &str) -> bool {
    GUARDED_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
}

fn verify_token(authorization: Option<&HeaderValue>, api_keys: &[String]) -> bool {
    let Some(authorization) = authorization else {
        return false;
    };
    let Ok(authorization) = authorization.to_str() else {
        return false;
    };
    let Some((scheme, token)) = authorization.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }

    let token_hash = sha256_digest(token.as_bytes());
    let mut token_match = false;
    for api_key in api_keys {
        token_match |= constant_time_eq(&token_hash, &sha256_digest(api_key.as_bytes()));
    }
    token_match
}

fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    use subtle::ConstantTimeEq;

    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use super::{constant_time_eq, sha256_digest};

    #[test]
    fn constant_time_eq_checks_sha256_digests() {
        assert!(constant_time_eq(
            &sha256_digest(b"secret"),
            &sha256_digest(b"secret")
        ));
        assert!(!constant_time_eq(
            &sha256_digest(b"secret"),
            &sha256_digest(b"secrex")
        ));
        assert!(!constant_time_eq(
            &sha256_digest(b"secret"),
            &sha256_digest(b"secret-more")
        ));
    }
}
