// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::state::{ApiKeyHash, AppState, hash_api_key};

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

    if verify_token(req.headers().get(AUTHORIZATION), state.api_key_hashes()) {
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

fn verify_token(authorization: Option<&HeaderValue>, api_key_hashes: &[ApiKeyHash]) -> bool {
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

    let token_hash = hash_api_key(token);
    let mut token_match = false;
    for api_key_hash in api_key_hashes {
        token_match |= constant_time_eq(&token_hash, api_key_hash);
    }
    token_match
}

fn constant_time_eq(left: &ApiKeyHash, right: &ApiKeyHash) -> bool {
    use subtle::ConstantTimeEq;

    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;
    use crate::state::hash_api_key;

    #[test]
    fn constant_time_eq_checks_sha256_digests() {
        assert!(constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secret")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secrex")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secret-more")
        ));
    }
}
