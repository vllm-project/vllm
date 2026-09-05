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

/// Authenticate guarded HTTP routes with an OpenAI-compatible bearer token
/// or Anthropic-compatible `x-api-key` header.
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

    if verify_token(req.headers(), state.api_key_hashes()) {
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

fn token_matches(token: &str, api_key_hashes: &[ApiKeyHash]) -> bool {
    let token_hash = hash_api_key(token);
    let mut token_match = false;
    for api_key_hash in api_key_hashes {
        token_match |= constant_time_eq(&token_hash, api_key_hash);
    }
    token_match
}

fn verify_bearer_token(
    authorization: Option<&HeaderValue>,
    api_key_hashes: &[ApiKeyHash],
) -> bool {
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

    token_matches(token, api_key_hashes)
}

fn verify_x_api_key(api_key: Option<&HeaderValue>, api_key_hashes: &[ApiKeyHash]) -> bool {
    let Some(api_key) = api_key else {
        return false;
    };
    let Ok(api_key) = api_key.to_str() else {
        return false;
    };
    token_matches(api_key, api_key_hashes)
}

fn verify_token(headers: &axum::http::HeaderMap, api_key_hashes: &[ApiKeyHash]) -> bool {
    verify_bearer_token(headers.get(AUTHORIZATION), api_key_hashes)
        || verify_x_api_key(headers.get("x-api-key"), api_key_hashes)
}

fn constant_time_eq(left: &ApiKeyHash, right: &ApiKeyHash) -> bool {
    use subtle::ConstantTimeEq;

    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::{constant_time_eq, verify_token};
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

    #[test]
    fn verify_token_accepts_bearer_or_x_api_key() {
        let api_key_hashes = [hash_api_key("secret")];

        let mut bearer_headers = HeaderMap::new();
        bearer_headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        assert!(verify_token(&bearer_headers, &api_key_hashes));

        let mut x_api_key_headers = HeaderMap::new();
        x_api_key_headers.insert("x-api-key", HeaderValue::from_static("secret"));
        assert!(verify_token(&x_api_key_headers, &api_key_hashes));

        let mut wrong_headers = HeaderMap::new();
        wrong_headers.insert("x-api-key", HeaderValue::from_static("wrong"));
        assert!(!verify_token(&wrong_headers, &api_key_hashes));
        assert!(!verify_token(&HeaderMap::new(), &api_key_hashes));
    }
}
