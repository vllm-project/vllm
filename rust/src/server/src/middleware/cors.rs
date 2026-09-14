// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! CORS support mirroring Python's Starlette `CORSMiddleware`.
//!
//! Built on `tower_http::cors::CorsLayer`, configured to reproduce Starlette's
//! `CORSMiddleware` behavior for the `--allowed-origins` / `--allowed-methods` /
//! `--allowed-headers` / `--allow-credentials` settings. Two intentional
//! behavioral differences remain, both invisible to real clients:
//!
//! - A rejected preflight returns `200` (empty) rather than Starlette's
//!   `400 "Disallowed CORS ..."`. The browser denies the request either way
//!   (the disallowed `Access-Control-Allow-*` headers are simply absent), and
//!   tower-http makes the preflight reject decision inside its short-circuit,
//!   so matching the `400` would mean re-implementing the layer.
//! - A bare `OPTIONS` (no `Access-Control-Request-Method`) returns `200`
//!   rather than `405`. No real client sends one.

use std::time::Duration;

use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::Response;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use crate::config::CorsConfig;

/// The method set that `"*"` expands to.
const ALL_METHODS: [Method; 7] = [
    Method::DELETE,
    Method::GET,
    Method::HEAD,
    Method::OPTIONS,
    Method::PATCH,
    Method::POST,
    Method::PUT,
];

/// Headers always treated as allowed (the CORS safelist).
const SAFELISTED_HEADERS: [&str; 4] = [
    "accept",
    "accept-language",
    "content-language",
    "content-type",
];

fn is_wildcard(values: &[String]) -> bool {
    values.iter().any(|value| value == "*")
}

/// Build a `CorsLayer` from the resolved [`CorsConfig`].
///
/// Values are assumed valid: [`CorsConfig::validate`] runs at startup before
/// the router is built.
pub fn cors_layer(cfg: &CorsConfig) -> CorsLayer {
    let wildcard_origins = is_wildcard(&cfg.allow_origins);

    let allow_origin = if wildcard_origins {
        if cfg.allow_credentials {
            // `*` with credentials is illegal, so reflect the request origin
            // instead; this also avoids tower-http's wildcard+credentials panic.
            AllowOrigin::mirror_request()
        } else {
            AllowOrigin::any()
        }
    } else {
        AllowOrigin::list(
            cfg.allow_origins
                .iter()
                .map(|origin| origin.parse::<HeaderValue>().expect("validated origin"))
                .collect::<Vec<_>>(),
        )
    };

    // Expand `*` to an explicit list rather than `Any`, so we emit the method
    // names (not `*`) and never hit tower-http's `Any`+credentials panic.
    let allow_methods = if is_wildcard(&cfg.allow_methods) {
        AllowMethods::list(ALL_METHODS)
    } else {
        AllowMethods::list(
            cfg.allow_methods
                .iter()
                .map(|method| method.parse::<Method>().expect("validated method"))
                .collect::<Vec<_>>(),
        )
    };

    let allow_headers = if is_wildcard(&cfg.allow_headers) {
        // `*` mirrors the requested headers.
        AllowHeaders::mirror_request()
    } else {
        // Union the safelisted headers, lowercased and sorted.
        let mut names: Vec<String> = SAFELISTED_HEADERS.iter().map(|s| s.to_string()).collect();
        names.extend(cfg.allow_headers.iter().map(|h| h.to_ascii_lowercase()));
        names.sort();
        names.dedup();
        AllowHeaders::list(
            names
                .iter()
                .map(|header| header.parse::<HeaderName>().expect("validated header"))
                .collect::<Vec<_>>(),
        )
    };

    // Emit `Vary: Origin` only when the allow-origin is dynamic (explicit
    // origins, or credentials); the wildcard + no-credentials case emits no
    // `Vary` at all, and an empty list disables the header here.
    let vary: Vec<HeaderName> = if !wildcard_origins || cfg.allow_credentials {
        vec![header::ORIGIN]
    } else {
        vec![]
    };

    CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_methods(allow_methods)
        .allow_headers(allow_headers)
        .allow_credentials(cfg.allow_credentials)
        .max_age(Duration::from_secs(600))
        .vary(vary)
}

/// Strip CORS response headers when the request carried no `Origin`.
///
/// A request without an `Origin` should carry no CORS headers, but tower-http
/// emits `Vary` and `Access-Control-Allow-*` unconditionally. Removing them on
/// no-`Origin` requests keeps non-CORS responses (e.g. `/health`, plain `curl`)
/// clean.
pub async fn strip_cors_on_no_origin(req: Request, next: Next) -> Response {
    let had_origin = req.headers().contains_key(header::ORIGIN);
    let mut response = next.run(req).await;
    if !had_origin {
        let headers = response.headers_mut();
        headers.remove(header::VARY);
        headers.remove(header::ACCESS_CONTROL_ALLOW_ORIGIN);
        headers.remove(header::ACCESS_CONTROL_ALLOW_CREDENTIALS);
        headers.remove(header::ACCESS_CONTROL_ALLOW_METHODS);
        headers.remove(header::ACCESS_CONTROL_ALLOW_HEADERS);
        headers.remove(header::ACCESS_CONTROL_MAX_AGE);
        headers.remove(header::ACCESS_CONTROL_EXPOSE_HEADERS);
    }
    response
}
