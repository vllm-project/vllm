// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use axum::Router;
use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::header::HeaderName;

/// Header that pins a request to one data-parallel rank (engine index), read by
/// the OpenAI routes through `resolve_request_context`.
const X_DATA_PARALLEL_RANK: HeaderName = HeaderName::from_static("x-data-parallel-rank");

/// Request extension naming the data-parallel rank of the listener that
/// accepted the request. Absent on the main listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ListenerRank(pub u32);

/// Wrap `router` for a listener bound to one data-parallel rank: every request
/// goes to `rank`'s engine, whatever `X-data-parallel-rank` it carries, and is
/// tagged with [`ListenerRank`].
pub(crate) fn with_data_parallel_rank(router: Router, rank: u32) -> Router {
    let value = HeaderValue::from(rank);
    router.layer(axum::middleware::map_request(move |mut req: Request| {
        let value = value.clone();
        async move {
            req.headers_mut().insert(X_DATA_PARALLEL_RANK, value);
            req.extensions_mut().insert(ListenerRank(rank));
            req
        }
    }))
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{HeaderMap, Request};
    use axum::routing::get;
    use tower::ServiceExt;

    use super::*;

    async fn echo_rank(headers: HeaderMap, rank: Option<axum::Extension<ListenerRank>>) -> String {
        let header = headers
            .get(&X_DATA_PARALLEL_RANK)
            .map(|value| value.to_str().unwrap().to_owned())
            .unwrap_or_default();
        format!("{header} {:?}", rank.map(|axum::Extension(rank)| rank.0))
    }

    async fn rank_seen(router: Router, header: Option<&str>) -> String {
        let mut request = Request::builder().uri("/");
        if let Some(header) = header {
            request = request.header("X-data-parallel-rank", header);
        }
        let response = router.oneshot(request.body(Body::empty()).unwrap()).await.unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn listener_rank_pins_the_request() {
        let router = with_data_parallel_rank(Router::new().route("/", get(echo_rank)), 5);
        assert_eq!(rank_seen(router, None).await, "5 Some(5)");
    }

    #[tokio::test]
    async fn listener_rank_overrides_the_header() {
        let router = with_data_parallel_rank(Router::new().route("/", get(echo_rank)), 5);
        assert_eq!(rank_seen(router, Some("2")).await, "5 Some(5)");
    }
}
