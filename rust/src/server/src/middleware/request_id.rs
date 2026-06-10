use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::header::HeaderName;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Echo the request's `X-Request-Id` on the response, or generate a fresh
/// `uuid4` hex if the request did not provide one.
///
/// Original Python:
/// `vllm.entrypoints.openai.server_utils.XRequestIdMiddleware`.
pub async fn set_request_id_header(req: Request, next: Next) -> Response {
    let incoming = req.headers().get(&X_REQUEST_ID).cloned();
    let mut response = next.run(req).await;
    let value = incoming.unwrap_or_else(|| {
        HeaderValue::from_str(&Uuid::new_v4().simple().to_string())
            .expect("uuid hex is valid header value")
    });
    response.headers_mut().insert(X_REQUEST_ID, value);
    response
}
