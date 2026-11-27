// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes, HttpBody};
use axum::response::Response;
use http_body::{Frame, SizeHint};

/// Keep `guard` alive until the response body is dropped: after the last
/// frame is sent, or earlier when the client goes away mid-stream.
pub(super) fn hold_until_body_dropped<G>(response: Response, guard: G) -> Response
where
    G: Send + Unpin + 'static,
{
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(GuardedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A wrapper around response bodies that holds a guard, which runs its `Drop`
/// when the body is fully consumed and dropped.
struct GuardedBody<G> {
    inner: Body,
    _guard: G,
}

// Simply delegate all `HttpBody` methods to the inner body.
impl<G: Unpin> HttpBody for GuardedBody<G> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
