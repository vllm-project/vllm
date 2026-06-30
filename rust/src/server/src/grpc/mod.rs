//! gRPC Generate service backed by the shared [`vllm_text::TextLlm`] facade.

mod convert;

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt as _, stream};
use thiserror_ext::AsReport as _;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc;
use tokio_openssl::SslStream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::server::{Connected, TcpConnectInfo};
use tonic::{Request, Response, Status};
use tracing::info;
use vllm_text::{DecodedTextEvent, TextOutputStreamExt as _};

use self::convert::ResponseOpts;
use crate::listener::{Listener, ListenerIo};
use crate::state::AppState;

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub use pb::generate_server::GenerateServer;

#[cfg(test)]
mod tests;

/// Newtype over `tokio-openssl`'s `SslStream` so we can implement tonic's
/// [`Connected`] on it (the orphan rule blocks doing so on the foreign type).
pub(crate) struct GrpcTlsStream {
    inner: SslStream<ListenerIo>,
}

impl GrpcTlsStream {
    pub(crate) fn new(inner: SslStream<ListenerIo>) -> Self {
        Self { inner }
    }
}

impl AsyncRead for GrpcTlsStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for GrpcTlsStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

impl Connected for GrpcTlsStream {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> TcpConnectInfo {
        self.inner.get_ref().connect_info()
    }
}

/// Adapt the shared server listener into tonic's incoming stream shape.
pub(crate) fn incoming(listener: Listener) -> impl Stream<Item = std::io::Result<ListenerIo>> {
    stream::unfold(listener, |mut listener| async move {
        let (io, _) = axum::serve::Listener::accept(&mut listener).await;
        Some((Ok(io), listener))
    })
}

/// Wrap the gRPC listener so each accepted connection completes a TLS handshake
/// before tonic serves it.
pub(crate) fn tls_incoming(
    listener: Listener,
    context: openssl::ssl::SslContext,
    handshake_timeout: std::time::Duration,
) -> impl Stream<Item = std::io::Result<GrpcTlsStream>> {
    tls_listener::builder(context)
        .handshake_timeout(handshake_timeout)
        .listen(listener)
        .map(|res| {
            res.map(|(inner, _addr)| GrpcTlsStream::new(inner))
                .map_err(std::io::Error::other)
        })
}

/// gRPC Generate service implementation backed by the shared application state.
pub struct GenerateServiceImpl {
    state: Arc<AppState>,
}

impl GenerateServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl pb::generate_server::Generate for GenerateServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request =
            convert::to_text_request(proto_req, false, self.state.served_model_names())?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let collected = stream.collect_output().await.map_err(text_error_to_status)?;

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let text_request =
            convert::to_text_request(proto_req, true, self.state.served_model_names())?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(text_error_to_status(e)),
                    Ok(DecodedTextEvent::Start {
                        prompt_token_ids,
                        prompt_logprobs,
                    }) => {
                        let prompt_info = convert::to_prompt_info(
                            &prompt_token_ids,
                            prompt_logprobs.as_ref(),
                            &response_opts,
                        );
                        Ok(pb::GenerateResponse {
                            prompt_info: Some(prompt_info),
                            outputs: None,
                        })
                    }
                    Ok(DecodedTextEvent::TextDelta {
                        delta,
                        token_ids,
                        logprobs,
                        finished,
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &delta,
                            &token_ids,
                            logprobs.as_ref(),
                            finished.as_ref(),
                            &response_opts,
                        )),
                    }),
                };

                if tx.send(response).await.is_err() {
                    // Client disconnected.
                    break;
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

fn text_error_to_status(error: vllm_text::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}
