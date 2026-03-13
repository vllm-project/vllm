use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust engine-core client.
#[derive(Debug, Error)]
pub enum Error {
    #[error("messagepack encode failed")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("messagepack decode failed")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("messagepack value decode failed")]
    ValueDecode(#[from] rmpv::decode::Error),
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("transport error")]
    Transport(#[from] zeromq::ZmqError),
    #[error("startup handshake timed out while waiting for {stage} after {timeout:?}")]
    HandshakeTimeout {
        stage: &'static str,
        timeout: Duration,
    },
    #[error("engine input registration timed out after {timeout:?}")]
    InputRegistrationTimeout { timeout: Duration },
    #[error(
        "unexpected engine identity in startup handshake: expected {expected:?}, got {actual:?}"
    )]
    UnexpectedHandshakeIdentity { expected: Vec<u8>, actual: Vec<u8> },
    #[error("unexpected startup handshake message: {reason}")]
    UnexpectedHandshakeMessage { reason: String },
    #[error("unsupported auxiliary frame(s): expected 1 frame, got {frame_count}")]
    UnsupportedAuxFrames { frame_count: usize },
    #[error("unsupported field `{field}` in {context}")]
    UnsupportedField {
        context: &'static str,
        field: &'static str,
    },
    #[error("engine control channel closed unexpectedly: {0}")]
    ControlClosed(String),
    #[error("request `{request_id}` is already in flight")]
    DuplicateRequestId { request_id: String },
    #[error("engine-core output dispatcher closed: {reason}")]
    DispatcherClosed { reason: String },
    #[error("engine-core client is closed: {reason}")]
    ClientClosed { reason: String },
    #[error("request output stream for `{request_id}` closed unexpectedly")]
    RequestStreamClosed { request_id: String },

    /// A special variant to allow cloning the same error.
    #[error(transparent)]
    Shared(Arc<Self>),
}
