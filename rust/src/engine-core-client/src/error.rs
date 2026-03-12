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
    #[error("transport error")]
    Transport(#[from] zeromq::ZmqError),
    #[error("ready handshake timed out after {timeout:?}")]
    ReadyTimeout { timeout: Duration },
    #[error("unexpected engine identity in ready handshake: expected {expected:?}, got {actual:?}")]
    UnexpectedReadyIdentity { expected: Vec<u8>, actual: Vec<u8> },
    #[error("unexpected ready message: {reason}")]
    UnexpectedReadyMessage { reason: String },
    #[error("unsupported auxiliary frame(s): expected 1 frame, got {frame_count}")]
    UnsupportedAuxFrames { frame_count: usize },
    #[error("unsupported field `{field}` in {context}")]
    UnsupportedField {
        context: &'static str,
        field: &'static str,
    },
    #[error("engine control channel closed unexpectedly: {0}")]
    ControlClosed(String),
    #[error("output stream closed")]
    OutputClosed,
}
