// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Silero VAD and Smart Turn v3.2 for server-side turn detection, without ONNX Runtime.
//!
//! Both models are read from their published ONNX files (the bytes are checked
//! against pinned sha256 digests, then the weights are pulled out of the graph);
//! the forward passes are written by hand. Nothing is converted or redistributed.
//!
//! - [`silero`]: Silero VAD v6.2, 16 kHz, one 512-sample frame per call with
//!   per-stream state.
//! - [`smart_turn`]: Smart Turn v3.2, the fp32 graph, plus the Whisper log-mel
//!   front end it expects.

mod kernels;
mod onnx;
pub mod silero;
pub mod smart_turn;

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Where the supported model files come from. Callers download them; this crate
/// only checks that the bytes are the pinned ones.
pub mod pins {
    pub const SILERO_REPO: &str = "istupakov/silero-vad-onnx";
    pub const SILERO_REVISION: &str = "8b14476858ef240c50b3884bb38cc67290c1cc70";
    pub const SILERO_FILE: &str = "silero_vad.onnx";
    pub const SILERO_SHA256: &str =
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";

    pub const SMART_TURN_REPO: &str = "pipecat-ai/smart-turn-v3";
    pub const SMART_TURN_REVISION: &str = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93";
    /// The fp32 graph. The int8 `smart-turn-v3.2-cpu.onnx` is not supported: its
    /// uint8 activation quantization makes the output depend on runtime details.
    pub const SMART_TURN_FILE: &str = "smart-turn-v3.2-gpu.onnx";
    pub const SMART_TURN_SHA256: &str =
        "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be";
}

pub type Result<T> = std::result::Result<T, TurnDetectionError>;

#[derive(Debug, Error)]
pub enum TurnDetectionError {
    /// The model file could not be read.
    #[error("failed to read {}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The file is not the pinned one.
    #[error("checksum mismatch for {}: expected {expected}, got {actual}", path.display())]
    Checksum {
        path: PathBuf,
        expected: &'static str,
        actual: String,
    },
    /// The file is not the model this crate expects (missing tensor, wrong shape, bad protobuf).
    #[error("unsupported model: {message}")]
    Model { message: String },
}

/// Read a model file and, when `sha256` is given, check its digest.
pub(crate) fn read_model(path: &Path, sha256: Option<&'static str>) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).map_err(|source| TurnDetectionError::Io {
        path: path.to_owned(),
        source,
    })?;
    if let Some(expected) = sha256 {
        use sha2::{Digest, Sha256};
        let actual: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        if actual != expected {
            return Err(TurnDetectionError::Checksum {
                path: path.to_owned(),
                expected,
                actual,
            });
        }
    }
    Ok(bytes)
}
