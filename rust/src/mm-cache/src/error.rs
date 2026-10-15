// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the shm multimodal cache.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("failed to create shm multimodal cache {name:?}: {message}")]
    MmShmCacheCreate { name: String, message: String },

    #[error("failed to serialize multimodal item for shm cache: {0}")]
    Serialize(#[from] vllm_engine_core_client::Error),
}
