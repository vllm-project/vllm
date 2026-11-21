// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the shm multimodal cache.
#[derive(Debug, Error)]
pub enum Error {
    #[error("shm object storage error")]
    ShmStorage(#[from] vllm_shm_object_storage::Error),
}
