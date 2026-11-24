// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the shm object storage.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("failed to create shm object storage {name:?}: {message}")]
    ShmStorageCreate { name: String, message: String },
    #[error("max object size {max_object_size} exceeds the chunk size limit {limit}")]
    MaxObjectSizeTooLarge {
        max_object_size: usize,
        limit: usize,
    },
}
