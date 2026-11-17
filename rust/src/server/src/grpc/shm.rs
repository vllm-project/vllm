// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Copy immutable, content-verified local inputs before engine submission.

use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _};
use std::os::unix::fs::{FileExt as _, OpenOptionsExt as _};
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest as _, Sha256};
use tonic::Status;

use super::pb;

const MAX_SHARED_BYTES: u64 = 256 * 1024 * 1024;

/// An open directory pins the input namespace across path renames/replacement.
pub(super) struct SharedMemoryDirectory(File);

impl SharedMemoryDirectory {
    pub(super) fn open(path: &Path) -> io::Result<Self> {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .map(Self)
    }

    fn read(&self, input: &pb::SharedMemoryInput) -> Result<Vec<u8>, Status> {
        if input.name.is_empty()
            || input.name.len() > 255
            || !input.name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(Status::invalid_argument(
                "kwargs_shm.name must be one safe filename",
            ));
        }
        let end = input
            .offset
            .checked_add(input.length)
            .filter(|&end| end <= i64::MAX as u64)
            .ok_or_else(|| Status::invalid_argument("kwargs_shm range overflow"))?;
        if input.length == 0 || input.length > MAX_SHARED_BYTES {
            return Err(Status::resource_exhausted(
                "kwargs_shm length must be in 1..=256 MiB",
            ));
        }
        if input.sha256.len() != 32 {
            return Err(Status::invalid_argument(
                "kwargs_shm.sha256 must contain 32 raw bytes",
            ));
        }
        let name = CString::new(input.name.as_bytes()).expect("validated ASCII name");
        // SAFETY: both descriptors and the NUL-terminated name are valid. A
        // successful openat returns a new owned fd; no symlink is followed.
        let fd = unsafe {
            libc::openat(
                self.0.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(Status::invalid_argument("cannot open kwargs_shm input"));
        }
        // SAFETY: openat succeeded and this File is the fd's sole owner.
        let file = unsafe { File::from_raw_fd(fd) };
        let metadata = file
            .metadata()
            .map_err(|_| Status::invalid_argument("cannot stat kwargs_shm input"))?;
        if !metadata.is_file() || end > metadata.len() {
            return Err(Status::invalid_argument(
                "kwargs_shm requires a regular file containing the range",
            ));
        }
        let length = usize::try_from(input.length)
            .map_err(|_| Status::resource_exhausted("kwargs_shm exceeds platform limits"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Status::resource_exhausted("cannot allocate kwargs_shm snapshot"))?;
        bytes.resize(length, 0);
        file.read_exact_at(&mut bytes, input.offset)
            .map_err(|_| Status::invalid_argument("cannot read kwargs_shm range"))?;
        if Sha256::digest(&bytes).as_slice() != input.sha256 {
            return Err(Status::invalid_argument("kwargs_shm SHA-256 mismatch"));
        }
        Ok(bytes)
    }
}

/// Return the number of copied bytes, so the inline limit remains independent.
pub(super) async fn import_kwargs(
    media: &mut Vec<pb::MediaItem>,
    directory: Option<Arc<SharedMemoryDirectory>>,
) -> Result<usize, Status> {
    let mut total = 0u64;
    for item in media.iter() {
        if let Some(pb::media_item::Source::Features(features)) = &item.source
            && let Some(input) = &features.kwargs_shm
        {
            if features.kwargs.is_some() {
                return Err(Status::invalid_argument(
                    "set only one of features.kwargs and features.kwargs_shm",
                ));
            }
            total = total
                .checked_add(input.length)
                .filter(|&size| size <= MAX_SHARED_BYTES)
                .ok_or_else(|| {
                    Status::resource_exhausted("aggregate kwargs_shm exceeds 256 MiB")
                })?;
        }
    }
    if !media.iter().any(|item| {
        matches!(&item.source,
        Some(pb::media_item::Source::Features(features)) if features.kwargs_shm.is_some())
    }) {
        return Ok(0);
    }
    let directory = directory
        .ok_or_else(|| Status::failed_precondition("SHM inputs require --grpc-shm-directory"))?;
    let mut inputs = std::mem::take(media);
    let imported = tokio::task::spawn_blocking(move || {
        for item in &mut inputs {
            if let Some(pb::media_item::Source::Features(features)) = &mut item.source
                && let Some(input) = features.kwargs_shm.take()
            {
                features.kwargs = Some(directory.read(&input)?);
            }
        }
        Ok::<_, Status>(inputs)
    })
    .await
    .map_err(|_| Status::internal("SHM import task failed"))??;
    *media = imported;
    Ok(total as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_is_bound_to_pinned_directory_and_digest() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("inputs");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("data"), b"original").unwrap();
        let reader = SharedMemoryDirectory::open(&path).unwrap();
        std::fs::rename(&path, parent.path().join("old")).unwrap();
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("data"), b"replaced").unwrap();
        let input = pb::SharedMemoryInput {
            name: "data".into(),
            offset: 0,
            length: 8,
            sha256: Sha256::digest(b"original").to_vec(),
        };
        assert_eq!(reader.read(&input).unwrap(), b"original");
        let stale = pb::SharedMemoryInput {
            sha256: Sha256::digest(b"replaced").to_vec(),
            ..input
        };
        assert_eq!(
            reader.read(&stale).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
    }
}
