// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Process-group helpers so a benchmarked command's entire subtree (e.g. a
//! shell wrapping `vllm serve`, which may itself fork engine-core workers)
//! can be torn down together, instead of leaking orphans that would keep the
//! port bound or the GPU busy for the next run.
//!
//! Adapted from the equivalent helper in `vllm-managed-engine`; duplicated
//! here rather than shared so this crate stays a small, dependency-light
//! standalone tool.

use std::io;

use anyhow::{Context, Result};
use tokio::process::Command;

/// Place the child into its own process group so its whole subtree can be
/// signaled together.
pub fn configure(command: &mut Command) {
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Send SIGTERM to the process group led by `pid`.
pub fn terminate(pid: u32) -> Result<()> {
    signal(pid, libc::SIGTERM)
}

/// Send SIGKILL to the process group led by `pid`.
pub fn kill(pid: u32) -> Result<()> {
    signal(pid, libc::SIGKILL)
}

/// Deliver one signal to the process group led by `pid`. A missing process
/// (already exited) is treated as success.
fn signal(pid: u32, signal: i32) -> Result<()> {
    let rc = unsafe { libc::kill(-(pid as i32), signal) };
    if rc == 0 {
        return Ok(());
    }

    let error = io::Error::last_os_error();
    if matches!(error.raw_os_error(), Some(code) if code == libc::ESRCH) {
        return Ok(());
    }
    Err(error).context("failed to signal process group")
}
