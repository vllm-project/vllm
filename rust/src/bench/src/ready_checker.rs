// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::backends::{RequestFuncInput, RequestFuncOutput, get_backend};
use crate::cli::BackendKind;
use crate::error::{BenchError, Result};

/// Wait for the serving endpoint to become available.
///
/// Sends test requests with retry until success or timeout.
/// Mirrors Python's `wait_for_endpoint` in ready_checker.py.
pub async fn wait_for_endpoint(
    backend: BackendKind,
    client: &reqwest::Client,
    test_input: &RequestFuncInput,
    timeout_seconds: u64,
    retry_interval: u64,
) -> Result<RequestFuncOutput> {
    let backend = get_backend(backend)?;
    let deadline = Instant::now() + std::time::Duration::from_secs(timeout_seconds);

    tracing::info!(
        timeout_seconds,
        retry_interval,
        "waiting for endpoint readiness"
    );

    let pb = ProgressBar::new(timeout_seconds);
    pb.set_style(
        ProgressStyle::with_template("{msg} |{bar:40}| {elapsed} elapsed, {eta} remaining")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut last_error = String::new();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let elapsed = timeout_seconds.saturating_sub(remaining.as_secs());
        pb.set_position(elapsed);

        if remaining.is_zero() {
            pb.finish_and_clear();
            break;
        }

        // Ping the endpoint
        match backend.send_request(test_input, client).await {
            Ok(output) if output.success => {
                pb.finish_and_clear();
                return Ok(output);
            }
            Ok(output) => {
                let err = output.error.clone();
                let err_last_line = err.lines().last().unwrap_or(&err);
                pb.suspend(|| {
                    tracing::warn!(error = err_last_line, "endpoint is not ready");
                });
                last_error = err;
            }
            Err(e) => {
                last_error = e.to_string();
            }
        }

        // Retry after delay
        let sleep_dur = std::cmp::min(std::time::Duration::from_secs(retry_interval), remaining);
        if !sleep_dur.is_zero() {
            tokio::time::sleep(sleep_dur).await;
        }
    }

    Err(BenchError::EndpointTimeout(timeout_seconds, last_error))
}
