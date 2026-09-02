// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::future::Future;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use thiserror_ext::AsReport as _;

use crate::backends::{RequestFuncInput, RequestFuncOutput, get_backend};
use crate::cli::BackendKind;
use crate::error::{BenchError, Result};

/// Retry an operation until it succeeds or the readiness timeout expires.
pub(crate) async fn retry_with_timeout<T, F, Fut>(
    timeout_seconds: u64,
    retry_interval: u64,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    if timeout_seconds == 0 {
        return operation().await;
    }

    let deadline = Instant::now() + std::time::Duration::from_secs(timeout_seconds);

    loop {
        let error = match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        tracing::warn!(error = %error.as_report(), "endpoint is not ready");

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(BenchError::EndpointTimeout(
                timeout_seconds,
                format!("{}", error.as_report()),
            ));
        }

        let sleep_duration =
            std::cmp::min(std::time::Duration::from_secs(retry_interval), remaining);
        if !sleep_duration.is_zero() {
            tokio::time::sleep(sleep_duration).await;
        }
    }
}

/// Fetch the first model from the server, retrying while it starts.
pub(crate) async fn get_first_model(
    base_url: &str,
    client: &reqwest::Client,
    extra_headers: &Option<std::collections::HashMap<String, String>>,
    timeout_seconds: u64,
) -> Result<(String, String)> {
    let url = format!("{base_url}/v1/models");
    retry_with_timeout(timeout_seconds, 5, || async {
        let mut request = client.get(&url);
        if let Some(headers) = extra_headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        // Add API key from environment
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?.error_for_status()?;
        let data: serde_json::Value = response.json().await?;
        if let Some(model) = data
            .get("data")
            .and_then(|value| value.as_array())
            .and_then(|models| models.first())
        {
            let id =
                model.get("id").and_then(|value| value.as_str()).unwrap_or_default().to_string();
            let root =
                model.get("root").and_then(|value| value.as_str()).unwrap_or(&id).to_string();
            return Ok((id, root));
        }

        Err(BenchError::Config(format!(
            "No models found on the server at {base_url}"
        )))
    })
    .await
}

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
                last_error = format!("{}", e.as_report());
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        let attempts = AtomicUsize::new(0);

        let value = retry_with_timeout(1, 0, || async {
            if attempts.fetch_add(1, Ordering::Relaxed) < 2 {
                Err(BenchError::Backend("not ready".into()))
            } else {
                Ok(42)
            }
        })
        .await
        .unwrap();

        assert_eq!(value, 42);
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn zero_timeout_does_not_retry() {
        let attempts = AtomicUsize::new(0);

        let error = retry_with_timeout(0, 0, || async {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>(BenchError::Backend("not ready".into()))
        })
        .await
        .unwrap_err();

        assert!(matches!(error, BenchError::Backend(_)));
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }
}
