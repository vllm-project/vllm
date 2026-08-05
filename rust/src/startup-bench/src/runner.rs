// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Spawn one variant's command, time how long it takes to become ready, and
//! tear it down again.

use std::fs::File;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::process_group;

/// Outcome of a single timed (or warmup) run of one variant.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunOutcome {
    /// Wall-clock seconds from process spawn to the readiness endpoint first
    /// responding successfully. `None` if the run failed (crashed or timed
    /// out) before becoming ready.
    pub ready_secs: Option<f64>,
}

/// Spawn `command` via `sh -c`, poll `health_url` until it responds
/// successfully or `ready_timeout` elapses, then terminate the process group.
///
/// `log_path`, if given, receives the child's combined stdout/stderr — handy
/// for diagnosing a run that fails to become ready.
pub async fn run_once(
    command: &str,
    health_url: &str,
    ready_timeout: Duration,
    poll_interval: Duration,
    shutdown_timeout: Duration,
    log_path: Option<&Path>,
) -> Result<RunOutcome> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);

    match log_path {
        Some(path) => {
            let out = File::create(path)
                .with_context(|| format!("failed to create log file {}", path.display()))?;
            let err = out.try_clone().context("failed to clone log file handle")?;
            cmd.stdout(Stdio::from(out)).stderr(Stdio::from(err));
        }
        None => {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }
    cmd.stdin(Stdio::null());
    process_group::configure(&mut cmd);

    let mut child = cmd.spawn().with_context(|| format!("failed to spawn command: {command}"))?;
    let pid = child.id();

    let ready_secs = wait_until_ready(&mut child, health_url, ready_timeout, poll_interval).await?;

    if let Some(pid) = pid {
        shut_down(&mut child, pid, shutdown_timeout).await;
    }

    Ok(RunOutcome { ready_secs })
}

/// Poll `health_url` until it returns a successful status, the process
/// exits, or `timeout` elapses. Returns the elapsed seconds on success.
async fn wait_until_ready(
    child: &mut tokio::process::Child,
    health_url: &str,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<Option<f64>> {
    let start = Instant::now();
    // Short per-request timeout: readiness polling relies on the outer retry
    // loop, not on any single request hanging around.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("failed to build HTTP client")?;

    loop {
        if let Some(status) = child.try_wait().context("failed to poll child status")? {
            tracing::warn!(%status, "process exited before becoming ready");
            return Ok(None);
        }

        if let Ok(response) = client.get(health_url).send().await
            && response.status().is_success()
        {
            return Ok(Some(start.elapsed().as_secs_f64()));
        }

        if start.elapsed() >= timeout {
            tracing::warn!(?timeout, "timed out waiting for readiness");
            return Ok(None);
        }

        tokio::time::sleep(poll_interval).await;
    }
}

/// Terminate the process group led by `pid`, escalating to SIGKILL if it
/// doesn't exit within `timeout`.
async fn shut_down(child: &mut tokio::process::Child, pid: u32, timeout: Duration) {
    if let Err(error) = process_group::terminate(pid) {
        tracing::warn!(?error, "failed to send SIGTERM to process group");
    }

    if tokio::time::timeout(timeout, child.wait()).await.is_ok() {
        return;
    }

    tracing::warn!("process did not exit after SIGTERM, sending SIGKILL");
    if let Err(error) = process_group::kill(pid) {
        tracing::warn!(?error, "failed to send SIGKILL to process group");
    }
    let _ = child.wait().await;
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    /// Reserve an ephemeral port by binding then immediately releasing it.
    /// Small TOCTOU race in theory; acceptable for these best-effort tests.
    fn reserve_port() -> u16 {
        TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    fn python_http_server_cmd(port: u16, delay_secs: f64) -> String {
        format!(
            "python3 -c \"import time, http.server, socketserver; \
             time.sleep({delay_secs}); \
             socketserver.TCPServer.allow_reuse_address = True; \
             httpd = socketserver.TCPServer(('127.0.0.1', {port}), http.server.SimpleHTTPRequestHandler); \
             httpd.serve_forever()\""
        )
    }

    // Requires python3; run explicitly with `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn reports_ready_secs_once_endpoint_responds() {
        let port = reserve_port();
        let command = python_http_server_cmd(port, 0.0);
        let health_url = format!("http://127.0.0.1:{port}/");

        let outcome = run_once(
            &command,
            &health_url,
            Duration::from_secs(10),
            Duration::from_millis(20),
            Duration::from_secs(5),
            None,
        )
        .await
        .unwrap();

        assert!(outcome.ready_secs.is_some());
    }

    // Requires python3; run explicitly with `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn slower_variant_reports_larger_ready_secs() {
        let fast_port = reserve_port();
        let slow_port = reserve_port();
        let fast_command = python_http_server_cmd(fast_port, 0.0);
        let slow_command = python_http_server_cmd(slow_port, 1.0);
        let fast_url = format!("http://127.0.0.1:{fast_port}/");
        let slow_url = format!("http://127.0.0.1:{slow_port}/");

        let fast = run_once(
            &fast_command,
            &fast_url,
            Duration::from_secs(10),
            Duration::from_millis(20),
            Duration::from_secs(5),
            None,
        );
        let slow = run_once(
            &slow_command,
            &slow_url,
            Duration::from_secs(10),
            Duration::from_millis(20),
            Duration::from_secs(5),
            None,
        );

        let (fast, slow) = tokio::join!(fast, slow);
        let fast_secs = fast.unwrap().ready_secs.unwrap();
        let slow_secs = slow.unwrap().ready_secs.unwrap();
        assert!(slow_secs > fast_secs, "expected {slow_secs} > {fast_secs}");
    }

    // Requires python3; run explicitly with `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn process_that_exits_without_serving_reports_no_ready_secs() {
        let outcome = run_once(
            "python3 -c \"import time; time.sleep(0.1)\"",
            "http://127.0.0.1:1/never-bound",
            Duration::from_secs(5),
            Duration::from_millis(20),
            Duration::from_secs(5),
            None,
        )
        .await
        .unwrap();

        assert!(outcome.ready_secs.is_none());
    }
}
