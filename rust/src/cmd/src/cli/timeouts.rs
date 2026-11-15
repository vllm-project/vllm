// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Listener timeout options shared by the OpenAI frontend and the render
//! server.

use std::time::Duration;

use clap::Args;
use serde::Deserialize;
use vllm_server::DEFAULT_KEEP_ALIVE_TIMEOUT;

/// Drain and keep-alive timeouts of the HTTP and gRPC listeners, shared by the
/// OpenAI frontend and the render server.
#[derive(Clone, Debug, Default, Args, PartialEq, Eq, Deserialize)]
pub struct ServerTimeoutArgs {
    /// Maximum time to wait for active requests to drain during shutdown.
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub shutdown_timeout: u64,
    /// Maximum idle time (seconds) on a keep-alive HTTP connection before the
    /// server closes it (default 5).
    #[arg(long = "http-timeout-keep-alive", env = "VLLM_HTTP_TIMEOUT_KEEP_ALIVE")]
    #[serde(default)]
    pub http_timeout_keep_alive: Option<u64>,
}

impl ServerTimeoutArgs {
    /// Maximum time to wait for active requests to drain during shutdown.
    pub fn shutdown_timeout(&self) -> Duration {
        Duration::from_secs(self.shutdown_timeout)
    }

    /// Maximum idle time on a keep-alive HTTP connection before the server
    /// closes it.
    pub fn keep_alive_timeout(&self) -> Duration {
        self.http_timeout_keep_alive
            .map_or(DEFAULT_KEEP_ALIVE_TIMEOUT, Duration::from_secs)
    }
}
