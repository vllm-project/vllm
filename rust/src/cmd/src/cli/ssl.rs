// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! TLS CLI options mirroring Python's uvicorn `ssl_*` arguments, shared by the
//! OpenAI frontend and the render server.

use clap::Args;
use serde::Deserialize;
use vllm_server::TlsConfig;

/// TLS options mirroring Python's uvicorn `ssl_*` arguments, shared by the
/// OpenAI frontend and the render server.
#[derive(Clone, Debug, Default, Args, PartialEq, Eq, Deserialize)]
pub struct SslArgs {
    /// The file path to the SSL key file. When omitted, the key is read from
    /// `--ssl-certfile` (combined PEM).
    #[arg(long)]
    #[serde(default)]
    pub ssl_keyfile: Option<String>,

    /// The file path to the SSL cert file. Enables TLS when set.
    #[arg(long)]
    #[serde(default)]
    pub ssl_certfile: Option<String>,

    /// The CA certificates file used to verify client certificates (mTLS).
    #[arg(long)]
    #[serde(default)]
    pub ssl_ca_certs: Option<String>,

    /// Whether a client certificate is required: 0 = none, 1 = optional,
    /// 2 = required (mirrors Python's `ssl.CERT_*`).
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=2))]
    #[serde(default)]
    pub ssl_cert_reqs: i32,

    /// OpenSSL cipher string for HTTPS (TLS 1.2 and below).
    /// When unset, the linked OpenSSL's default suites are used.
    #[arg(long)]
    #[serde(default)]
    pub ssl_ciphers: Option<String>,
}

impl SslArgs {
    /// Build the TLS config: `Some` when any `ssl_*` argument is set, else
    /// `None` (plaintext). The combination is validated in [`TlsConfig::validate`].
    pub fn tls_config(&self) -> Option<TlsConfig> {
        let tls_requested = self.ssl_certfile.is_some()
            || self.ssl_keyfile.is_some()
            || self.ssl_ca_certs.is_some()
            || self.ssl_cert_reqs != 0
            || self.ssl_ciphers.is_some();
        tls_requested.then(|| TlsConfig {
            cert_file: self.ssl_certfile.clone(),
            key_file: self.ssl_keyfile.clone(),
            ca_certs: self.ssl_ca_certs.clone(),
            cert_reqs: self.ssl_cert_reqs,
            ciphers: self.ssl_ciphers.clone(),
        })
    }
}
