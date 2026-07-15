// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenSSL server-config construction for TLS termination.
//!
//! Builds an OpenSSL [`SslContext`] from the uvicorn-style `ssl_*` arguments
//! (certificate chain, private key, mTLS client verifier, optional cipher list).
//! The `tls-listener` crate drives the handshake on each accepted connection.
//!
//! Crypto runs through whichever OpenSSL the binary links (system by default,
//! vendored when built with that feature).

use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result};
use openssl::ssl::{
    AlpnError, SslAcceptor, SslAcceptorBuilder, SslContext, SslContextBuilder, SslFiletype,
    SslMethod, SslOptions, SslVerifyMode, select_next_proto,
};

use crate::config::TlsConfig;

/// Time a client has to complete the TLS handshake before the connection is dropped.
pub(crate) const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);

/// ALPN wire bytes for HTTP/2 (length-prefixed).
const ALPN_H2: &[u8] = b"\x02h2";

/// Build the shared OpenSSL acceptor from validated [`TlsConfig`]: the full
/// certificate chain, the private key (`key_file`, or the certificate file when
/// unset), the mTLS client verifier, and an optional cipher list.
///
/// Starts from the Mozilla intermediate baseline (forward-secret AEAD suites,
/// TLS 1.2 floor, server cipher preference, no compression), a slightly
/// stricter subset of the Python frontend's default suites; `--ssl-ciphers`
/// overrides it.
fn build_server_builder(tls: &TlsConfig) -> Result<SslAcceptorBuilder> {
    let cert_file = tls.cert_file.as_deref().context("--ssl-certfile is required to enable TLS")?;

    let mut builder = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls_server())
        .context("failed to initialize TLS")?;
    builder.set_options(SslOptions::CIPHER_SERVER_PREFERENCE);

    // Load the whole chain (leaf + intermediates), not just the leaf, so
    // deployments behind an intermediate CA serve a complete chain.
    ensure_exists(cert_file, "--ssl-certfile")?;
    builder.set_certificate_chain_file(cert_file).with_context(|| {
        format!("failed to parse certificate chain in --ssl-certfile {cert_file:?}")
    })?;

    // When `key_file` is unset the key is read from the certificate file
    // (combined PEM).
    let key_file = tls.key_file.as_deref().unwrap_or(cert_file);
    ensure_exists(key_file, "private key file")?;
    builder
        .set_private_key_file(key_file, SslFiletype::PEM)
        .with_context(|| format!("failed to parse private key in {key_file:?}"))?;
    builder
        .check_private_key()
        .context("the certificate and private key do not match")?;

    configure_client_auth(&mut builder, tls)?;

    if let Some(ciphers) = tls.ciphers.as_deref().filter(|c| !c.is_empty()) {
        builder
            .set_cipher_list(ciphers)
            .with_context(|| format!("invalid --ssl-ciphers {ciphers:?}"))?;
    }

    Ok(builder)
}

/// Build the HTTP [`SslContext`] (HTTP/1.1; no ALPN, matching uvicorn).
pub(crate) fn build_server_config(tls: &TlsConfig) -> Result<SslContext> {
    Ok(build_server_builder(tls)?.build().into_context())
}

/// Build the gRPC [`SslContext`]: identical to [`build_server_config`] but
/// negotiates ALPN `h2`, which HTTP/2 over TLS requires.
pub(crate) fn build_grpc_server_config(tls: &TlsConfig) -> Result<SslContext> {
    let mut builder = build_server_builder(tls)?;
    builder.set_alpn_select_callback(|_ssl, client| {
        select_next_proto(ALPN_H2, client).ok_or(AlpnError::NOACK)
    });
    Ok(builder.build().into_context())
}

/// Fail loudly with a flag-named message when a configured file is missing,
/// distinguishing it from a malformed-PEM error raised later by OpenSSL (whose
/// `ErrorStack` does not name the offending file).
fn ensure_exists(path: &str, what: &str) -> Result<()> {
    std::fs::metadata(Path::new(path))
        .map(drop)
        .with_context(|| format!("failed to read {what} {path:?}"))
}

/// Apply the `cert_reqs` client-certificate policy: 0 = none, 1 = optional
/// (verify if presented, allow anonymous), 2 = required. `PEER` without a custom
/// verify callback still rejects a presented-but-untrusted certificate.
fn configure_client_auth(builder: &mut SslContextBuilder, tls: &TlsConfig) -> Result<()> {
    if tls.cert_reqs == 0 {
        builder.set_verify(SslVerifyMode::NONE);
        return Ok(());
    }

    let ca_file = tls
        .ca_certs
        .as_deref()
        .context("--ssl-ca-certs is required for client certificate verification")?;
    ensure_exists(ca_file, "--ssl-ca-certs")?;
    builder
        .set_ca_file(ca_file)
        .with_context(|| format!("failed to parse --ssl-ca-certs {ca_file:?}"))?;

    let mut mode = SslVerifyMode::PEER;
    if tls.cert_reqs == 2 {
        mode |= SslVerifyMode::FAIL_IF_NO_PEER_CERT;
    }
    builder.set_verify(mode);
    Ok(())
}
