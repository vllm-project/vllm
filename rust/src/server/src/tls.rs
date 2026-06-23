//! TLS termination for the HTTP listener.
//!
//! Builds an OpenSSL [`SslContext`] from the uvicorn-style `ssl_*` arguments and
//! wraps the unified [`Listener`](crate::listener::Listener) so each accepted
//! connection completes its TLS handshake lazily, inside axum's per-connection
//! task. Driving the handshake there (rather than inside `accept()`) keeps the
//! serial accept loop non-blocking, so one slow or stalled handshake cannot
//! stall new connections.
//!
//! Crypto runs through whichever OpenSSL the binary links (system by default,
//! vendored when built with that feature).

use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context as _, Result};
use openssl::ssl::{Ssl, SslContext, SslContextBuilder, SslFiletype, SslMethod, SslVerifyMode};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_openssl::SslStream;

use crate::config::TlsConfig;

/// Build an OpenSSL [`SslContext`] from validated [`TlsConfig`]: the full
/// certificate chain, the private key (`key_file`, or the certificate file when
/// unset), the mTLS client verifier, and an optional cipher list.
///
/// No TLS floor or cipher list unless `--ssl-ciphers` is set; defers to the
/// linked OpenSSL's defaults.
pub(crate) fn build_server_config(tls: &TlsConfig) -> Result<SslContext> {
    let cert_file = tls.cert_file.as_deref().context("--ssl-certfile is required to enable TLS")?;

    let mut builder =
        SslContextBuilder::new(SslMethod::tls_server()).context("failed to initialize TLS")?;

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

    Ok(builder.build())
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

/// A TLS connection whose handshake is driven lazily on the first I/O poll.
pub(crate) enum MaybeTlsStream<IO> {
    Handshaking(Pin<Box<SslStream<IO>>>),
    Streaming(Pin<Box<SslStream<IO>>>),
    /// Setup failed in the infallible `accept()`; surfaced on first poll.
    SetupFailed(Option<io::Error>),
}

impl<IO: AsyncRead + AsyncWrite + Unpin> MaybeTlsStream<IO> {
    /// Drive the handshake to completion if it has not finished, returning the
    /// established TLS stream. Returns `Pending` while the handshake is in
    /// flight and propagates a handshake or setup error as an I/O error.
    fn poll_established(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<io::Result<Pin<&mut SslStream<IO>>>> {
        let this = self.get_mut();
        // Confine the borrow of the handshaking stream to this match so the enum
        // can be reassigned afterwards.
        let handshake = match this {
            MaybeTlsStream::Streaming(stream) => return Poll::Ready(Ok(stream.as_mut())),
            MaybeTlsStream::SetupFailed(err) => {
                let err =
                    err.take().unwrap_or_else(|| io::Error::other("TLS connection setup failed"));
                return Poll::Ready(Err(err));
            }
            MaybeTlsStream::Handshaking(stream) => stream.as_mut().poll_accept(cx),
        };
        match handshake {
            Poll::Ready(Ok(())) => {
                let stream = match std::mem::replace(this, MaybeTlsStream::SetupFailed(None)) {
                    MaybeTlsStream::Handshaking(stream) => stream,
                    _ => unreachable!("handshake just completed"),
                };
                *this = MaybeTlsStream::Streaming(stream);
                match this {
                    MaybeTlsStream::Streaming(stream) => Poll::Ready(Ok(stream.as_mut())),
                    _ => unreachable!("just set to Streaming"),
                }
            }
            // Stays `Handshaking` on error; re-polling OpenSSL errors, not panics.
            Poll::Ready(Err(err)) => Poll::Ready(Err(io::Error::other(err))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<IO: AsyncRead + AsyncWrite + Unpin> AsyncRead for MaybeTlsStream<IO> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.poll_established(cx) {
            Poll::Ready(Ok(stream)) => stream.poll_read(cx, buf),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<IO: AsyncRead + AsyncWrite + Unpin> AsyncWrite for MaybeTlsStream<IO> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.poll_established(cx) {
            Poll::Ready(Ok(stream)) => stream.poll_write(cx, buf),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        match self.poll_established(cx) {
            Poll::Ready(Ok(stream)) => stream.poll_flush(cx),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        match self.poll_established(cx) {
            Poll::Ready(Ok(stream)) => stream.poll_shutdown(cx),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A listener that terminates TLS on an inner axum listener. Generic so it can
/// wrap the already-tapped listener, keeping `TCP_NODELAY` on the raw TCP.
pub(crate) struct TlsListener<L> {
    inner: L,
    context: Arc<SslContext>,
}

impl<L> TlsListener<L> {
    pub(crate) fn new(inner: L, context: Arc<SslContext>) -> Self {
        Self { inner, context }
    }
}

impl<L> axum::serve::Listener for TlsListener<L>
where
    L: axum::serve::Listener,
    L::Io: AsyncRead + AsyncWrite + Unpin,
{
    type Io = MaybeTlsStream<L::Io>;
    type Addr = L::Addr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let (io, addr) = self.inner.accept().await;
        let io = match Ssl::new(&self.context).and_then(|ssl| SslStream::new(ssl, io)) {
            Ok(stream) => MaybeTlsStream::Handshaking(Box::pin(stream)),
            Err(err) => MaybeTlsStream::SetupFailed(Some(io::Error::other(err))),
        };
        (io, addr)
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}
