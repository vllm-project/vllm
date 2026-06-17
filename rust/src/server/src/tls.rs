//! TLS termination for the HTTP listener.
//!
//! Builds a rustls `ServerConfig` from the uvicorn-style `ssl_*` arguments and
//! wraps the unified [`Listener`](crate::listener::Listener) so each accepted
//! connection completes its TLS handshake lazily, inside axum's per-connection
//! task. Driving the handshake there (rather than inside `accept()`) keeps the
//! serial accept loop non-blocking, so one slow or stalled handshake cannot
//! stall new connections.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context as _, Result, bail};
use rustls::ServerConfig;
use rustls::server::WebPkiClientVerifier;
use rustls::server::danger::ClientCertVerifier;
use rustls_pki_types::pem::PemObject as _;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_rustls::server::TlsStream;
use tokio_rustls::{Accept, TlsAcceptor};

use crate::config::TlsConfig;

/// Build a rustls [`ServerConfig`] from validated [`TlsConfig`]: the full
/// certificate chain, the private key (`key_file`, or the certificate file when
/// unset), and the mTLS client verifier.
pub(crate) fn build_server_config(tls: &TlsConfig) -> Result<ServerConfig> {
    // Pin the ring provider so `ServerConfig::builder()` cannot panic if a future
    // dependency also enables rustls's aws-lc-rs feature.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_file = tls.cert_file.as_deref().context("--ssl-certfile is required to enable TLS")?;

    // Load the whole chain, not just the leaf, so deployments behind an
    // intermediate CA serve a complete chain.
    let cert_chain = CertificateDer::pem_file_iter(cert_file)
        .with_context(|| format!("failed to read --ssl-certfile {cert_file:?}"))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to parse certificates in --ssl-certfile {cert_file:?}"))?;
    if cert_chain.is_empty() {
        bail!("no certificates found in --ssl-certfile {cert_file:?}");
    }

    let key_file = tls.key_file.as_deref().unwrap_or(cert_file);
    let key = PrivateKeyDer::from_pem_file(key_file)
        .with_context(|| format!("failed to read private key from {key_file:?}"))?;

    let verifier = build_client_verifier(tls)?;

    ServerConfig::builder()
        .with_client_cert_verifier(verifier)
        .with_single_cert(cert_chain, key)
        .context("failed to build TLS server config; certificate and key may not match")
}

/// Map `cert_reqs` to a rustls client-cert verifier: 0 = none, 1 = optional
/// (verify if presented, allow anonymous), 2 = required.
fn build_client_verifier(tls: &TlsConfig) -> Result<Arc<dyn ClientCertVerifier>> {
    if tls.cert_reqs == 0 {
        return Ok(WebPkiClientVerifier::no_client_auth());
    }

    let ca_file = tls
        .ca_certs
        .as_deref()
        .context("--ssl-ca-certs is required for client certificate verification")?;
    let mut roots = rustls::RootCertStore::empty();
    for ca in CertificateDer::pem_file_iter(ca_file)
        .with_context(|| format!("failed to read --ssl-ca-certs {ca_file:?}"))?
    {
        let ca = ca.with_context(|| format!("failed to parse --ssl-ca-certs {ca_file:?}"))?;
        roots
            .add(ca)
            .with_context(|| format!("invalid CA certificate in --ssl-ca-certs {ca_file:?}"))?;
    }

    let builder = WebPkiClientVerifier::builder(Arc::new(roots));
    let builder = if tls.cert_reqs == 1 {
        builder.allow_unauthenticated()
    } else {
        builder
    };
    builder.build().context("failed to build client certificate verifier")
}

/// A TLS connection whose handshake is driven lazily on the first I/O poll.
pub(crate) enum MaybeTlsStream<IO> {
    Handshaking(Box<Accept<IO>>),
    Streaming(Box<TlsStream<IO>>),
}

impl<IO: AsyncRead + AsyncWrite + Unpin> MaybeTlsStream<IO> {
    /// Drive the handshake to completion if it has not finished, returning the
    /// established TLS stream. Returns `Pending` while the handshake is in
    /// flight and propagates a handshake error as an I/O error.
    fn poll_established(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<io::Result<Pin<&mut TlsStream<IO>>>> {
        let this = self.get_mut();
        // Confine the borrow of the `Accept` future to this match so the enum
        // can be reassigned afterwards.
        let handshake = match this {
            MaybeTlsStream::Streaming(stream) => return Poll::Ready(Ok(Pin::new(stream))),
            MaybeTlsStream::Handshaking(accept) => Pin::new(accept).poll(cx),
        };
        match handshake {
            Poll::Ready(Ok(stream)) => {
                *this = MaybeTlsStream::Streaming(Box::new(stream));
                match this {
                    MaybeTlsStream::Streaming(stream) => Poll::Ready(Ok(Pin::new(stream))),
                    MaybeTlsStream::Handshaking(_) => unreachable!("handshake just completed"),
                }
            }
            // On error the enum stays `Handshaking` over a spent `Accept`, which
            // would panic if polled again; safe because axum/hyper drops the
            // connection on an I/O error and never re-polls the stream.
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
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
    acceptor: TlsAcceptor,
}

impl<L> TlsListener<L> {
    pub(crate) fn new(inner: L, config: Arc<ServerConfig>) -> Self {
        Self {
            inner,
            acceptor: TlsAcceptor::from(config),
        }
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
        (
            MaybeTlsStream::Handshaking(Box::new(self.acceptor.accept(io))),
            addr,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}
