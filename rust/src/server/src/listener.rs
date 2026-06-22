//! Unified HTTP listener wrapper for the Rust frontend.
//!
//! This module hides the difference between TCP and Unix-domain listeners so
//! the rest of the server can bind or inherit one socket and pass it to
//! `axum::serve(...)` through a single type.

use std::fmt;
use std::io::Result;
use std::net::TcpListener as StdTcpListener;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::net::UnixListener as StdUnixListener;
use std::sync::Arc;

use socket2::Socket;
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio_rustls::TlsAcceptor;
use tokio_util::either::Either;
use tracing::{info, warn};

use crate::HttpListenerMode;

/// TLS-wrapped TCP stream for use with tokio-rustls
pub type TlsStream = tokio_rustls::server::TlsStream<TcpStream>;

/// Runtime listener type used by the OpenAI-compatible HTTP server, which is
/// either a TCP listener, a TLS-enabled TCP listener, or a Unix-domain listener.
pub enum Listener {
    Tcp(TcpListener),
    TcpTls(TcpListener, Arc<TlsAcceptor>),
    Unix(UnixListener),
}

impl fmt::Debug for Listener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tcp(listener) => f.debug_tuple("Tcp").field(listener).finish(),
            Self::TcpTls(listener, _) => f.debug_tuple("TcpTls").field(listener).field(&"<TlsAcceptor>").finish(),
            Self::Unix(listener) => f.debug_tuple("Unix").field(listener).finish(),
        }
    }
}

impl Listener {
    /// Bind or adopt the listener described by the frontend configuration.
    ///
    /// For inherited sockets, the concrete listener kind is detected from the
    /// socket family of the supplied file descriptor.
    pub async fn bind(mode: &HttpListenerMode) -> Result<Self> {
        
        match mode {
            HttpListenerMode::BindTcp { host, port } => {
                Ok(Self::Tcp(TcpListener::bind((host.as_str(), *port)).await?))
            }
            HttpListenerMode::BindTcpTls {
                host,
                port,
                cert_path,
                key_path,
                ca_certs_path,
                ssl_cert_reqs,
                ssl_ciphers,
            } => {
                let tcp_listener = TcpListener::bind((host.as_str(), *port)).await?;
                let tls_acceptor = Self::load_tls_acceptor(cert_path, key_path, ca_certs_path.as_deref(), *ssl_cert_reqs, ssl_ciphers.as_deref())?;
                info!("TLS Listener bound to {}:{}", host, port);
                Ok(Self::TcpTls(tcp_listener, Arc::new(tls_acceptor)))
            }
            HttpListenerMode::BindUnix { path } => Ok(Self::Unix(UnixListener::bind(path)?)),
            HttpListenerMode::InheritedFd { fd } => Self::from_inherited_fd(*fd),
        }
    }

    /// Validate and log SSL cipher suites for CLI compatibility.
    /// 
    /// IMPORTANT: This is VALIDATION AND LOGGING ONLY. Rustls uses hardcoded secure-by-default
    /// cipher suites and provides no API to restrict or configure which ciphers are used.
    /// Any ciphers specified in ssl_ciphers are validated and logged, but NOT enforced.
    /// The server will always use all rustls default ciphers regardless of this parameter.
    /// 
    /// This function exists for CLI compatibility with Python/Uvicorn, which do support
    /// cipher configuration. Unsupported/weak ciphers are logged as warnings but do not
    /// cause errors - the server continues to use rustls secure defaults.
    fn validate_ssl_ciphers(ciphers_str: &str) -> Result<()> {
        // Rustls (v0.23 with aws-lc-rs) supports only these modern secure cipher suites:
        // These are the actual internal names rustls uses.
        let supported_ciphers = [
            // TLS 1.3 cipher suites (rustls internal names)
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256",
            // TLS 1.3 cipher suites (OpenSSL-compatible aliases)
            "TLS13-AES-256-GCM-SHA384",
            "TLS13-CHACHA20-POLY1305-SHA256",
            "TLS13-AES-128-GCM-SHA256",
            // TLS 1.2 ECDHE cipher suites (ECDSA) - rustls internal names
            "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
            "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
            // TLS 1.2 ECDHE cipher suites (ECDSA) - OpenSSL-compatible aliases
            "ECDHE-ECDSA-AES256-GCM-SHA384",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
            "ECDHE-ECDSA-CHACHA20-POLY1305",
            // TLS 1.2 ECDHE cipher suites (RSA) - rustls internal names
            "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            // TLS 1.2 ECDHE cipher suites (RSA) - OpenSSL-compatible aliases
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES128-GCM-SHA256",
            "ECDHE-RSA-CHACHA20-POLY1305",
        ];

        let mut unsupported = Vec::new();
        let mut supported = Vec::new();

        for cipher in ciphers_str.split(':').filter(|c| !c.is_empty()) {
            if supported_ciphers.contains(&cipher) {
                supported.push(cipher);
            } else {
                unsupported.push(cipher);
            }
        }

        if !unsupported.is_empty() {
            warn!(
                "ssl_ciphers contains {} cipher suite(s) that are not supported or recognized by rustls and will be ignored: {}",
                unsupported.len(),
                unsupported.join(", ")
            );
        }

        if !supported.is_empty() {
            info!(
                "ssl_ciphers contains {} supported cipher suite(s): {}",
                supported.len(),
                supported.join(", ")
            );
        }

        // Always inform user that rustls uses secure defaults and doesn't allow weak ciphers
        info!(
            "Rustls uses hardcoded secure-by-default cipher suites and does not allow configuration \
            of weak or legacy ciphers. The ssl_ciphers parameter is accepted for CLI compatibility \
            with Python/Uvicorn but is not enforced."
        );

        Ok(())
    }

    /// Load TLS configuration from certificate and key files.
    /// Optionally loads CA certificates for client authentication.
    /// ssl_cert_reqs: 0=CERT_NONE (no client auth), 1=CERT_OPTIONAL (optional client auth), 2=CERT_REQUIRED (mandatory client auth)
    /// ssl_ciphers: VALIDATION AND LOGGING ONLY. Rustls uses hardcoded secure defaults and provides no API
    /// to restrict or configure cipher suites. Any specified ciphers are validated and logged but NOT enforced.
    /// The server will always use all rustls default ciphers regardless of this parameter.
    fn load_tls_acceptor(cert_path: &str, key_path: &str, ca_certs_path: Option<&str>, ssl_cert_reqs: u32, ssl_ciphers: Option<&str>) -> Result<TlsAcceptor> {
        use rustls_pemfile::{certs, private_key};
        use std::fs;
        use std::io::BufReader;

        // Validate and log ssl_ciphers parameter
        if let Some(ciphers) = ssl_ciphers {
            Self::validate_ssl_ciphers(ciphers)?;
        }

        // Load certificate chain
        let cert_file = fs::File::open(cert_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to open cert file {}: {}", cert_path, e)))?;
        let mut reader = BufReader::new(cert_file);
        let cert_chain: Vec<_> = certs(&mut reader)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to parse certificates from {}: {}", cert_path, e)))?;

        if cert_chain.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("No certificates found in {}", cert_path),
            ));
        }
        info!("Loaded {} certificate(s) from {}", cert_chain.len(), cert_path);

        // Load private key (supports PKCS8, RSA, EC, and Ed25519 formats)
        // Note: rustls_pemfile's private_key function automatically detects the key type and supports multiple formats, so we don't need to manually try different parsers.
        // Also the uvicorn TLS implementation supports DSA and DH keys but rustls does not, so we don't attempt to parse those formats and will error if they're used.
        let key_file = fs::File::open(key_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to open key file {}: {}", key_path, e)))?;
        let mut reader = BufReader::new(key_file);
        let private_key = private_key(&mut reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to parse private key from {}: {}", key_path, e)))?
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("No private key found in {}. Supported formats: PKCS8, RSA, EC, Ed25519", key_path),
            ))?;

        info!("Loaded private key from {}", key_path);

        // Create TLS server config with client certificate verification
        // ssl_cert_reqs: 0=CERT_NONE, 2=CERT_REQUIRED
        // Note: CERT_OPTIONAL (1) is not supported by rustls v0.23 and will be treated as CERT_NONE
        let server_config = match ssl_cert_reqs {
            0 => {
                // CERT_NONE: No client certificate verification
                info!("Client certificates not required (CERT_NONE)");
                tokio_rustls::rustls::ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(cert_chain, private_key)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Invalid TLS config: {}", e)))?
            }
            1 => {
                // CERT_OPTIONAL: Not supported in rustls v0.23 - fallback to CERT_NONE
                warn!("ssl_cert_reqs=1 (CERT_OPTIONAL) is not supported. Falling back to CERT_NONE (client certificates not required).");
                tokio_rustls::rustls::ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(cert_chain, private_key)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Invalid TLS config: {}", e)))?
            }
            2 => {
                // CERT_REQUIRED: Client certificates are required
                if let Some(ca_path) = ca_certs_path {
                    info!("Client certificates required (CERT_REQUIRED)");
                    // Load CA certificates for required client authentication
                    let ca_file = std::fs::File::open(ca_path)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to open CA certs file {}: {}", ca_path, e)))?;
                    let mut ca_reader = std::io::BufReader::new(ca_file);
                    let ca_certs: Vec<_> = certs(&mut ca_reader)
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to parse CA certificates from {}: {}", ca_path, e)))?;

                    if ca_certs.is_empty() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("No CA certificates found in {}", ca_path),
                        ));
                    }
                    info!("Loaded {} CA certificate(s) from {}", ca_certs.len(), ca_path);

                    // Build root cert store
                    let mut root_store = tokio_rustls::rustls::RootCertStore::empty();
                    for cert in ca_certs {
                        root_store.add(cert)
                            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Invalid CA certificate: {}", e)))?;
                    }

                    // Use builder API to create required client verifier
                    let client_verifier = tokio_rustls::rustls::server::WebPkiClientVerifier::builder(std::sync::Arc::new(root_store))
                        .build()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to build client verifier: {}", e)))?;

                    tokio_rustls::rustls::ServerConfig::builder()
                        .with_client_cert_verifier(client_verifier)
                        .with_single_cert(cert_chain, private_key)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("Invalid TLS config with required client auth: {}", e)))?
                } else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "CERT_REQUIRED specified but no CA certificates provided".to_string(),
                    ));
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Invalid ssl_cert_reqs value: {}. Supported values: 0 (CERT_NONE), 2 (CERT_REQUIRED). Note: 1 (CERT_OPTIONAL) is not supported and will fallback to CERT_NONE.", ssl_cert_reqs),
                ));
            }
        };

        Ok(TlsAcceptor::from(Arc::new(server_config)))
    }

    /// Return a log-friendly local address string for either TCP or Unix
    /// sockets.
    pub fn local_addr(&self) -> Result<String> {
        match self {
            Self::Tcp(listener) => Ok(listener.local_addr()?.to_string()),
            Self::TcpTls(listener, _) => Ok(listener.local_addr()?.to_string()),
            Self::Unix(listener) => Ok(match listener.local_addr()?.as_pathname() {
                Some(path) => format!("unix:{}", path.display()),
                None => "unix:<unnamed>".to_string(),
            }),
        }
    }

    fn from_inherited_fd(fd: i32) -> Result<Self> {
        // SAFETY: We trust the caller to only pass valid listener fds, and we only use
        // this fd once to create a single listener.
        let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let socket = Socket::from(owned_fd);

        // The Python supervisor pre-binds the socket to reserve the endpoint early, but
        // Rust is responsible for transitioning inherited stream sockets into
        // the listening state before accepting connections.
        socket.listen(libc::SOMAXCONN)?;
        socket.set_nonblocking(true)?;

        if socket.local_addr()?.is_unix() {
            let std_listener = unsafe { StdUnixListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Unix(UnixListener::from_std(std_listener)?))
        } else {
            let std_listener = unsafe { StdTcpListener::from_raw_fd(socket.into_raw_fd()) };
            Ok(Self::Tcp(TcpListener::from_std(std_listener)?))
        }
    }
}

/// IO stream type that can be either plain TCP, TLS-wrapped TCP, or Unix domain.
pub enum ListenerIo {
    Tcp(TcpStream),
    TcpTls(TlsStream),
    Unix(UnixStream),
}

impl tokio::io::AsyncRead for ListenerIo {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match &mut *self {
            ListenerIo::Tcp(stream) => std::pin::Pin::new(stream).poll_read(cx, buf),
            ListenerIo::TcpTls(stream) => std::pin::Pin::new(stream).poll_read(cx, buf),
            ListenerIo::Unix(stream) => std::pin::Pin::new(stream).poll_read(cx, buf),
        }
    }
}

impl tokio::io::AsyncWrite for ListenerIo {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match &mut *self {
            ListenerIo::Tcp(stream) => std::pin::Pin::new(stream).poll_write(cx, buf),
            ListenerIo::TcpTls(stream) => std::pin::Pin::new(stream).poll_write(cx, buf),
            ListenerIo::Unix(stream) => std::pin::Pin::new(stream).poll_write(cx, buf),
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match &mut *self {
            ListenerIo::Tcp(stream) => std::pin::Pin::new(stream).poll_flush(cx),
            ListenerIo::TcpTls(stream) => std::pin::Pin::new(stream).poll_flush(cx),
            ListenerIo::Unix(stream) => std::pin::Pin::new(stream).poll_flush(cx),
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match &mut *self {
            ListenerIo::Tcp(stream) => std::pin::Pin::new(stream).poll_shutdown(cx),
            ListenerIo::TcpTls(stream) => std::pin::Pin::new(stream).poll_shutdown(cx),
            ListenerIo::Unix(stream) => std::pin::Pin::new(stream).poll_shutdown(cx),
        }
    }
}

/// Allow the unified listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for Listener {
    type Addr = Either<std::net::SocketAddr, tokio::net::unix::SocketAddr>;
    type Io = ListenerIo;

    #[allow(refining_impl_trait)]
    fn accept(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = (Self::Io, Self::Addr)> + Send + '_>> {
        Box::pin(async {
            match self {
                Self::Tcp(listener) => {
                    let (io, addr) = listener.accept().await;
                    (ListenerIo::Tcp(io), Either::Left(addr))
                }
                Self::TcpTls(listener, tls_acceptor) => {
                    let (io, addr) = listener.accept().await;
                    // Perform TLS handshake
                    match tls_acceptor.accept(io).await {
                        Ok(tls_io) => (ListenerIo::TcpTls(tls_io), Either::Left(addr)),
                        Err(e) => {
                            // Log the TLS error for debugging
                            warn!("TLS handshake failed from {}: {}", addr, e);
                            // On TLS error, recursively accept the next connection instead
                            // This prevents the server from crashing on TLS errors
                            self.accept().await
                        }
                    }
                }
                Self::Unix(listener) => {
                    let (io, addr) = listener.accept().await;
                    (ListenerIo::Unix(io), Either::Right(addr))
                }
            }
        })
    }

    fn local_addr(&self) -> Result<Self::Addr> {
        match self {
            Self::Tcp(listener) => listener.local_addr().map(Either::Left),
            Self::TcpTls(listener, _) => listener.local_addr().map(Either::Left),
            Self::Unix(listener) => listener.local_addr().map(Either::Right),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddrV4};
    use std::os::fd::IntoRawFd;

    use socket2::{Domain, SockAddr, Socket, Type};
    use uuid::Uuid;

    use super::Listener;
    use crate::HttpListenerMode;

    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_detects_tcp_listener_without_uds_hint() {
        let socket = Socket::new(Domain::IPV4, Type::STREAM, None).unwrap();
        socket.bind(&SockAddr::from(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let fd = socket.into_raw_fd();

        let listener = Listener::bind(&HttpListenerMode::InheritedFd { fd }).await.unwrap();

        assert!(matches!(listener, Listener::Tcp(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inherited_fd_detects_unix_listener_from_fd() {
        let path = std::env::temp_dir().join(format!("vllm-rs-{}.sock", Uuid::new_v4()));
        let socket = Socket::new(Domain::UNIX, Type::STREAM, None).unwrap();
        socket.bind(&SockAddr::unix(&path).unwrap()).unwrap();
        let fd = socket.into_raw_fd();

        let listener = Listener::bind(&HttpListenerMode::InheritedFd { fd }).await.unwrap();

        assert!(matches!(listener, Listener::Unix(_)));
        let _ = std::fs::remove_file(path);
    }
}
