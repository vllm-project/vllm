//! Unified listener wrapper for the Rust frontend.
//!
//! This module hides the difference between TCP and Unix-domain listeners so
//! the rest of the server can bind or inherit one socket and pass it to
//! `axum::serve(...)` through a single type.

use std::io::Result;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::net::UnixListener as StdUnixListener;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use auto_enums::enum_derive;
use openssl::ssl::SslContext;
use socket2::Socket;
use tls_listener::{AsyncAccept, AsyncListener};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tonic::transport::server::{Connected, TcpConnectInfo};
use tracing::trace;

use crate::{HttpListenerMode, tls};

/// Runtime listener type used by the OpenAI-compatible HTTP or gRPC server,
/// which is either a TCP listener or a Unix-domain listener.
#[derive(Debug)]
pub enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener),
}

/// Runtime listener I/O type which is either a TCP stream or a Unix-domain stream.
#[derive(Debug)]
#[enum_derive(tokio1::AsyncRead, tokio1::AsyncWrite)]
pub enum ListenerIo {
    Tcp(TcpStream),
    Unix(UnixStream),
}

/// Runtime listener address type which is either a TCP address or a Unix-domain address.
#[derive(Debug)]
#[allow(dead_code)]
pub enum ListenerAddr {
    Tcp(SocketAddr),
    Unix(tokio::net::unix::SocketAddr),
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
            HttpListenerMode::BindUnix { path } => Ok(Self::Unix(UnixListener::bind(path)?)),
            HttpListenerMode::InheritedFd { fd } => Self::from_inherited_fd(*fd),
        }
    }

    /// Return a log-friendly local address string for either TCP or Unix
    /// sockets.
    pub fn local_addr_display(&self) -> Result<String> {
        match self {
            Self::Tcp(listener) => Ok(listener.local_addr()?.to_string()),
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

    fn local_addr(&self) -> Result<ListenerAddr> {
        match self {
            Self::Tcp(listener) => listener.local_addr().map(ListenerAddr::Tcp),
            Self::Unix(listener) => listener.local_addr().map(ListenerAddr::Unix),
        }
    }
}

/// Allow the unified listener to plug directly into tonic's gRPC server.
impl Connected for ListenerIo {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> TcpConnectInfo {
        match self {
            Self::Tcp(stream) => stream.connect_info(),
            Self::Unix(_) => TcpConnectInfo {
                local_addr: None,
                remote_addr: None,
            },
        }
    }
}

/// Attempt to set `TCP_NODELAY` on the accepted TCP stream.
fn enable_tcp_nodelay(stream: TcpStream) -> TcpStream {
    if let Err(err) = stream.set_nodelay(true) {
        trace!(error = %err, "failed to enable TCP_NODELAY on accepted TCP connection");
    }
    stream
}

/// Allow the unified listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for Listener {
    type Addr = ListenerAddr;
    type Io = ListenerIo;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self {
            Self::Tcp(listener) => {
                let (io, addr) = axum::serve::Listener::accept(listener).await;
                (
                    ListenerIo::Tcp(enable_tcp_nodelay(io)),
                    ListenerAddr::Tcp(addr),
                )
            }
            Self::Unix(listener) => {
                let (io, addr) = axum::serve::Listener::accept(listener).await;
                (ListenerIo::Unix(io), ListenerAddr::Unix(addr))
            }
        }
    }

    fn local_addr(&self) -> Result<Self::Addr> {
        self.local_addr()
    }
}

/// Allow the unified listener to be adaptable to `tls_listener`.
impl AsyncAccept for Listener {
    type Connection = ListenerIo;
    type Address = ListenerAddr;
    type Error = std::io::Error;

    fn poll_accept(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(Self::Connection, Self::Address)>> {
        match self.get_mut() {
            Self::Tcp(listener) => {
                let (io, addr) = ready!(listener.poll_accept(cx))?;
                Poll::Ready(Ok((
                    ListenerIo::Tcp(enable_tcp_nodelay(io)),
                    ListenerAddr::Tcp(addr),
                )))
            }
            Self::Unix(listener) => {
                let (io, addr) = ready!(listener.poll_accept(cx))?;
                Poll::Ready(Ok((ListenerIo::Unix(io), ListenerAddr::Unix(addr))))
            }
        }
    }
}
impl AsyncListener for Listener {
    fn local_addr(&self) -> Result<Self::Address> {
        self.local_addr()
    }
}

/// A listener that may be either a plain TCP/UDS listener or a TLS listener over it.
pub enum MaybeTlsListener {
    Plain(Listener),
    Tls(tls_listener::TlsListener<Listener, SslContext>),
}

impl MaybeTlsListener {
    /// Create a plain listener without TLS.
    pub fn plain(listener: Listener) -> Self {
        Self::Plain(listener)
    }

    /// Create a TLS listener over the given plain listener.
    pub fn tls(listener: Listener, context: SslContext) -> Self {
        Self::Tls(
            tls_listener::builder(context)
                .handshake_timeout(tls::TLS_HANDSHAKE_TIMEOUT)
                .listen(listener),
        )
    }
}

/// Listener I/O type that may be either a plain TCP/UDS stream or a TLS stream over it.
#[derive(Debug)]
#[enum_derive(tokio1::AsyncRead, tokio1::AsyncWrite)]
pub enum MaybeTlsStream {
    Plain(ListenerIo),
    Tls(tokio_openssl::SslStream<ListenerIo>),
}

/// Allow the maybe-TLS listener to plug directly into `axum::serve(...)`.
impl axum::serve::Listener for MaybeTlsListener {
    type Addr = ListenerAddr;
    type Io = MaybeTlsStream;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self {
            Self::Plain(listener) => {
                let (io, addr) = axum::serve::Listener::accept(listener).await;
                (MaybeTlsStream::Plain(io), addr)
            }
            Self::Tls(tls_listener) => {
                let (io, addr) = axum::serve::Listener::accept(tls_listener).await;
                (MaybeTlsStream::Tls(io), addr)
            }
        }
    }

    fn local_addr(&self) -> tokio::io::Result<Self::Addr> {
        match self {
            Self::Plain(listener) => listener.local_addr(),
            Self::Tls(tls_listener) => tls_listener.local_addr(),
        }
    }
}

/// Allow the maybe-TLS listener to plug directly into tonic's gRPC server.
impl Connected for MaybeTlsStream {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> TcpConnectInfo {
        match self {
            Self::Plain(stream) => stream.connect_info(),
            Self::Tls(stream) => stream.get_ref().connect_info(),
        }
    }
}

/// Allow the maybe-TLS listener to be adaptable to tonic's incoming stream shape.
impl futures::Stream for MaybeTlsListener {
    type Item = std::io::Result<MaybeTlsStream>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.get_mut() {
            Self::Plain(listener) => {
                let listener = Pin::new(listener);
                let (io, _) = ready!(listener.poll_accept(cx))?;
                Poll::Ready(Some(Ok(MaybeTlsStream::Plain(io))))
            }
            Self::Tls(tls_listener) => {
                let tls_listener = Pin::new(tls_listener);
                let (io, _) =
                    ready!(tls_listener.poll_accept(cx)).map_err(std::io::Error::other)?;
                Poll::Ready(Some(Ok(MaybeTlsStream::Tls(io))))
            }
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
