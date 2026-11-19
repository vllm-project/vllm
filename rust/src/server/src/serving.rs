// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Serving the HTTP and gRPC listeners of the OpenAI and render servers.

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context as _, Result};
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::Request;
use futures::future::BoxFuture;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use openssl::ssl::SslContext;
use thiserror_ext::AsReport as _;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tonic::transport::Server as TonicServer;
use tower::ServiceExt as _;
use tracing::{info, trace, warn};

use crate::config::{DEFAULT_KEEP_ALIVE_TIMEOUT, HttpListenerMode, TlsConfig};
use crate::grpc::HealthMonitor;
use crate::listener::{Listener, ListenerAddr, MaybeTlsListener};
use crate::tls;

/// How often the server PINGs an idle gRPC connection to reap a dead peer;
/// tonic enables no keepalive by default. 2h matches the gRPC-core default.
const GRPC_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(7200);
/// How long the server waits for a keepalive PING reply before dropping the gRPC
/// connection. 20s matches the gRPC-core default.
const GRPC_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(20);

/// How long the servers keep draining after `OnDrainDeadline` runs, so the
/// requests it ends can send their final responses. Aborted requests finish
/// within milliseconds; this only bounds a stuck engine. Matches the Python
/// launcher's `_ENGINE_TEARDOWN_S`.
const DEADLINE_HOOK_GRACE: Duration = Duration::from_secs(5);

/// Runs once when the drain deadline passes, before the servers are aborted,
/// so in-flight requests can end with a final response instead of a closed
/// connection.
pub(crate) type OnDrainDeadline = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

/// Per-connection timeouts applied while serving HTTP/HTTPS.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionTimeouts {
    /// HTTP/1 header-read timeout (bounds idle keep-alive and the head read).
    pub(crate) header_read: Duration,
    /// Whether HTTP/1 keep-alive is enabled; `false` closes after each response.
    pub(crate) keep_alive_enabled: bool,
}

impl ConnectionTimeouts {
    /// Timeouts for an idle keep-alive bound of `keep_alive_timeout`.
    pub(crate) fn for_keep_alive(keep_alive_timeout: Duration) -> Self {
        // 0 disables keep-alive but still bounds the head read (default), so a
        // silent client cannot hold the connection open.
        Self {
            header_read: if keep_alive_timeout.is_zero() {
                DEFAULT_KEEP_ALIVE_TIMEOUT
            } else {
                keep_alive_timeout
            },
            keep_alive_enabled: !keep_alive_timeout.is_zero(),
        }
    }
}

/// Serve `app` per connection (HTTP/1) with a keep-alive idle timeout and
/// graceful drain. Hand-rolled on hyper because [`axum::serve()`] takes no config.
pub(crate) async fn serve_connections<L>(
    mut listener: L,
    app: Router,
    shutdown: impl Future<Output = ()> + Send,
    timeouts: ConnectionTimeouts,
) -> Result<()>
where
    L: axum::serve::Listener<Addr = ListenerAddr>,
{
    let graceful = GracefulShutdown::new();
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        let (io, addr) = tokio::select! {
            conn = listener.accept() => conn,
            () = &mut shutdown => break,
        };

        // Expose the TCP peer to handlers, as `ConnectInfo` does for `axum::serve`.
        let peer = match addr {
            ListenerAddr::Tcp(addr) => Some(ConnectInfo(addr)),
            ListenerAddr::Unix(_) => None,
        };
        let service =
            TowerToHyperService::new(app.clone().map_request(move |req: Request<Incoming>| {
                let mut req = req.map(Body::new);
                if let Some(peer) = peer {
                    req.extensions_mut().insert(peer);
                }
                req
            }));
        let mut builder = http1::Builder::new();
        builder.timer(TokioTimer::new()).header_read_timeout(timeouts.header_read);
        if !timeouts.keep_alive_enabled {
            builder.keep_alive(false);
        }
        let connection = builder.serve_connection(TokioIo::new(io), service);
        let connection = graceful.watch(connection);

        tokio::spawn(async move {
            if let Err(err) = connection.await {
                trace!(error = %err.as_report(), "failed to serve connection");
            }
        });
    }

    drop(listener);
    graceful.shutdown().await;
    Ok(())
}

/// TLS contexts for a server's listeners, built before startup so a bad
/// certificate or key fails fast.
pub(crate) struct ListenerTls {
    pub(crate) http: Option<SslContext>,
    pub(crate) grpc: Option<SslContext>,
}

impl ListenerTls {
    /// Build the contexts for `tls`, the gRPC one only when `grpc` is served.
    pub(crate) fn build(tls: Option<&TlsConfig>, grpc: bool) -> Result<Self> {
        let Some(tls) = tls else {
            return Ok(Self {
                http: None,
                grpc: None,
            });
        };
        Ok(Self {
            http: Some(tls::build_server_config(tls).context("invalid TLS configuration")?),
            // gRPC reuses the HTTP TLS config (same SslContext) plus ALPN h2.
            grpc: grpc
                .then(|| tls::build_grpc_server_config(tls))
                .transpose()
                .context("invalid gRPC TLS configuration")?,
        })
    }
}

/// A bound listener, terminating TLS when configured.
pub(crate) struct BoundListener {
    /// The bound address, for logs.
    address: String,
    listener: MaybeTlsListener,
    tls: bool,
}

impl BoundListener {
    /// Bind `mode` now, so bind errors surface before serving.
    pub(crate) async fn bind(mode: &HttpListenerMode, tls: Option<SslContext>) -> Result<Self> {
        let listener = Listener::bind(mode).await?;
        let address = listener.local_addr_display()?;
        Ok(Self {
            address,
            tls: tls.is_some(),
            listener: match tls {
                Some(context) => MaybeTlsListener::tls(listener, context),
                None => MaybeTlsListener::plain(listener),
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn address(&self) -> &str {
        &self.address
    }
}

/// A server's HTTP side: its listener and router.
pub(crate) struct HttpServer {
    pub(crate) listener: BoundListener,
    pub(crate) app: Router,
    pub(crate) timeouts: ConnectionTimeouts,
}

/// Serves gRPC services on a listener until the given signal.
pub(crate) type GrpcServe = Box<
    dyn FnOnce(
            MaybeTlsListener,
            WaitForCancellationFutureOwned,
        ) -> futures::future::BoxFuture<'static, Result<(), tonic::transport::Error>>
        + Send,
>;

/// A server's gRPC side: its listener, the health services it publishes, and
/// its services, which include the health service.
pub(crate) struct GrpcServer {
    pub(crate) listener: BoundListener,
    pub(crate) health: HealthMonitor,
    pub(crate) serve: GrpcServe,
}

/// A tonic server builder with the gRPC-core keepalive defaults.
pub(crate) fn grpc_server_builder() -> TonicServer {
    TonicServer::builder()
        .http2_keepalive_interval(Some(GRPC_KEEPALIVE_INTERVAL))
        .http2_keepalive_timeout(Some(GRPC_KEEPALIVE_TIMEOUT))
}

impl GrpcServer {
    /// Serve until `server_shutdown`, withdrawing the health services before
    /// the listener stops. If the server exits on its own, cancel
    /// `server_shutdown` so its sibling drains too. Returns `None` if `force`
    /// aborted the server.
    pub(crate) async fn run(
        self,
        server_shutdown: CancellationToken,
        force: impl Future<Output = ()>,
    ) -> Option<Result<()>> {
        let Self {
            listener,
            health,
            serve,
        } = self;
        health
            .serve(server_shutdown.child_token(), force, |stop| async move {
                let result = serve(listener.listener, stop).await.context("gRPC server failed");
                server_shutdown.cancel();
                result
            })
            .await
    }
}

/// Serve `http` and, when set, `grpc` until `shutdown`.
///
/// Caller cancellation propagates into both protocols; if either protocol
/// exits first, the other begins a graceful drain too. Once shutdown starts,
/// both drain for at most `shutdown_timeout`. Then `on_drain_deadline`, when
/// set, runs and the servers drain for up to `DEADLINE_HOOK_GRACE` more before
/// they are aborted. Returns the time the servers are aborted at the latest,
/// which the caller's own cleanup should also respect.
pub(crate) async fn serve(
    name: &str,
    model: &str,
    http: HttpServer,
    grpc: Option<GrpcServer>,
    shutdown_timeout: Duration,
    on_drain_deadline: Option<OnDrainDeadline>,
    shutdown: CancellationToken,
) -> Result<Instant> {
    // Run HTTP and gRPC concurrently under a child token of the caller's shutdown
    // token. Caller cancellation propagates into both protocols; if either
    // protocol exits first, we cancel this child token so its sibling also
    // begins a graceful drain.
    let server_shutdown = shutdown.child_token();
    let force_shutdown = CancellationToken::new();
    let shutdown_deadline = Arc::new(OnceLock::new());

    // Spawn a task to trigger `force_shutdown` after shutdown deadline elapses.
    tokio::spawn({
        let shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        let shutdown_deadline = shutdown_deadline.clone();

        async move {
            shutdown.cancelled().await;
            let deadline = Instant::now() + shutdown_timeout;
            // With a hook, the servers keep draining while the requests it
            // ends send their final responses, so the caller's cleanup waits
            // for that too.
            let abort_at = if on_drain_deadline.is_some() {
                deadline + DEADLINE_HOOK_GRACE
            } else {
                deadline
            };
            let _ = shutdown_deadline.set(abort_at);

            if !shutdown_timeout.is_zero() {
                sleep_until(deadline).await;
            }
            if let Some(on_drain_deadline) = on_drain_deadline {
                on_drain_deadline().await;
                sleep_until(abort_at).await;
            }
            force_shutdown.cancel();
        }
    });

    let http_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let HttpServer {
                listener,
                app,
                timeouts,
            } = http;
            let scheme = if listener.tls { "https" } else { "http" };
            let bind_address = listener.address;
            let server =
                serve_connections(listener.listener, app, shutdown.cancelled_owned(), timeouts);

            info!(
                bind_address,
                scheme, model, "{name} is ready to accept requests"
            );

            let result = tokio::select! {
                result = server => {
                    result.context("HTTP server failed")
                }
                _ = force_shutdown.cancelled() => {
                    warn!("HTTP graceful shutdown deadline elapsed; aborting server");
                    Ok(())
                }
            };

            server_shutdown.cancel();
            result
        }
    };

    let grpc_fut = {
        let shutdown = server_shutdown.child_token();
        let server_shutdown = server_shutdown.clone();
        let force_shutdown = force_shutdown.clone();
        async move {
            let Some(grpc) = grpc else {
                // No gRPC configured: just wait for shutdown so we do not race the
                // join! by resolving early and tripping the cancellation token.
                shutdown.cancelled().await;
                return Ok(());
            };
            let addr = grpc.listener.address.clone();
            let tls = grpc.listener.tls;
            let server = grpc.run(server_shutdown, force_shutdown.cancelled());

            info!(%addr, tls, model, "gRPC server is ready to accept requests");

            server.await.unwrap_or_else(|| {
                warn!("gRPC graceful shutdown deadline elapsed; aborting server");
                Ok(())
            })
        }
    };

    let (http_res, grpc_res) = tokio::join!(http_fut, grpc_fut);
    http_res.and(grpc_res)?;

    Ok(shutdown_deadline
        .get()
        .copied()
        .unwrap_or_else(|| Instant::now() + shutdown_timeout))
}

#[cfg(test)]
mod tests {
    use axum::routing::get;
    use futures::FutureExt as _;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpStream;
    use tokio::sync::Notify;

    use super::*;

    #[tokio::test]
    async fn drain_deadline_hook_lets_in_flight_request_finish() {
        // A request that waits until released, like one still generating at the
        // deadline; the hook releases it, like the abort the OpenAI server runs.
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let app = Router::new().route(
            "/wait",
            get({
                let entered = entered.clone();
                let release = release.clone();
                move || {
                    let entered = entered.clone();
                    let release = release.clone();
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        "done"
                    }
                }
            }),
        );
        let hook: OnDrainDeadline = {
            let release = release.clone();
            Box::new(move || async move { release.notify_one() }.boxed())
        };

        let listener = BoundListener::bind(
            &HttpListenerMode::BindTcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            None,
        )
        .await
        .expect("bind");
        let addr = listener.address().to_owned();
        let http = HttpServer {
            listener,
            app,
            timeouts: ConnectionTimeouts::for_keep_alive(Duration::ZERO),
        };
        let drain = Duration::from_millis(200);
        let shutdown = CancellationToken::new();
        let server = tokio::spawn(serve(
            "test server",
            "test-model",
            http,
            None,
            drain,
            Some(hook),
            shutdown.clone(),
        ));

        let mut stream = TcpStream::connect(&addr).await.expect("connect");
        stream
            .write_all(b"GET /wait HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .await
            .expect("write request");
        entered.notified().await;

        let started = Instant::now();
        shutdown.cancel();

        let mut response = String::new();
        tokio::time::timeout(Duration::from_secs(3), stream.read_to_string(&mut response))
            .await
            .expect("the request must finish once the hook runs")
            .expect("read response");
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.ends_with("done"), "{response}");
        // Released by the hook at the deadline, not earlier.
        assert!(started.elapsed() >= drain, "{:?}", started.elapsed());

        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .expect("the servers stop once the request finishes, before the grace ends")
            .expect("join")
            .expect("serve");
    }
}
