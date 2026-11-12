// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Serving the HTTP and gRPC listeners of the OpenAI and render servers.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::Request;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use thiserror_ext::AsReport as _;
use tower::ServiceExt as _;
use tracing::trace;

use crate::listener::ListenerAddr;

/// How often the server PINGs an idle gRPC connection to reap a dead peer;
/// tonic enables no keepalive by default. 2h matches the gRPC-core default.
pub(crate) const GRPC_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(7200);
/// How long the server waits for a keepalive PING reply before dropping the gRPC
/// connection. 20s matches the gRPC-core default.
pub(crate) const GRPC_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(20);

/// Per-connection timeouts applied while serving HTTP/HTTPS.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionTimeouts {
    /// HTTP/1 header-read timeout (bounds idle keep-alive and the head read).
    pub(crate) header_read: Duration,
    /// Whether HTTP/1 keep-alive is enabled; `false` closes after each response.
    pub(crate) keep_alive_enabled: bool,
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
