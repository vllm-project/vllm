//! Hot-reload of the TLS certificate material (`--enable-ssl-refresh`), mirroring
//! Python's `SSLCertRefresher`: when the cert/key/CA files change on disk, new
//! handshakes serve the new material without a restart.
//!
//! OpenSSL's [`SslContext`] is immutable once built, so a change rebuilds it and
//! swaps it into a [`ReloadableTls`] cell read per handshake; in-flight
//! connections keep the context they negotiated with.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use notify::{RecursiveMode, Watcher};
use openssl::ssl::SslContext;
use tls_listener::AsyncTls;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::task::AbortOnDropHandle;
use tracing::{info, warn};

use crate::config::TlsConfig;
use crate::tls;

/// Event-grouping windows mirroring watchfiles' defaults: reload once `STEP`
/// passes with no new event, but never group longer than `DEBOUNCE` in total.
const STEP: Duration = Duration::from_millis(50);
const DEBOUNCE: Duration = Duration::from_millis(1600);

/// A swappable OpenSSL context: the acceptor reads the current context per
/// handshake, so a reloader can install a rebuilt one at runtime.
#[derive(Clone)]
pub(crate) struct ReloadableTls(Arc<ArcSwap<SslContext>>);

impl ReloadableTls {
    pub(crate) fn new(context: SslContext) -> Self {
        Self(Arc::new(ArcSwap::from_pointee(context)))
    }

    fn store(&self, context: SslContext) {
        self.0.store(Arc::new(context));
    }
}

/// Delegate to the crate's own `SslContext` acceptor. `accept` snapshots the
/// context into a fresh `Ssl` and returns an owned future, so the loaded handle
/// may drop.
impl<C> AsyncTls<C> for ReloadableTls
where
    C: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = tokio_openssl::SslStream<C>;
    type Error = openssl::ssl::Error;
    type AcceptFuture = <SslContext as AsyncTls<C>>::AcceptFuture;

    fn accept(&self, conn: C) -> Self::AcceptFuture {
        self.0.load_full().accept(conn)
    }
}

/// Spawn the certificate watcher; dropping the returned handle (on shutdown)
/// aborts it.
pub(crate) fn spawn_cert_reloader(
    tls: TlsConfig,
    http: ReloadableTls,
    grpc: Option<ReloadableTls>,
) -> AbortOnDropHandle<()> {
    AbortOnDropHandle::new(tokio::spawn(run(tls, http, grpc)))
}

async fn run(tls: TlsConfig, http: ReloadableTls, grpc: Option<ReloadableTls>) {
    let files = watched_files(&tls);
    let dirs = watched_dirs(&files);

    // Watch the parent directories: a symlink-swap rotation changes
    // the leaf inode, which a file watch would stop following.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut watcher = match notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    }) {
        Ok(watcher) => watcher,
        Err(err) => {
            warn!(error = %err, "failed to start the SSL certificate watcher; hot-reload disabled");
            return;
        }
    };
    let mut watching = false;
    for dir in &dirs {
        match watcher.watch(dir, RecursiveMode::NonRecursive) {
            Ok(()) => watching = true,
            Err(err) => {
                warn!(error = %err, dir = %dir.display(), "failed to watch SSL certificate directory")
            }
        }
    }
    if !watching {
        warn!("no SSL certificate directory could be watched; hot-reload is inactive");
        return;
    }
    info!(?files, "watching SSL certificate files for changes");

    // Any event triggers a whole-context rebuild, so we never inspect which file changed.
    while let Some(item) = rx.recv().await {
        log_watch_error(&item);
        // Coalesce the burst a rotation emits into one reload.
        let ceiling = tokio::time::sleep(DEBOUNCE);
        tokio::pin!(ceiling);
        loop {
            tokio::select! {
                _ = tokio::time::sleep(STEP) => break,
                _ = &mut ceiling => break,
                item = rx.recv() => match item {
                    Some(item) => log_watch_error(&item),
                    None => return,
                },
            }
        }
        reload(&tls, &http, grpc.as_ref());
    }
}

/// Log a watcher-reported error (e.g. inotify overflow) instead of dropping it.
/// The reload that follows re-reads the files, so an error is noted, not fatal.
fn log_watch_error(item: &notify::Result<notify::Event>) {
    if let Err(err) = item {
        warn!(error = %err, "SSL certificate watcher reported an error");
    }
}

/// Rebuild the active TLS context(s) and swap them in, all-or-nothing: if any
/// rebuild fails (a partial or malformed write), keep the current certificate
/// and log, never serving a half-updated pair.
pub(crate) fn reload(tls: &TlsConfig, http: &ReloadableTls, grpc: Option<&ReloadableTls>) {
    let http_context = match tls::build_server_config(tls) {
        Ok(context) => context,
        Err(err) => return keep_current(err),
    };
    let grpc_context = match grpc.map(|_| tls::build_grpc_server_config(tls)).transpose() {
        Ok(context) => context,
        Err(err) => return keep_current(err),
    };

    http.store(http_context);
    if let (Some(cell), Some(context)) = (grpc, grpc_context) {
        cell.store(context);
    }
    info!("Reloaded SSL certificate chain");
}

fn keep_current(err: anyhow::Error) {
    warn!(error = %err, "SSL certificate reload failed; keeping the current certificate");
}

/// The certificate files to watch: the cert chain (always), plus the key and CA
/// when configured separately.
fn watched_files(tls: &TlsConfig) -> Vec<PathBuf> {
    [
        tls.cert_file.as_deref(),
        tls.key_file.as_deref(),
        tls.ca_certs.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(PathBuf::from)
    .collect()
}

fn watched_dirs(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = files
        .iter()
        .map(|file| match file.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        })
        .collect();
    dirs.sort();
    dirs.dedup();
    dirs
}
