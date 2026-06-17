//! TLS tests: `build_server_config` unit checks plus end-to-end handshakes over
//! a real socket, driving the production `serve_listener` path. A trivial router
//! stands in for the full app since TLS termination is app-agnostic. Certs are
//! generated at test time (see `TestCerts`), so nothing is committed; the client
//! is tokio-rustls (not reqwest) for full control over client-certificate
//! behavior.

use std::sync::Arc;

use axum::Router;
use axum::routing::get;
use rcgen::{BasicConstraints, Certificate, CertificateParams, IsCa, Issuer, KeyPair, SigningKey};
use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::pem::PemObject as _;
use rustls_pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_util::sync::CancellationToken;

use crate::config::{HttpListenerMode, TlsConfig};
use crate::listener::Listener;
use crate::{serve_listener, tls};

// ============================================================================
// Test infrastructure
// ============================================================================

/// A throwaway CA + server/client/untrusted cert set, generated at test time
/// (matching vLLM's SSL-test convention of not committing key material) as PEM
/// files in a temp dir. Hold it for the test's lifetime; dropping it deletes the
/// files.
struct TestCerts {
    dir: TempDir,
}

impl TestCerts {
    fn generate() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");

        let ca_key = KeyPair::generate().expect("ca key");
        let mut ca_params = CertificateParams::new(Vec::new()).expect("ca params");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = ca_params.self_signed(&ca_key).expect("self-sign ca");
        let issuer = Issuer::from_params(&ca_params, &ca_key);

        let (server, server_key) = ca_signed(
            vec!["127.0.0.1".to_string(), "localhost".to_string()],
            &issuer,
        );
        let (client, client_key) = ca_signed(Vec::new(), &issuer);

        let untrusted_key = KeyPair::generate().expect("untrusted key");
        let untrusted = CertificateParams::new(Vec::new())
            .expect("untrusted params")
            .self_signed(&untrusted_key)
            .expect("self-sign untrusted");

        let server_pem = server.pem();
        let server_key_pem = server_key.serialize_pem();
        let files = [
            ("ca.pem", ca.pem()),
            ("server.pem", server_pem.clone()),
            ("server.key", server_key_pem.clone()),
            ("client.pem", client.pem()),
            ("client.key", client_key.serialize_pem()),
            ("untrusted_client.pem", untrusted.pem()),
            ("untrusted_client.key", untrusted_key.serialize_pem()),
            (
                "server_combined.pem",
                format!("{server_pem}{server_key_pem}"),
            ),
        ];
        for (name, pem) in files {
            std::fs::write(dir.path().join(name), pem).expect("write fixture");
        }
        Self { dir }
    }

    /// Absolute path to a fixture by name; the file need not exist.
    fn path(&self, name: &str) -> String {
        self.dir.path().join(name).to_str().expect("utf-8 path").to_string()
    }
}

fn ca_signed(sans: Vec<String>, issuer: &Issuer<'_, impl SigningKey>) -> (Certificate, KeyPair) {
    let key = KeyPair::generate().expect("key");
    let cert = CertificateParams::new(sans)
        .expect("params")
        .signed_by(&key, issuer)
        .expect("sign");
    (cert, key)
}

fn server_tls(certs: &TestCerts, cert_reqs: i32) -> TlsConfig {
    TlsConfig {
        cert_file: Some(certs.path("server.pem")),
        key_file: Some(certs.path("server.key")),
        ca_certs: (cert_reqs != 0).then(|| certs.path("ca.pem")),
        cert_reqs,
    }
}

/// A plaintext-listener TLS config for `build_server_config` checks (`cert_reqs`
/// 0, no client auth), with the cert/key files chosen by the caller.
fn build_tls(certs: &TestCerts, cert: &str, key: Option<&str>) -> TlsConfig {
    TlsConfig {
        cert_file: Some(certs.path(cert)),
        key_file: key.map(|k| certs.path(k)),
        ca_certs: None,
        cert_reqs: 0,
    }
}

/// Bind an ephemeral listener and serve a trivial router via the production
/// `serve_listener`, optionally with TLS. The listener is bound (and thus
/// accepting into the backlog) before returning, so a client may connect
/// immediately without a sleep.
async fn spawn_server(tls_config: Option<TlsConfig>) -> (String, CancellationToken) {
    let listener = Listener::bind(&HttpListenerMode::BindTcp {
        host: "127.0.0.1".to_string(),
        port: 0,
    })
    .await
    .expect("bind listener");
    let addr = listener.local_addr().expect("local addr");

    let server_config = tls_config
        .map(|cfg| Arc::new(tls::build_server_config(&cfg).expect("build server config")));
    let app = Router::new().route("/health", get(|| async { "ok" }));
    let shutdown = CancellationToken::new();
    let server_shutdown = shutdown.clone();
    tokio::spawn(async move {
        let _ = serve_listener(
            listener,
            server_config,
            app,
            server_shutdown.cancelled_owned(),
        )
        .await;
    });
    (addr, shutdown)
}

/// Client config trusting the test CA, optionally presenting a client identity
/// (`<name>.pem` + `<name>.key`) for mTLS.
fn client_config(certs: &TestCerts, identity: Option<&str>) -> ClientConfig {
    let mut roots = RootCertStore::empty();
    for ca in CertificateDer::pem_file_iter(certs.path("ca.pem")).expect("read ca") {
        roots.add(ca.expect("parse ca")).expect("add ca");
    }
    let builder = ClientConfig::builder().with_root_certificates(roots);
    match identity {
        Some(name) => {
            let chain = CertificateDer::pem_file_iter(certs.path(&format!("{name}.pem")))
                .expect("read client cert")
                .collect::<std::result::Result<Vec<_>, _>>()
                .expect("parse client cert");
            let key = PrivateKeyDer::from_pem_file(certs.path(&format!("{name}.key")))
                .expect("read client key");
            builder.with_client_auth_cert(chain, key).expect("client auth cert")
        }
        None => builder.with_no_client_auth(),
    }
}

async fn https_get(
    certs: &TestCerts,
    addr: &str,
    identity: Option<&str>,
) -> std::io::Result<String> {
    let tcp = TcpStream::connect(addr).await?;
    let connector = TlsConnector::from(Arc::new(client_config(certs, identity)));
    let server_name = ServerName::try_from("127.0.0.1").expect("server name");
    let mut tls = connector.connect(server_name, tcp).await?;
    tls.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await?;
    let mut response = String::new();
    tls.read_to_string(&mut response).await?;
    Ok(response)
}

async fn plain_get(addr: &str) -> std::io::Result<String> {
    let mut tcp = TcpStream::connect(addr).await?;
    tcp.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await?;
    let mut response = String::new();
    tcp.read_to_string(&mut response).await?;
    Ok(response)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn builds_from_combined_pem() {
    // Key omitted: it is read from the combined cert+key file.
    let certs = TestCerts::generate();
    assert!(tls::build_server_config(&build_tls(&certs, "server_combined.pem", None)).is_ok());
}

#[test]
fn rejects_missing_cert_file() {
    let certs = TestCerts::generate();
    assert!(tls::build_server_config(&build_tls(&certs, "does_not_exist.pem", None)).is_err());
}

#[tokio::test]
async fn https_request_succeeds_over_tls() {
    let certs = TestCerts::generate();
    let (addr, shutdown) = spawn_server(Some(server_tls(&certs, 0))).await;
    let response = https_get(&certs, &addr, None).await.expect("https request");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    shutdown.cancel();
}

#[tokio::test]
async fn mtls_required_rejects_client_without_certificate() {
    let certs = TestCerts::generate();
    let (addr, shutdown) = spawn_server(Some(server_tls(&certs, 2))).await;
    let result = https_get(&certs, &addr, None).await;
    assert!(
        result.is_err(),
        "handshake must fail without a client certificate"
    );
    shutdown.cancel();
}

#[tokio::test]
async fn mtls_required_accepts_valid_client_certificate() {
    let certs = TestCerts::generate();
    let (addr, shutdown) = spawn_server(Some(server_tls(&certs, 2))).await;
    let response = https_get(&certs, &addr, Some("client")).await.expect("mtls request");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    shutdown.cancel();
}

#[tokio::test]
async fn mtls_optional_allows_anonymous_and_authenticated() {
    let certs = TestCerts::generate();
    let (addr, shutdown) = spawn_server(Some(server_tls(&certs, 1))).await;
    let anonymous = https_get(&certs, &addr, None).await.expect("anonymous request");
    assert!(anonymous.starts_with("HTTP/1.1 200"), "{anonymous}");
    let authenticated =
        https_get(&certs, &addr, Some("client")).await.expect("authenticated request");
    assert!(authenticated.starts_with("HTTP/1.1 200"), "{authenticated}");
    shutdown.cancel();
}

#[tokio::test]
async fn mtls_rejects_untrusted_client_certificate() {
    // Optional (1) still verifies a presented cert, so a self-signed cert not
    // chained to the CA is rejected in both modes, not just required (2).
    let certs = TestCerts::generate();
    for cert_reqs in [1, 2] {
        let (addr, shutdown) = spawn_server(Some(server_tls(&certs, cert_reqs))).await;
        let result = https_get(&certs, &addr, Some("untrusted_client")).await;
        assert!(
            result.is_err(),
            "cert_reqs={cert_reqs}: untrusted client cert must be rejected"
        );
        shutdown.cancel();
    }
}

#[tokio::test]
async fn plain_http_serves_when_tls_is_disabled() {
    let (addr, shutdown) = spawn_server(None).await;
    let response = plain_get(&addr).await.expect("http request");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    shutdown.cancel();
}
