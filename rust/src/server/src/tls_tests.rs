//! TLS tests: `build_server_config` unit checks plus end-to-end OpenSSL handshakes
//! through the production `serve_listener` path, with a trivial router since TLS
//! terminates below the app.

use axum::Router;
use axum::routing::get;
use openssl::asn1::Asn1Time;
use openssl::bn::{BigNum, MsbOption};
use openssl::ec::{EcGroup, EcKey};
use openssl::hash::MessageDigest;
use openssl::nid::Nid;
use openssl::pkey::{PKey, Private};
use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
use openssl::x509::extension::{BasicConstraints, KeyUsage, SubjectAlternativeName};
use openssl::x509::{X509, X509NameBuilder};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_openssl::SslStream;
use tokio_util::sync::CancellationToken;

use crate::config::{HttpListenerMode, TlsConfig};
use crate::listener::Listener;
use crate::{serve_listener, tls};

// ============================================================================
// Test infrastructure
// ============================================================================

/// A throwaway CA + server/client/untrusted/chain cert set as PEM files in a
/// temp dir; dropping it deletes them.
struct TestCerts {
    dir: TempDir,
}

impl TestCerts {
    fn generate() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");

        let (ca, ca_key) = build_ca();
        let (server, server_key) = build_leaf("server", &["127.0.0.1", "localhost"], &ca, &ca_key);
        let (client, client_key) = build_leaf("client", &[], &ca, &ca_key);
        let (untrusted, untrusted_key) = build_self_signed("untrusted client");

        // Leaf signed by an intermediate (itself signed by the root); the cert
        // file holds leaf + intermediate, for the chain-serving test.
        let (intermediate, intermediate_key) = build_intermediate(&ca, &ca_key);
        let (chain_leaf, chain_leaf_key) = build_leaf(
            "chain",
            &["127.0.0.1", "localhost"],
            &intermediate,
            &intermediate_key,
        );

        let server_pem = pem(&server);
        let server_key_pem = key_pem(&server_key);
        let files = [
            ("ca.pem", pem(&ca)),
            ("server.pem", server_pem.clone()),
            ("server.key", server_key_pem.clone()),
            ("client.pem", pem(&client)),
            ("client.key", key_pem(&client_key)),
            ("untrusted_client.pem", pem(&untrusted)),
            ("untrusted_client.key", key_pem(&untrusted_key)),
            (
                "server_combined.pem",
                format!("{server_pem}{server_key_pem}"),
            ),
            (
                "server_chain.pem",
                format!("{}{}", pem(&chain_leaf), pem(&intermediate)),
            ),
            ("server_chain.key", key_pem(&chain_leaf_key)),
        ];
        for (name, contents) in files {
            std::fs::write(dir.path().join(name), contents).expect("write fixture");
        }
        Self { dir }
    }

    /// Absolute path to a fixture by name; the file need not exist.
    fn path(&self, name: &str) -> String {
        self.dir.path().join(name).to_str().expect("utf-8 path").to_string()
    }
}

fn gen_key() -> PKey<Private> {
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).expect("ec group");
    let ec = EcKey::generate(&group).expect("ec key");
    PKey::from_ec_key(ec).expect("pkey")
}

fn serial() -> openssl::asn1::Asn1Integer {
    let mut bn = BigNum::new().expect("bignum");
    bn.rand(159, MsbOption::MAYBE_ZERO, false).expect("rand serial");
    bn.to_asn1_integer().expect("asn1 serial")
}

fn x509_name(cn: &str) -> openssl::x509::X509Name {
    let mut builder = X509NameBuilder::new().expect("name builder");
    builder.append_entry_by_text("CN", cn).expect("cn");
    builder.build()
}

fn pem(cert: &X509) -> String {
    String::from_utf8(cert.to_pem().expect("cert pem")).expect("utf-8 cert")
}

fn key_pem(key: &PKey<Private>) -> String {
    String::from_utf8(key.private_key_to_pem_pkcs8().expect("key pem")).expect("utf-8 key")
}

/// A self-signed CA used to sign the server/client leaf certs.
fn build_ca() -> (X509, PKey<Private>) {
    let key = gen_key();
    let name = x509_name("vLLM Test CA");
    let mut builder = X509::builder().expect("x509 builder");
    builder.set_version(2).expect("version");
    builder.set_serial_number(&serial()).expect("serial");
    builder.set_subject_name(&name).expect("subject");
    builder.set_issuer_name(&name).expect("issuer");
    builder.set_pubkey(&key).expect("pubkey");
    builder
        .set_not_before(&Asn1Time::days_from_now(0).expect("nb"))
        .expect("set nb");
    builder
        .set_not_after(&Asn1Time::days_from_now(3650).expect("na"))
        .expect("set na");
    builder
        .append_extension(BasicConstraints::new().critical().ca().build().expect("bc"))
        .expect("ext bc");
    builder
        .append_extension(
            KeyUsage::new().critical().key_cert_sign().crl_sign().build().expect("ku"),
        )
        .expect("ext ku");
    builder.sign(&key, MessageDigest::sha256()).expect("sign ca");
    (builder.build(), key)
}

/// A CA-signed leaf cert with optional subject-alternative names (IP or DNS).
fn build_leaf(cn: &str, sans: &[&str], ca: &X509, ca_key: &PKey<Private>) -> (X509, PKey<Private>) {
    let key = gen_key();
    let mut builder = X509::builder().expect("x509 builder");
    builder.set_version(2).expect("version");
    builder.set_serial_number(&serial()).expect("serial");
    builder.set_subject_name(&x509_name(cn)).expect("subject");
    builder.set_issuer_name(ca.subject_name()).expect("issuer");
    builder.set_pubkey(&key).expect("pubkey");
    builder
        .set_not_before(&Asn1Time::days_from_now(0).expect("nb"))
        .expect("set nb");
    builder
        .set_not_after(&Asn1Time::days_from_now(3650).expect("na"))
        .expect("set na");
    builder
        .append_extension(BasicConstraints::new().build().expect("bc"))
        .expect("ext bc");
    if !sans.is_empty() {
        let mut san = SubjectAlternativeName::new();
        for entry in sans {
            if entry.parse::<std::net::IpAddr>().is_ok() {
                san.ip(entry);
            } else {
                san.dns(entry);
            }
        }
        let ext = san.build(&builder.x509v3_context(Some(ca), None)).expect("san");
        builder.append_extension(ext).expect("ext san");
    }
    builder.sign(ca_key, MessageDigest::sha256()).expect("sign leaf");
    (builder.build(), key)
}

/// A self-signed leaf not chained to the CA, for the untrusted-client test.
fn build_self_signed(cn: &str) -> (X509, PKey<Private>) {
    let key = gen_key();
    let name = x509_name(cn);
    let mut builder = X509::builder().expect("x509 builder");
    builder.set_version(2).expect("version");
    builder.set_serial_number(&serial()).expect("serial");
    builder.set_subject_name(&name).expect("subject");
    builder.set_issuer_name(&name).expect("issuer");
    builder.set_pubkey(&key).expect("pubkey");
    builder
        .set_not_before(&Asn1Time::days_from_now(0).expect("nb"))
        .expect("set nb");
    builder
        .set_not_after(&Asn1Time::days_from_now(3650).expect("na"))
        .expect("set na");
    builder
        .append_extension(BasicConstraints::new().build().expect("bc"))
        .expect("ext bc");
    builder.sign(&key, MessageDigest::sha256()).expect("sign self");
    (builder.build(), key)
}

/// A CA-capable intermediate signed by the root, for the full-chain test.
fn build_intermediate(ca: &X509, ca_key: &PKey<Private>) -> (X509, PKey<Private>) {
    let key = gen_key();
    let mut builder = X509::builder().expect("x509 builder");
    builder.set_version(2).expect("version");
    builder.set_serial_number(&serial()).expect("serial");
    builder
        .set_subject_name(&x509_name("vLLM Test Intermediate CA"))
        .expect("subject");
    builder.set_issuer_name(ca.subject_name()).expect("issuer");
    builder.set_pubkey(&key).expect("pubkey");
    builder
        .set_not_before(&Asn1Time::days_from_now(0).expect("nb"))
        .expect("set nb");
    builder
        .set_not_after(&Asn1Time::days_from_now(3650).expect("na"))
        .expect("set na");
    builder
        .append_extension(BasicConstraints::new().critical().ca().build().expect("bc"))
        .expect("ext bc");
    builder
        .append_extension(
            KeyUsage::new().critical().key_cert_sign().crl_sign().build().expect("ku"),
        )
        .expect("ext ku");
    builder.sign(ca_key, MessageDigest::sha256()).expect("sign intermediate");
    (builder.build(), key)
}

fn server_tls(certs: &TestCerts, cert_reqs: i32) -> TlsConfig {
    TlsConfig {
        cert_file: Some(certs.path("server.pem")),
        key_file: Some(certs.path("server.key")),
        ca_certs: (cert_reqs != 0).then(|| certs.path("ca.pem")),
        cert_reqs,
        ciphers: None,
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
        ciphers: None,
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

    let server_config = tls_config.map(|cfg| {
        std::sync::Arc::new(tls::build_server_config(&cfg).expect("build server config"))
    });
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

/// Issue an HTTPS GET trusting the test CA, optionally presenting a client
/// identity (`<name>.pem` + `<name>.key`) for mTLS. Hostname verification is
/// disabled (the IP-SAN match is not under test); chain verification stays on,
/// so an untrusted server cert is still rejected.
async fn https_get(
    certs: &TestCerts,
    addr: &str,
    identity: Option<&str>,
) -> std::io::Result<String> {
    let tcp = TcpStream::connect(addr).await?;

    let mut builder = SslConnector::builder(SslMethod::tls_client()).expect("connector builder");
    builder.set_ca_file(certs.path("ca.pem")).expect("trust ca");
    if let Some(name) = identity {
        builder
            .set_certificate_chain_file(certs.path(&format!("{name}.pem")))
            .expect("client cert");
        builder
            .set_private_key_file(certs.path(&format!("{name}.key")), SslFiletype::PEM)
            .expect("client key");
    }
    let connector = builder.build();
    let mut config = connector.configure().expect("configure");
    config.set_verify_hostname(false);
    let ssl = config.into_ssl("127.0.0.1").expect("ssl");

    let stream = SslStream::new(ssl, tcp).expect("client ssl stream");
    tokio::pin!(stream);
    stream.as_mut().connect().await.map_err(std::io::Error::other)?;

    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await?;
    let mut response = String::new();
    stream.read_to_string(&mut response).await?;
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

#[test]
fn accepts_valid_cipher_list() {
    let certs = TestCerts::generate();
    let mut cfg = build_tls(&certs, "server.pem", Some("server.key"));
    cfg.ciphers = Some("ECDHE-ECDSA-AES256-GCM-SHA384".to_string());
    assert!(tls::build_server_config(&cfg).is_ok());
}

#[test]
fn rejects_invalid_cipher_list() {
    let certs = TestCerts::generate();
    let mut cfg = build_tls(&certs, "server.pem", Some("server.key"));
    cfg.ciphers = Some("THIS-IS-NOT-A-CIPHER".to_string());
    assert!(tls::build_server_config(&cfg).is_err());
}

#[test]
fn rejects_mismatched_cert_and_key() {
    // check_private_key must reject a key that does not match the certificate.
    let certs = TestCerts::generate();
    let tls = build_tls(&certs, "client.pem", Some("server.key"));
    assert!(tls::build_server_config(&tls).is_err());
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
async fn serves_full_certificate_chain() {
    // Cert file holds leaf + intermediate; a client trusting only the root can
    // verify only if the server sends the intermediate, guarding against a
    // leaf-only load.
    let certs = TestCerts::generate();
    let tls = TlsConfig {
        cert_file: Some(certs.path("server_chain.pem")),
        key_file: Some(certs.path("server_chain.key")),
        ca_certs: None,
        cert_reqs: 0,
        ciphers: None,
    };
    let (addr, shutdown) = spawn_server(Some(tls)).await;
    let response = https_get(&certs, &addr, None).await.expect("chained https request");
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
