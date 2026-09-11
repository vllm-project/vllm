// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Fetch a remote engine's KV handshake metadata over its gRPC control plane
//! and push it into the local engines before a request that references it is
//! submitted.

use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use bytes::Bytes;
use futures::future::BoxFuture;
use hyper_util::rt::TokioIo;
use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
use parking_lot::Mutex;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tokio::net::TcpStream;
use tokio::sync::OnceCell;
use tokio_openssl::SslStream;
use tonic::transport::{Channel, Endpoint, Uri};
use tonic::{Code, Status};
use tower::service_fn;
use tracing::{info, warn};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::handshake::{HandshakeTransport, KvTransferInfo};
use vllm_engine_core_client::protocol::kv_transfer::KvConnectorHandshakeEntry;
use vllm_engine_core_client::protocol::{decode_value, encode_msgpack};
use vllm_llm::{Error as LlmError, KvPeerHandshake};

use crate::config::TlsConfig;
use crate::grpc::pb;
use crate::grpc::pb::kv_transfer_client::KvTransferClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on one fetch and push, the RPC included.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
/// A failed fetch is not retried for this long.
const FAILURE_BACKOFF: Duration = Duration::from_secs(10);
/// Re-push a peer's payload after this long so a worker that evicted the
/// engine for inactivity (default TTL one hour) gets it again before use.
const REFRESH_AFTER: Duration = Duration::from_secs(30 * 60);
/// ALPN wire bytes for HTTP/2 (length-prefixed).
const ALPN_H2: &[u8] = b"\x02h2";

/// A peer engine named in a request's `kv_transfer_params` that advertises a
/// gRPC control plane.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KvPeer {
    engine_id: String,
    host: String,
    control_port: u16,
    tls: bool,
}

impl KvPeer {
    /// `None` when the params do not name a peer or the peer only offers the
    /// legacy ZMQ side channel, which the worker handles itself.
    fn from_params(params: &Value) -> Option<Self> {
        let engine_id = params.get("remote_engine_id")?.as_str()?;
        let host = params
            .get("remote_control_host")
            .and_then(Value::as_str)
            .or_else(|| params.get("remote_host")?.as_str())?;
        let control_port = u16::try_from(params.get("remote_control_port")?.as_u64()?).ok()?;
        let tls = params.get("remote_control_tls").and_then(Value::as_bool).unwrap_or(false);
        Some(Self {
            engine_id: engine_id.to_string(),
            host: host.to_string(),
            control_port,
            tls,
        })
    }

    /// The host without IPv6 brackets, as dialed and as verified against the
    /// peer's certificate.
    fn bare_host(&self) -> &str {
        self.host.trim_start_matches('[').trim_end_matches(']')
    }

    fn uri(&self) -> String {
        let scheme = if self.tls { "https" } else { "http" };
        let host = self.bare_host();
        if host.contains(':') {
            format!("{scheme}://[{host}]:{}", self.control_port)
        } else {
            format!("{scheme}://{host}:{}", self.control_port)
        }
    }
}

/// A peer failure rendered as its status code, its message when it carries one,
/// and what an unrouted path means for the handshake.
fn peer_status_message(status: &Status) -> String {
    let code = status.code();
    let mut message = match status.message() {
        "" => format!("{code:?}"),
        detail => format!("{code:?}: {detail}"),
    };
    if code == Code::Unimplemented {
        message.push_str(
            " (the peer does not mount the vllm.KvTransfer service; \
             check its --grpc-services, or whether it predates the service)",
        );
    }
    message
}

/// Client side of the TLS a peer's control plane terminates, taken from this
/// frontend's own `--ssl-*` flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerTlsConfig {
    /// CA bundle that verifies peer certificates; the system roots when `None`.
    pub ca_file: Option<String>,
    /// Certificate chain and private key presented to peers that require a
    /// client certificate.
    pub identity: Option<(String, String)>,
}

impl PeerTlsConfig {
    pub fn from_server_tls(tls: &TlsConfig) -> Self {
        Self {
            ca_file: tls.ca_certs.clone(),
            identity: tls.cert_file.clone().map(|cert| {
                let key = tls.key_file.clone().unwrap_or_else(|| cert.clone());
                (cert, key)
            }),
        }
    }

    fn connector(&self) -> Result<SslConnector> {
        let mut builder = SslConnector::builder(SslMethod::tls_client())
            .context("failed to initialize TLS for KV peer handshakes")?;
        match &self.ca_file {
            Some(ca_file) => builder
                .set_ca_file(ca_file)
                .with_context(|| format!("failed to load --ssl-ca-certs {ca_file:?}"))?,
            None => builder
                .set_default_verify_paths()
                .context("failed to load the system CA certificates")?,
        }
        if let Some((cert_file, key_file)) = &self.identity {
            builder
                .set_certificate_chain_file(cert_file)
                .with_context(|| format!("failed to load client certificate {cert_file:?}"))?;
            builder
                .set_private_key_file(key_file, SslFiletype::PEM)
                .with_context(|| format!("failed to load client private key {key_file:?}"))?;
            builder
                .check_private_key()
                .context("the client certificate and private key do not match")?;
        }
        builder
            .set_alpn_protos(ALPN_H2)
            .context("failed to set ALPN for KV peer handshakes")?;
        Ok(builder.build())
    }
}

/// The handshake transport of the local engines' workers; engines that report
/// none behave as `auto`.
pub(crate) fn local_handshake_transport(infos: &[KvTransferInfo]) -> HandshakeTransport {
    infos.iter().find_map(|info| info.handshake_transport).unwrap_or_default()
}

/// Reject a prefilling engine pinned to `handshake_transport=grpc` when this
/// frontend does not serve the handshake: that setting also closes the ZMQ side
/// channel, so peers would have no way to reach its metadata.
pub(crate) fn check_handshake_transport(
    infos: &[KvTransferInfo],
    kv_control_port: Option<u16>,
) -> Result<()> {
    if kv_control_port.is_some() {
        return Ok(());
    }
    if let Some(info) = infos.iter().find(|info| {
        info.handshake_transport == Some(HandshakeTransport::Grpc) && info.kv_role != "kv_consumer"
    }) {
        bail!(
            "engine {} ({}) sets handshake_transport=grpc, which closes the NIXL side \
             channel, so peers can only fetch its handshake metadata from this frontend: \
             start it with --grpc-port and keep kv-transfer in --grpc-services",
            info.engine_id,
            info.kv_role,
        );
    }
    Ok(())
}

/// The outcome of the last fetch for one peer.
#[derive(Debug, Clone)]
enum PeerState {
    Pushed(Instant),
    Failed { at: Instant, message: String },
}

impl PeerState {
    fn is_fresh(&self) -> bool {
        match self {
            Self::Pushed(at) => at.elapsed() < REFRESH_AFTER,
            Self::Failed { at, .. } => at.elapsed() < FAILURE_BACKOFF,
        }
    }
}

/// Peer handshake state shared by every request: one cell per peer address so
/// concurrent requests fetch once. A push is refreshed after
/// [`REFRESH_AFTER`] and a failure retried after [`FAILURE_BACKOFF`]. A failure
/// fails the request only under `handshake_transport=grpc`; under `auto` the
/// workers fall back to the ZMQ side channel, and under `zmq` nothing is fetched.
pub struct KvPeerHandshaker {
    transport: HandshakeTransport,
    tls: SslConnector,
    fetch_timeout: Duration,
    peers: Mutex<HashMap<KvPeer, Arc<OnceCell<PeerState>>>>,
}

impl KvPeerHandshaker {
    pub fn new(transport: HandshakeTransport, tls: &PeerTlsConfig) -> Result<Self> {
        Ok(Self {
            transport,
            tls: tls.connector()?,
            fetch_timeout: FETCH_TIMEOUT,
            peers: Mutex::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_fetch_timeout(mut self, fetch_timeout: Duration) -> Self {
        self.fetch_timeout = fetch_timeout;
        self
    }

    fn cell_for(&self, peer: &KvPeer) -> Arc<OnceCell<PeerState>> {
        let mut peers = self.peers.lock();
        if let Some(cell) = peers.get(peer)
            && cell.get().is_none_or(PeerState::is_fresh)
        {
            return cell.clone();
        }
        let cell = Arc::new(OnceCell::new());
        peers.insert(peer.clone(), cell.clone());
        cell
    }

    async fn connect(&self, peer: &KvPeer) -> Result<Channel, tonic::transport::Error> {
        let endpoint = Endpoint::from_shared(peer.uri())?.connect_timeout(CONNECT_TIMEOUT);
        if !peer.tls {
            return endpoint.connect().await;
        }
        let connector = self.tls.clone();
        let host = peer.bare_host().to_string();
        let port = peer.control_port;
        endpoint
            .connect_with_connector(service_fn(move |_: Uri| {
                let connector = connector.clone();
                let host = host.clone();
                async move {
                    let tcp = TcpStream::connect((host.as_str(), port)).await?;
                    let ssl = connector
                        .configure()
                        .and_then(|config| config.into_ssl(&host))
                        .map_err(io::Error::other)?;
                    let mut stream = SslStream::new(ssl, tcp).map_err(io::Error::other)?;
                    Pin::new(&mut stream).connect().await.map_err(io::Error::other)?;
                    Ok::<_, io::Error>(TokioIo::new(stream))
                }
            }))
            .await
    }

    async fn fetch_and_push(&self, client: &EngineCoreClient, peer: &KvPeer) -> Result<(), String> {
        let fail = |what: &str, error: String| {
            format!(
                "{what} for engine {} at {}: {error}",
                peer.engine_id,
                peer.uri()
            )
        };
        let channel = self
            .connect(peer)
            .await
            .map_err(|error| fail("connect", error.to_report_string()))?;
        let response = KvTransferClient::new(channel)
            .get_kv_handshake_metadata(pb::GetKvHandshakeMetadataRequest {
                engine_id: peer.engine_id.clone(),
            })
            .await
            .map_err(|status| fail("GetKvHandshakeMetadata", peer_status_message(&status)))?
            .into_inner();
        if response.ranks.is_empty() {
            return Err(fail(
                "GetKvHandshakeMetadata",
                "no handshake metadata served".to_string(),
            ));
        }
        let entries: Vec<KvConnectorHandshakeEntry> = response
            .ranks
            .into_iter()
            .map(|rank| KvConnectorHandshakeEntry {
                pp_rank: rank.pp_rank,
                tp_rank: rank.tp_rank,
                payload: Bytes::from(rank.payload),
                compatibility_hash: (!rank.compatibility_hash.is_empty())
                    .then_some(rank.compatibility_hash),
            })
            .collect();
        let rank_count = entries.len();
        // Utility args encode structs positionally; the engine's
        // KVConnectorHandshakeEntry dataclass wants the named-map form, so
        // pass the entries as pre-encoded map values.
        let entries = encode_msgpack(&entries)
            .and_then(|bytes| decode_value(&bytes))
            .map_err(|error| fail("encode entries", error.to_report_string()))?;
        client
            .call_utility::<(), _>("add_remote_kv_handshake", (&peer.engine_id, entries))
            .await
            .map_err(|error| fail("add_remote_kv_handshake", error.to_report_string()))?;
        info!(
            engine_id = %peer.engine_id,
            control = %peer.uri(),
            rank_count,
            "registered remote KV handshake metadata from peer control plane"
        );
        Ok(())
    }

    async fn fetch(&self, client: &EngineCoreClient, peer: &KvPeer) -> PeerState {
        let outcome =
            match tokio::time::timeout(self.fetch_timeout, self.fetch_and_push(client, peer)).await
            {
                Ok(outcome) => outcome,
                Err(_) => Err(format!(
                    "fetch for engine {} at {} timed out after {:?}",
                    peer.engine_id,
                    peer.uri(),
                    self.fetch_timeout
                )),
            };
        match outcome {
            Ok(()) => PeerState::Pushed(Instant::now()),
            Err(message) => {
                warn!(
                    transport = ?self.transport,
                    "KV peer handshake over the control plane failed, retrying after {:?}: {message}",
                    FAILURE_BACKOFF
                );
                PeerState::Failed {
                    at: Instant::now(),
                    message,
                }
            }
        }
    }
}

impl KvPeerHandshake for KvPeerHandshaker {
    fn ensure<'a>(
        &'a self,
        client: &'a EngineCoreClient,
        kv_transfer_params: &'a Value,
    ) -> BoxFuture<'a, vllm_llm::Result<()>> {
        Box::pin(async move {
            if self.transport == HandshakeTransport::Zmq {
                return Ok(());
            }
            let Some(peer) = KvPeer::from_params(kv_transfer_params) else {
                return Ok(());
            };
            let cell = self.cell_for(&peer);
            match cell.get_or_init(|| self.fetch(client, &peer)).await {
                PeerState::Failed { message, .. } if self.transport == HandshakeTransport::Grpc => {
                    Err(LlmError::KvPeerHandshake {
                        message: message.clone(),
                    })
                }
                PeerState::Pushed(_) | PeerState::Failed { .. } => Ok(()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tonic::Status;
    use vllm_engine_core_client::protocol::handshake::{HandshakeTransport, KvTransferInfo};

    use crate::config::TlsConfig;
    use crate::kv_peer::{
        KvPeer, KvPeerHandshaker, PeerTlsConfig, check_handshake_transport,
        local_handshake_transport, peer_status_message,
    };

    fn info(role: &str, transport: Option<HandshakeTransport>) -> KvTransferInfo {
        KvTransferInfo {
            engine_id: "p0".to_string(),
            kv_connector: "NixlConnector".to_string(),
            kv_role: role.to_string(),
            handshake_transport: transport,
        }
    }

    fn peer(host: &str) -> KvPeer {
        KvPeer {
            engine_id: "p0".to_string(),
            host: host.to_string(),
            control_port: 50051,
            tls: false,
        }
    }

    #[test]
    fn peer_status_message_names_the_code_and_an_unmounted_service() {
        let message = peer_status_message(&Status::unimplemented(""));
        assert!(message.starts_with("Unimplemented"), "{message}");
        assert!(message.contains("vllm.KvTransfer"), "{message}");

        assert_eq!(
            peer_status_message(&Status::not_found("unknown KV transfer engine_id \"p0\"")),
            "NotFound: unknown KV transfer engine_id \"p0\""
        );
    }

    #[test]
    fn peer_requires_engine_host_and_control_port() {
        assert_eq!(
            KvPeer::from_params(&json!({"do_remote_prefill": true})),
            None
        );
        assert_eq!(
            KvPeer::from_params(&json!({
                "remote_engine_id": "p0", "remote_host": "10.0.0.1", "remote_port": 5600
            })),
            None,
            "a ZMQ-only peer is left to the worker"
        );
        let peer = KvPeer::from_params(&json!({
            "remote_engine_id": "p0", "remote_host": "fd00::1", "remote_port": 5600,
            "remote_control_port": 50051
        }))
        .expect("peer with a control port");
        assert_eq!(peer.uri(), "http://[fd00::1]:50051");
        let via_service = KvPeer::from_params(&json!({
            "remote_engine_id": "p0", "remote_host": "10.0.0.1", "remote_port": 5600,
            "remote_control_port": 50051, "remote_control_host": "prefill-control"
        }))
        .expect("peer with a control host");
        assert_eq!(via_service.uri(), "http://prefill-control:50051");
    }

    #[test]
    fn tls_peer_dials_https_on_the_bare_host() {
        let peer = KvPeer::from_params(&json!({
            "remote_engine_id": "p0", "remote_host": "[fd00::1]",
            "remote_control_port": 50051, "remote_control_tls": true
        }))
        .expect("tls peer");
        assert!(peer.tls);
        assert_eq!(peer.bare_host(), "fd00::1");
        assert_eq!(peer.uri(), "https://[fd00::1]:50051");
    }

    #[test]
    fn peers_are_cached_per_address() {
        let handshaker = KvPeerHandshaker::new(HandshakeTransport::Auto, &PeerTlsConfig::default())
            .expect("handshaker");
        let real = handshaker.cell_for(&peer("10.0.0.1"));
        assert!(Arc::ptr_eq(&real, &handshaker.cell_for(&peer("10.0.0.1"))));
        let other_host = handshaker.cell_for(&peer("10.6.6.6"));
        assert!(
            !Arc::ptr_eq(&real, &other_host),
            "a request naming another host for the same engine must not share its cell"
        );
        let tls = KvPeer {
            tls: true,
            ..peer("10.0.0.1")
        };
        assert!(!Arc::ptr_eq(&real, &handshaker.cell_for(&tls)));
    }

    #[test]
    fn peer_tls_follows_the_server_flags() {
        let combined = TlsConfig {
            cert_file: Some("server.pem".to_string()),
            key_file: None,
            ca_certs: Some("ca.pem".to_string()),
            cert_reqs: 2,
            ciphers: None,
        };
        assert_eq!(
            PeerTlsConfig::from_server_tls(&combined),
            PeerTlsConfig {
                ca_file: Some("ca.pem".to_string()),
                identity: Some(("server.pem".to_string(), "server.pem".to_string())),
            }
        );
        let split = TlsConfig {
            key_file: Some("server.key".to_string()),
            ca_certs: None,
            ..combined
        };
        assert_eq!(
            PeerTlsConfig::from_server_tls(&split),
            PeerTlsConfig {
                ca_file: None,
                identity: Some(("server.pem".to_string(), "server.key".to_string())),
            }
        );
    }

    #[test]
    fn missing_peer_ca_file_fails_construction() {
        let error = KvPeerHandshaker::new(
            HandshakeTransport::Auto,
            &PeerTlsConfig {
                ca_file: Some("/nonexistent/ca.pem".to_string()),
                identity: None,
            },
        )
        .err()
        .expect("missing CA bundle");
        assert!(format!("{error:#}").contains("--ssl-ca-certs"), "{error:#}");
    }

    #[test]
    fn grpc_transport_needs_a_control_port_on_prefilling_engines() {
        let grpc = Some(HandshakeTransport::Grpc);
        for role in ["kv_producer", "kv_both"] {
            let error = check_handshake_transport(&[info(role, grpc)], None)
                .expect_err("grpc without a control port");
            assert!(format!("{error:#}").contains("--grpc-port"), "{error:#}");
            check_handshake_transport(&[info(role, grpc)], Some(50051))
                .expect("grpc with a control port");
        }
        check_handshake_transport(&[info("kv_consumer", grpc)], None)
            .expect("a consumer serves no handshake");
        check_handshake_transport(
            &[
                info("kv_both", Some(HandshakeTransport::Auto)),
                info("kv_both", None),
            ],
            None,
        )
        .expect("auto keeps the side channel");
    }

    #[test]
    fn local_transport_defaults_to_auto() {
        assert_eq!(local_handshake_transport(&[]), HandshakeTransport::Auto);
        assert_eq!(
            local_handshake_transport(&[info("kv_both", None)]),
            HandshakeTransport::Auto
        );
        assert_eq!(
            local_handshake_transport(&[info("kv_both", Some(HandshakeTransport::Zmq))]),
            HandshakeTransport::Zmq
        );
    }
}
