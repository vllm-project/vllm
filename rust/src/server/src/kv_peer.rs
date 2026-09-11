// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Fetch a remote engine's KV handshake metadata over its gRPC control plane
//! and push it into the local engines before a request that references it is
//! submitted.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tokio::sync::OnceCell;
use tonic::transport::Endpoint;
use tonic::{Code, Status};
use tracing::info;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::kv_transfer::KvConnectorHandshakeEntry;
use vllm_engine_core_client::protocol::{decode_value, encode_msgpack};
use vllm_llm::{Error as LlmError, KvPeerHandshake};

use crate::grpc::pb;
use crate::grpc::pb::kv_transfer_client::KvTransferClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Re-push a peer's payload after this long so a worker that evicted the
/// engine for inactivity (default TTL one hour) gets it again before use.
const REFRESH_AFTER: Duration = Duration::from_secs(30 * 60);

/// A peer engine named in a request's `kv_transfer_params` that advertises a
/// gRPC control plane.
#[derive(Debug, Clone, PartialEq, Eq)]
struct KvPeer {
    engine_id: String,
    host: String,
    control_port: u16,
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
        Some(Self {
            engine_id: engine_id.to_string(),
            host: host.to_string(),
            control_port,
        })
    }

    fn uri(&self) -> String {
        if self.host.contains(':') && !self.host.starts_with('[') {
            format!("http://[{}]:{}", self.host, self.control_port)
        } else {
            format!("http://{}:{}", self.host, self.control_port)
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

/// Peer handshake state shared by every request: one cell per remote engine so
/// concurrent first requests fetch once, refreshed after [`REFRESH_AFTER`].
#[derive(Default)]
pub struct KvPeerHandshaker {
    peers: Mutex<HashMap<String, Arc<OnceCell<Instant>>>>,
}

impl KvPeerHandshaker {
    pub fn new() -> Self {
        Self::default()
    }

    fn cell_for(&self, engine_id: &str) -> Arc<OnceCell<Instant>> {
        let mut peers = self.peers.lock();
        if let Some(cell) = peers.get(engine_id)
            && cell.get().is_none_or(|pushed_at| pushed_at.elapsed() < REFRESH_AFTER)
        {
            return cell.clone();
        }
        let cell = Arc::new(OnceCell::new());
        peers.insert(engine_id.to_string(), cell.clone());
        cell
    }

    async fn fetch_and_push(
        &self,
        client: &EngineCoreClient,
        peer: &KvPeer,
    ) -> vllm_llm::Result<()> {
        let fail = |what: &str, error: String| LlmError::KvPeerHandshake {
            message: format!(
                "{what} for engine {} at {}: {error}",
                peer.engine_id,
                peer.uri()
            ),
        };
        let channel = Endpoint::from_shared(peer.uri())
            .map_err(|error| fail("invalid control address", error.to_report_string()))?
            .connect_timeout(CONNECT_TIMEOUT)
            .connect()
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
}

impl KvPeerHandshake for KvPeerHandshaker {
    fn ensure<'a>(
        &'a self,
        client: &'a EngineCoreClient,
        kv_transfer_params: &'a Value,
    ) -> BoxFuture<'a, vllm_llm::Result<()>> {
        Box::pin(async move {
            let Some(peer) = KvPeer::from_params(kv_transfer_params) else {
                return Ok(());
            };
            let cell = self.cell_for(&peer.engine_id);
            cell.get_or_try_init(|| async {
                self.fetch_and_push(client, &peer).await.map(|()| Instant::now())
            })
            .await?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tonic::Status;

    use super::{KvPeer, peer_status_message};

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
        .unwrap();
        assert_eq!(peer.uri(), "http://[fd00::1]:50051");
        let via_service = KvPeer::from_params(&json!({
            "remote_engine_id": "p0", "remote_host": "10.0.0.1", "remote_port": 5600,
            "remote_control_port": 50051, "remote_control_host": "prefill-control"
        }))
        .unwrap();
        assert_eq!(via_service.uri(), "http://prefill-control:50051");
    }
}
