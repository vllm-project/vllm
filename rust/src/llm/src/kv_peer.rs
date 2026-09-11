// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use futures::future::BoxFuture;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;

/// Registers a remote engine's KV transfer handshake metadata with the local
/// engines before a request that references that engine is submitted.
///
/// The frontend sees `kv_transfer_params` before the engine does, so it can
/// fetch the peer's handshake payload over the peer's control plane and hand it
/// to the workers, which then never have to open a side channel themselves.
pub trait KvPeerHandshake: Send + Sync {
    fn ensure<'a>(
        &'a self,
        client: &'a EngineCoreClient,
        kv_transfer_params: &'a serde_json::Value,
    ) -> BoxFuture<'a, crate::Result<()>>;
}

/// The `kv_transfer_params` a request carries for engine-core, if any.
pub(crate) fn kv_transfer_params(request: &EngineCoreRequest) -> Option<&serde_json::Value> {
    request.sampling_params.as_ref()?.extra_args.as_ref()?.get("kv_transfer_params")
}
