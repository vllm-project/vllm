// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Test helpers for the crates that speak the protocol.

use bytes::Bytes;

use crate::request::{EngineCoreRequest, decode_engine_core_request};

/// Decode the `[request type, payload, aux frames...]` of an add request,
/// resolving the arrays sent in auxiliary frames.
pub fn decode_add_request(frames: &[Bytes]) -> EngineCoreRequest {
    decode_engine_core_request(&frames[1..]).expect("decode add request")
}

/// The prompt token ids of a request, which must be inline or resolved by
/// [`decode_add_request`].
pub fn decode_prompt_token_ids(request: &EngineCoreRequest) -> Option<Vec<u32>> {
    let ids = request.prompt_token_ids.as_ref()?;
    Some(ids.to_u32_vec().expect("prompt token ids are inline 32-bit integers"))
}
