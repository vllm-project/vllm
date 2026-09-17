// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod hy_v3;
mod hy_v4;

pub use hy_v3::HyV3UnifiedParser;
pub use hy_v4::HyV4UnifiedParser;

const HY_MARKER_STEMS: &[&str] = &[
    "think",
    "tool_calls",
    "tool_call",
    "tool_sep",
    "arg_key",
    "arg_value",
];

/// Detect the HY structural-token suffix from tokenizer added vocabulary.
fn detect_hy_token_suffix(tokenizer: &dyn vllm_tokenizer::Tokenizer) -> String {
    tokenizer
        .added_vocab()
        .iter()
        .filter_map(|(token, id)| hy_marker_suffix(token).map(|suffix| (*id, suffix)))
        .min_by_key(|(id, _)| *id)
        .map(|(_, suffix)| suffix.to_string())
        .unwrap_or_default()
}

/// Extract the suffix from one opening or closing HY structural token.
fn hy_marker_suffix(token: &str) -> Option<&str> {
    let body = token.strip_prefix('<')?.strip_suffix('>')?;
    let body = body.strip_prefix('/').unwrap_or(body);

    HY_MARKER_STEMS.iter().find_map(|stem| {
        let suffix = body.strip_prefix(stem)?;
        (suffix.is_empty() || suffix.starts_with(':')).then_some(suffix)
    })
}
