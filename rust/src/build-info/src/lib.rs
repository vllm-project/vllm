// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/// The vLLM package version supplied by the build system.
///
/// Direct Cargo builds fall back to the internal crate version.
pub const VERSION: &str = match option_env!("VLLM_RS_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};
