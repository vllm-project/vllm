// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::OnceLock;

/// Marks the slot that `tools/build_rust.py --stamp` fills after linking, so a
/// cached binary can take a new version without relinking.
const VERSION_SLOT_MAGIC: &[u8] = b"@@VLLM_RS_VERSION_SLOT@@";
const VERSION_SLOT_LEN: usize = 128;
const SLOT_SIZE: usize = VERSION_SLOT_MAGIC.len() + VERSION_SLOT_LEN;

#[used]
static VERSION_SLOT: [u8; SLOT_SIZE] = {
    let mut slot = [0; SLOT_SIZE];
    let mut i = 0;
    while i < VERSION_SLOT_MAGIC.len() {
        slot[i] = VERSION_SLOT_MAGIC[i];
        i += 1;
    }
    slot
};

/// The version supplied at compile time. Direct Cargo builds fall back to the
/// internal crate version.
const COMPILED_VERSION: &str = match option_env!("VLLM_RS_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

/// The vLLM package version: the stamped one if the binary was stamped,
/// otherwise the one supplied at compile time.
pub fn version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        // SAFETY: `VERSION_SLOT` is a valid, aligned static. The read is
        // volatile so the optimizer cannot fold in the unstamped contents.
        let slot = unsafe { std::ptr::read_volatile(&VERSION_SLOT) };
        let stamped = &slot[VERSION_SLOT_MAGIC.len()..];
        let len = stamped.iter().position(|&b| b == 0).unwrap_or(VERSION_SLOT_LEN);
        match std::str::from_utf8(&stamped[..len]) {
            Ok(version) if !version.is_empty() => version.to_owned(),
            _ => COMPILED_VERSION.to_owned(),
        }
    })
}
