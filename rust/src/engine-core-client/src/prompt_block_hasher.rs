// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prompt block hashes computed in the frontend, so that the engine doesn't
//! hash long prompts on its input thread.
//!
//! Each hash must be byte-identical to the engine's own, because the engine
//! chains the hashes of blocks completed by outputs onto them, and because KV
//! connectors and KV-event consumers compare them across instances.
//!
//! Original Python definitions:
//! - `hash_block_tokens`, `generate_block_hash_extra_keys` and
//!   `get_request_block_hasher`:
//!   <https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py>
//! - `sha256` and `sha256_cbor`:
//!   <https://github.com/vllm-project/vllm/blob/main/vllm/utils/hashing.py>

use sha2::{Digest, Sha256};

use crate::protocol::handshake::PrefixHashConfig;
use crate::protocol::request::{BlockHash, EngineCoreRequest};

/// Python's C pickler starts a new frame once one reaches 64 KiB, which this
/// encoder doesn't reproduce, so longer keys are left to the engine.
const PICKLE_FRAME_SIZE_TARGET: usize = 64 * 1024;

/// How the engine serializes `(parent_hash, tuple(block_tokens), extra_keys)`
/// before hashing it with SHA-256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyEncoding {
    /// `sha256`: `pickle.dumps(key, protocol=5)`.
    Pickle,
    /// `sha256_cbor`: `cbor2.dumps(key, canonical=True)`.
    Cbor,
}

/// A Python value in a block's extra keys, limited to the kinds this frontend
/// encodes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExtraKey<'a> {
    None,
    Int(u32),
    Str(&'a str),
    Tuple(Vec<ExtraKey<'a>>),
}

/// The request-level inputs to `generate_block_hash_extra_keys` that this
/// frontend encodes.
struct RequestExtraKeys<'a> {
    /// `("lora", lora_name, lora_path)`, on every block.
    lora: Option<ExtraKey<'a>>,
    /// A non-empty cache salt, keyed on the first block only.
    cache_salt: Option<&'a str>,
}

impl<'a> RequestExtraKeys<'a> {
    /// The extra keys of the block starting at `start`: a tuple of the LoRA
    /// and cache-salt keys, or `None` without either.
    fn for_block(&self, start: usize) -> ExtraKey<'a> {
        let cache_salt = self
            .cache_salt
            .filter(|_| start == 0)
            .map(|salt| ExtraKey::Tuple(vec![ExtraKey::Str("cache_salt"), ExtraKey::Str(salt)]));
        let keys: Vec<_> = self.lora.iter().cloned().chain(cache_salt).collect();
        if keys.is_empty() {
            ExtraKey::None
        } else {
            ExtraKey::Tuple(keys)
        }
    }
}

/// Computes the hashes of a prompt's full blocks the way the engine would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptBlockHasher {
    encoding: KeyEncoding,
    block_size: usize,
    none_hash: Vec<u8>,
    num_lookahead_tokens: usize,
}

impl PromptBlockHasher {
    /// Return a hasher for the engine's config, or `None` when this frontend
    /// can't reproduce its hashes, in which case the engine hashes prompts.
    pub(crate) fn new(config: &PrefixHashConfig) -> Option<Self> {
        // TODO: support `xxhash` and `xxhash_cbor` (XXH3-128 over the same
        // encodings).
        let encoding = match config.hash_algo.as_str() {
            "sha256" => KeyEncoding::Pickle,
            "sha256_cbor" => KeyEncoding::Cbor,
            _ => return None,
        };
        let block_size = usize::try_from(config.hash_block_size).ok()?;
        let num_lookahead_tokens = usize::try_from(config.num_lookahead_tokens).ok()?;
        let none_hash = config.none_hash.0.clone();
        // Both encodings below write the parent hash with a 1-byte length.
        let supported = block_size > 0 && none_hash.len() < 256;
        supported.then_some(Self {
            encoding,
            block_size,
            none_hash,
            num_lookahead_tokens,
        })
    }

    /// Return the request's prompt block hashes, or `None` if the engine has to
    /// compute them because they need inputs this frontend doesn't encode.
    pub(crate) fn hash_request(&self, request: &EngineCoreRequest) -> Option<Vec<BlockHash>> {
        // TODO: encode multimodal extra keys, `("mm", identifier, offset)` per
        // item overlapping the block, and the prompt-embeds keys of mixed-mode
        // requests once this frontend supports prompt embeds.
        let unsupported = request.mm_features.as_ref().is_some_and(|features| !features.is_empty())
            || request.prompt_embeds.is_some()
            || request.prompt_is_token_ids.is_some();
        if unsupported {
            return None;
        }
        let lora = request.lora_request.as_ref();
        // Python adds no cache-salt key for an empty salt.
        let cache_salt = request.cache_salt.as_deref().filter(|salt| !salt.is_empty());
        let strings = (lora.into_iter())
            .flat_map(|lora| [lora.lora_name.as_str(), lora.lora_path.as_str()])
            .chain(cache_salt);
        if self.encoding == KeyEncoding::Pickle && may_share_pickled_objects(strings) {
            return None;
        }
        let keys = RequestExtraKeys {
            lora: lora.map(|lora| {
                ExtraKey::Tuple(vec![
                    ExtraKey::Str("lora"),
                    ExtraKey::Str(&lora.lora_name),
                    ExtraKey::Str(&lora.lora_path),
                ])
            }),
            cache_salt,
        };
        self.hash_prompt(request.prompt_token_ids.as_deref()?, &keys)
    }

    /// Hash each full block of `token_ids` whose lookahead tokens are also in
    /// the prompt, chained from the engine's `NONE_HASH`. Returns `None` for a
    /// key this frontend can't encode exactly.
    fn hash_prompt(
        &self,
        token_ids: &[u32],
        keys: &RequestExtraKeys<'_>,
    ) -> Option<Vec<BlockHash>> {
        let lookahead = self.num_lookahead_tokens;
        let num_blocks = token_ids.len().saturating_sub(lookahead) / self.block_size;
        let mut key = Vec::new();
        let mut parent = self.none_hash.clone();
        (0..num_blocks)
            .map(|index| {
                let start = index * self.block_size;
                let end = start + self.block_size;
                let mut extra_keys = keys.for_block(start);
                // With lookahead, a block's extra keys become `(E, *tokens)`
                // with the lookahead tokens after it.
                if lookahead > 0 {
                    let tokens = token_ids[end..end + lookahead].iter().map(|&t| ExtraKey::Int(t));
                    extra_keys = ExtraKey::Tuple([extra_keys].into_iter().chain(tokens).collect());
                }
                key.clear();
                match self.encoding {
                    KeyEncoding::Pickle => {
                        encode_pickle_key(&mut key, &parent, &token_ids[start..end], &extra_keys)?
                    }
                    KeyEncoding::Cbor => {
                        encode_cbor_key(&mut key, &parent, &token_ids[start..end], &extra_keys)
                    }
                }
                parent = Sha256::digest(&key).to_vec();
                Some(BlockHash(parent.clone()))
            })
            .collect()
    }
}

/// Whether pickle may write one of these strings as a reference to an
/// identical object earlier in the key instead of writing it again: pickle's
/// memo is by object identity, CPython shares the empty and single-Latin-1-
/// character strings, and a string could also be the very object of a key
/// literal. Conservative; requests where this matters are left to the engine.
fn may_share_pickled_objects<'a>(strings: impl Iterator<Item = &'a str>) -> bool {
    let shared_by_cpython = |s: &str| {
        let mut chars = s.chars();
        match (chars.next(), chars.next()) {
            (None, _) => true,
            (Some(c), None) => u32::from(c) < 0x100,
            _ => false,
        }
    };
    let strings: Vec<_> = strings.collect();
    strings.iter().enumerate().any(|(i, s)| {
        ["lora", "cache_salt"].contains(s) || (shared_by_cpython(s) && strings[i + 1..].contains(s))
    })
}

const MEMOIZE: u8 = 0x94;

/// Write `pickle.dumps((parent, tuple(block), extra_keys), protocol=5)`, as
/// CPython's C pickler writes it.
fn encode_pickle_key(
    out: &mut Vec<u8>,
    parent: &[u8],
    block: &[u32],
    extra_keys: &ExtraKey<'_>,
) -> Option<()> {
    // PROTO 5, then FRAME with the length of everything after it.
    out.extend_from_slice(&[0x80, 0x05, 0x95]);
    let frame_len_at = out.len();
    out.extend_from_slice(&[0; 8]);
    // SHORT_BINBYTES parent.
    out.push(b'C');
    out.push(u8::try_from(parent.len()).ok()?);
    out.extend_from_slice(parent);
    out.push(MEMOIZE);
    write_pickle_tuple(out, block.len(), |out| {
        block.iter().try_for_each(|&token| write_pickle_int(out, token))
    })?;
    write_pickle_value(out, extra_keys)?;
    // TUPLE3 for the key itself, then STOP.
    out.extend_from_slice(&[0x87, MEMOIZE, b'.']);
    let frame_len = out.len() - frame_len_at - 8;
    if frame_len >= PICKLE_FRAME_SIZE_TARGET {
        return None;
    }
    out[frame_len_at..frame_len_at + 8].copy_from_slice(&(frame_len as u64).to_le_bytes());
    Some(())
}

/// Write a non-empty tuple whose items `write_items` writes.
fn write_pickle_tuple(
    out: &mut Vec<u8>,
    len: usize,
    write_items: impl FnOnce(&mut Vec<u8>) -> Option<()>,
) -> Option<()> {
    if len == 0 {
        // EMPTY_TUPLE, which pickle doesn't memoize.
        out.push(b')');
        return Some(());
    }
    // Tuples of up to three items have dedicated opcodes; longer ones use MARK.
    if len > 3 {
        out.push(b'(');
    }
    write_items(out)?;
    out.push(match len {
        1 => 0x85, // TUPLE1
        2 => 0x86, // TUPLE2
        3 => 0x87, // TUPLE3
        _ => b't', // TUPLE
    });
    out.push(MEMOIZE);
    Some(())
}

fn write_pickle_int(out: &mut Vec<u8>, value: u32) -> Option<()> {
    match value {
        // BININT1, BININT2, BININT (signed).
        0..=0xff => out.extend_from_slice(&[b'K', value as u8]),
        0x100..=0xffff => {
            out.push(b'M');
            out.extend_from_slice(&(value as u16).to_le_bytes());
        }
        0x1_0000..=0x7fff_ffff => {
            out.push(b'J');
            out.extend_from_slice(&value.to_le_bytes());
        }
        // Larger ints are pickled with LONG1, which no real token id needs.
        _ => return None,
    }
    Some(())
}

fn write_pickle_value(out: &mut Vec<u8>, value: &ExtraKey<'_>) -> Option<()> {
    match value {
        ExtraKey::None => out.push(b'N'),
        ExtraKey::Int(value) => write_pickle_int(out, *value)?,
        ExtraKey::Str(s) => {
            // SHORT_BINUNICODE or BINUNICODE, of the UTF-8 bytes.
            match u8::try_from(s.len()) {
                Ok(len) => out.extend_from_slice(&[0x8c, len]),
                Err(_) => {
                    out.push(b'X');
                    out.extend_from_slice(&u32::try_from(s.len()).ok()?.to_le_bytes());
                }
            }
            out.extend_from_slice(s.as_bytes());
            out.push(MEMOIZE);
        }
        ExtraKey::Tuple(items) => write_pickle_tuple(out, items.len(), |out| {
            items.iter().try_for_each(|item| write_pickle_value(out, item))
        })?,
    }
    Some(())
}

const CBOR_BYTES: u8 = 0x40;
const CBOR_TEXT: u8 = 0x60;
const CBOR_ARRAY: u8 = 0x80;
const CBOR_NULL: u8 = 0xf6;

/// Write `cbor2.dumps((parent, tuple(block), extra_keys), canonical=True)`.
fn encode_cbor_key(out: &mut Vec<u8>, parent: &[u8], block: &[u32], extra_keys: &ExtraKey<'_>) {
    out.push(CBOR_ARRAY | 3);
    write_cbor_head(out, CBOR_BYTES, parent.len() as u64);
    out.extend_from_slice(parent);
    write_cbor_head(out, CBOR_ARRAY, block.len() as u64);
    for &token in block {
        write_cbor_head(out, 0x00, token.into());
    }
    write_cbor_value(out, extra_keys);
}

fn write_cbor_value(out: &mut Vec<u8>, value: &ExtraKey<'_>) {
    match value {
        ExtraKey::None => out.push(CBOR_NULL),
        ExtraKey::Int(value) => write_cbor_head(out, 0x00, (*value).into()),
        ExtraKey::Str(s) => {
            write_cbor_head(out, CBOR_TEXT, s.len() as u64);
            out.extend_from_slice(s.as_bytes());
        }
        ExtraKey::Tuple(items) => {
            write_cbor_head(out, CBOR_ARRAY, items.len() as u64);
            items.iter().for_each(|item| write_cbor_value(out, item));
        }
    }
}

/// Write a CBOR item head in its shortest form, as canonical encoding requires.
fn write_cbor_head(out: &mut Vec<u8>, major: u8, value: u64) {
    match value {
        0..=23 => out.push(major | value as u8),
        24..=0xff => out.extend_from_slice(&[major | 24, value as u8]),
        0x100..=0xffff => {
            out.push(major | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(major | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            out.push(major | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde::Deserialize;

    use super::*;
    use crate::protocol::lora::LoraRequest;

    const SHA256_NONE_HASH: &str =
        "066b1e8987e027138e98701c46128929378935f9b6cd1549105ae6b4f0d71198";
    const SHA256_CBOR_NONE_HASH: &str =
        "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0";

    fn hasher(hash_algo: &str, block_size: u64, lookahead: u64) -> PromptBlockHasher {
        let none_hash = match hash_algo {
            "sha256" => SHA256_NONE_HASH,
            _ => SHA256_CBOR_NONE_HASH,
        };
        PromptBlockHasher::new(&PrefixHashConfig {
            hash_algo: hash_algo.to_string(),
            hash_block_size: block_size,
            none_hash: BlockHash(hex::decode(none_hash).unwrap()),
            num_lookahead_tokens: lookahead,
        })
        .unwrap()
    }

    fn request(prompt_token_ids: Vec<u32>) -> EngineCoreRequest {
        EngineCoreRequest {
            prompt_token_ids: Some(prompt_token_ids),
            ..EngineCoreRequest::default()
        }
    }

    fn lora_request(lora_name: &str, lora_path: &str) -> LoraRequest {
        LoraRequest {
            lora_name: lora_name.to_string(),
            lora_int_id: 1,
            lora_path: lora_path.to_string(),
            base_model_name: None,
            load_inplace: false,
            is_3d_lora_weight: false,
        }
    }

    /// Prompt block hashes from the Python engine, kept current by
    /// `tests/v1/core/test_prompt_block_hash_vectors.py` in the repo root, which
    /// fails if the engine's hashing changes.
    const PYTHON_VECTORS: &str = include_str!("tests/prompt_block_hash_vectors.json");

    #[derive(Deserialize)]
    struct Vectors {
        none_hashes: HashMap<String, String>,
        cases: Vec<VectorCase>,
    }

    #[derive(Deserialize)]
    struct VectorCase {
        name: String,
        hash_algo: String,
        block_size: u64,
        num_lookahead_tokens: u64,
        lora: Option<VectorLora>,
        cache_salt: Option<String>,
        prompt_token_ids: Vec<u32>,
        block_hashes: Vec<String>,
    }

    #[derive(Deserialize)]
    struct VectorLora {
        name: String,
        path: String,
    }

    /// Plain, LoRA, cache-salt and lookahead requests at several block sizes,
    /// against `Request.block_hashes` from the real `get_request_block_hasher`
    /// with `sha256` and `sha256_cbor`.
    #[test]
    fn prompt_block_hashes_match_python_engine() {
        let vectors: Vectors = serde_json::from_str(PYTHON_VECTORS).unwrap();
        // The vectors chain from the engine's default-seed `NONE_HASH`.
        assert_eq!(vectors.none_hashes["sha256"], SHA256_NONE_HASH);
        assert_eq!(vectors.none_hashes["sha256_cbor"], SHA256_CBOR_NONE_HASH);
        assert!(!vectors.cases.is_empty());
        for case in vectors.cases {
            let request = EngineCoreRequest {
                lora_request: case.lora.map(|lora| lora_request(&lora.name, &lora.path)),
                cache_salt: case.cache_salt,
                ..request(case.prompt_token_ids)
            };
            let hashes = hasher(&case.hash_algo, case.block_size, case.num_lookahead_tokens)
                .hash_request(&request)
                .unwrap_or_else(|| panic!("{} {}: not hashed", case.hash_algo, case.name));
            let hashes: Vec<_> = hashes.iter().map(|hash| hex::encode(&hash.0)).collect();
            assert_eq!(
                hashes, case.block_hashes,
                "{} {}",
                case.hash_algo, case.name
            );
        }
    }

    #[test]
    fn prompt_without_a_full_block_has_no_hashes() {
        assert_eq!(
            hasher("sha256", 16, 0).hash_request(&request(vec![1; 15])),
            Some(vec![])
        );
        // The lookahead tokens of the only full block are missing.
        assert_eq!(
            hasher("sha256", 16, 2).hash_request(&request(vec![1; 17])),
            Some(vec![])
        );
    }

    #[test]
    fn token_id_beyond_i32_is_left_to_the_engine() {
        let tokens = vec![1, 0x8000_0000];
        assert_eq!(
            hasher("sha256", 2, 0).hash_request(&request(tokens.clone())),
            None
        );
        // CBOR encodes every u32 directly.
        assert!(hasher("sha256_cbor", 2, 0).hash_request(&request(tokens)).is_some());
    }

    #[test]
    fn key_longer_than_a_pickle_frame_is_left_to_the_engine() {
        let tokens = vec![70_000; 16 * 1024];
        assert_eq!(
            hasher("sha256", 16 * 1024, 0).hash_request(&request(tokens.clone())),
            None
        );
        assert!(hasher("sha256_cbor", 16 * 1024, 0).hash_request(&request(tokens)).is_some());
    }

    #[test]
    fn unsupported_configs_are_left_to_the_engine() {
        let config = |hash_algo: &str, hash_block_size| PrefixHashConfig {
            hash_algo: hash_algo.to_string(),
            hash_block_size,
            none_hash: BlockHash(vec![0; 32]),
            num_lookahead_tokens: 0,
        };
        assert!(PromptBlockHasher::new(&config("xxhash", 16)).is_none());
        assert!(PromptBlockHasher::new(&config("sha256", 0)).is_none());
    }

    #[test]
    fn unsupported_requests_are_left_to_the_engine() {
        let pickle = hasher("sha256", 2, 0);
        let base = request(vec![1, 2, 3, 4]);
        // Python adds no cache-salt key for an empty salt.
        let empty_salt = EngineCoreRequest {
            cache_salt: Some(String::new()),
            ..base.clone()
        };
        assert_eq!(pickle.hash_request(&empty_salt), pickle.hash_request(&base));
        let mixed = EngineCoreRequest {
            prompt_is_token_ids: Some(vec![true; 4]),
            ..base.clone()
        };
        assert_eq!(pickle.hash_request(&mixed), None);
        // Pickle might write these strings as references to an earlier object:
        // repeated shared strings, or one that could be a key literal.
        let lora = |lora_name: &str, lora_path: &str, cache_salt: Option<&str>| EngineCoreRequest {
            lora_request: Some(lora_request(lora_name, lora_path)),
            cache_salt: cache_salt.map(str::to_string),
            ..base.clone()
        };
        for request in [
            lora("a", "a", None),
            lora("", "", None),
            lora("x", "/p", Some("x")),
            lora("lora", "/p", None),
            lora("adapter", "/p", Some("cache_salt")),
        ] {
            assert_eq!(pickle.hash_request(&request), None, "{request:?}");
        }
        // Equal strings that CPython doesn't share are distinct objects.
        assert!(pickle.hash_request(&lora("adapter", "/p", Some("adapter"))).is_some());
        assert!(pickle.hash_request(&lora("a", "b", None)).is_some());
        // CBOR has no memo.
        let cbor = hasher("sha256_cbor", 2, 0);
        assert!(cbor.hash_request(&lora("a", "a", None)).is_some());
    }
}
