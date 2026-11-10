// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prefix-caching block hashes, bitwise identical to `vllm.utils.hashing` and
//! `vllm.v1.core.kv_cache_utils.hash_block_tokens`.

mod cbor;
mod pickle;
mod value;

use sha2::{Digest, Sha256};
use smallvec::SmallVec;

pub use self::value::{ExtraKeys, ObjectId, Value};

/// The digest of a single KV-cache block used for prefix caching.
///
/// Mirrors Python's `BlockHash` (`bytes`): 32 bytes for SHA-256 based
/// algorithms and 16 bytes for xxh3-128 based ones.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash(pub SmallVec<[u8; 32]>);

impl BlockHash {
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self(SmallVec::from_slice(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0.iter().take(8) {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Append the encodings of `tokens`, each given as up to 5 bytes and the
/// number of them that are used.
#[inline(always)]
fn extend_encoded(out: &mut Vec<u8>, tokens: &[i32], encode: impl Fn(i32) -> ([u8; 5], usize)) {
    let start = out.len();
    // Writing whole 5-byte chunks into a pre-sized buffer keeps the loop free
    // of capacity checks and variable-length copies.
    out.resize(start + tokens.len() * 5, 0);
    let buf = &mut out[start..];
    let mut pos = 0;
    for &t in tokens {
        let (bytes, len) = encode(t);
        buf[pos..pos + 5].copy_from_slice(&bytes);
        pos += len;
    }
    out.truncate(start + pos);
}

/// A prefix-caching hash algorithm (`--prefix-caching-hash-algo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashAlgo {
    /// SHA-256 of the pickled key (`vllm.utils.hashing.sha256`).
    Sha256,
    /// SHA-256 of the canonical CBOR key (`sha256_cbor`).
    Sha256Cbor,
    /// xxh3-128 of the pickled key (`xxhash`).
    Xxhash,
    /// xxh3-128 of the canonical CBOR key (`xxhash_cbor`).
    XxhashCbor,
}

impl HashAlgo {
    /// The algorithm named as in `vllm.utils.hashing.get_hash_fn_by_name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sha256" => Self::Sha256,
            "sha256_cbor" => Self::Sha256Cbor,
            "xxhash" => Self::Xxhash,
            "xxhash_cbor" => Self::XxhashCbor,
            _ => return None,
        })
    }

    fn is_cbor(self) -> bool {
        matches!(self, Self::Sha256Cbor | Self::XxhashCbor)
    }

    fn digest(self, data: &[u8]) -> BlockHash {
        match self {
            Self::Sha256 | Self::Sha256Cbor => BlockHash::from_slice(&Sha256::digest(data)),
            // `xxhash.xxh3_128_digest` is the big-endian canonical form.
            Self::Xxhash | Self::XxhashCbor => {
                BlockHash::from_slice(&xxhash_rust::xxh3::xxh3_128(data).to_be_bytes())
            }
        }
    }

    /// Hash an arbitrary value, as the Python hash function named by `self`.
    pub fn hash_value(self, value: &Value) -> BlockHash {
        self.digest(&self.serialize_value(value))
    }

    /// Serialize `value` as the Python hash function named by `self` does
    /// before digesting it.
    pub fn serialize_value(self, value: &Value) -> Vec<u8> {
        let mut buf = Vec::new();
        if self.is_cbor() {
            cbor::dumps_into(&mut buf, value);
        } else {
            pickle::dumps_into(&mut buf, value);
        }
        buf
    }

    fn hash_block_key(
        self,
        buf: &mut Vec<u8>,
        parent: &[u8],
        tokens: &[i32],
        extra_keys: ExtraKeys<'_>,
    ) -> BlockHash {
        if self.is_cbor() {
            cbor::dumps_block_key_into(buf, parent, tokens, extra_keys);
        } else {
            pickle::dumps_block_key_into(buf, parent, tokens, extra_keys);
        }
        self.digest(buf)
    }
}

/// Computes a hash value corresponding to the contents of a block and the
/// contents of the preceding block(s), as
/// `kv_cache_utils.hash_block_tokens(hash_fn, parent_block_hash,
/// curr_block_token_ids, extra_keys)`.
///
/// `parent` is `None` (or empty) for the first block, which then chains from
/// `none_hash` (`NONE_HASH`).
pub fn hash_block_tokens(
    algo: HashAlgo,
    parent: Option<&BlockHash>,
    none_hash: &BlockHash,
    tokens: &[i32],
    extra_keys: ExtraKeys<'_>,
) -> BlockHash {
    let mut buf = Vec::new();
    hash_block_tokens_into(&mut buf, algo, parent, none_hash, tokens, extra_keys)
}

fn hash_block_tokens_into(
    buf: &mut Vec<u8>,
    algo: HashAlgo,
    parent: Option<&BlockHash>,
    none_hash: &BlockHash,
    tokens: &[i32],
    extra_keys: ExtraKeys<'_>,
) -> BlockHash {
    // Python: `if not parent_block_hash: parent_block_hash = NONE_HASH`.
    let parent = parent.filter(|p| !p.0.is_empty()).unwrap_or(none_hash);
    algo.hash_block_key(buf, parent.as_bytes(), tokens, extra_keys)
}

/// The request-level inputs of `generate_block_hash_extra_keys` that Rust
/// supports.
///
/// The strings are pickled as objects distinct from the `"lora"` and
/// `"cache_salt"` key literals, so they must not be those very Python objects
/// (only possible for strings equal to them). Wrap a string in
/// [`Value::Ident`] if it is the same object as another one here, e.g. a LoRA
/// path that is its name.
// TODO: multi-modal and prompt-embeds extra keys, once requests are ported.
#[derive(Clone, Debug, Default)]
pub struct RequestExtraKeys {
    /// `(lora_name, lora_path)` of the request's LoRA adapter, keying every
    /// block as `("lora", lora_name, lora_path)`.
    pub lora: Option<(Value, Value)>,
    /// The non-empty cache salt, keying the first block as
    /// `("cache_salt", cache_salt)`.
    pub cache_salt: Option<Value>,
}

impl RequestExtraKeys {
    /// The extra keys of the block starting at token `start`, as
    /// `generate_block_hash_extra_keys`: a tuple of the LoRA and cache-salt
    /// keys, or `None` without either.
    fn for_block(&self, start: usize) -> Option<Value> {
        let lora = self
            .lora
            .iter()
            .map(|(name, path)| Value::tuple([Value::str("lora"), name.clone(), path.clone()]));
        let cache_salt = (self.cache_salt.as_ref())
            .filter(|_| start == 0)
            .map(|salt| Value::tuple([Value::str("cache_salt"), salt.clone()]));
        let keys: Vec<_> = lora.chain(cache_salt).collect();
        (!keys.is_empty()).then_some(Value::Tuple(keys))
    }
}

/// Computes the un-computed block hashes of a request, as the closure
/// returned by `kv_cache_utils.get_request_block_hasher`.
///
/// Hashes are computed at `hash_block_size` granularity and chained over the
/// full prefix, so each hash uniquely fingerprints the prefix ending at its
/// boundary.
///
/// With `num_hash_lookahead_tokens` (L) > 0, each hash also covers the L
/// tokens after its block, which the drafter's KV at the block's last position
/// depends on, so it is emitted only once those tokens exist. The block tokens
/// stay the hashed tokens; the extra keys become `(E, *tokens)`, where `E`
/// covers the block and those tokens.
#[derive(Clone, Debug)]
pub struct RequestBlockHasher {
    algo: HashAlgo,
    none_hash: BlockHash,
    hash_block_size: usize,
    num_hash_lookahead_tokens: usize,
}

impl RequestBlockHasher {
    pub fn new(
        algo: HashAlgo,
        none_hash: BlockHash,
        hash_block_size: usize,
        num_hash_lookahead_tokens: usize,
    ) -> Self {
        assert!(hash_block_size > 0, "hash_block_size must be positive");
        Self {
            algo,
            none_hash,
            hash_block_size,
            num_hash_lookahead_tokens,
        }
    }

    /// The hashes of the full blocks of `all_token_ids` after the
    /// `num_existing` blocks already hashed, the last of which is
    /// `last_hash`.
    pub fn new_block_hashes(
        &self,
        all_token_ids: &[i32],
        num_existing: usize,
        last_hash: Option<&BlockHash>,
        extra_keys: &RequestExtraKeys,
    ) -> Vec<BlockHash> {
        let bs = self.hash_block_size;
        let lookahead = self.num_hash_lookahead_tokens;
        let num_tokens = all_token_ids.len().saturating_sub(lookahead);
        let mut start = num_existing * bs;
        if start + bs > num_tokens {
            // Early stop when there no new full blocks created.
            return Vec::new();
        }

        // Blocks after the first new one all have the same keys.
        let first_start = start;
        let first_keys = extra_keys.for_block(start);
        let later_keys = extra_keys.for_block(start + bs);
        let mut hashes = Vec::with_capacity((num_tokens - start) / bs);
        let mut buf = Vec::new();
        let mut parent = last_hash.cloned();
        while start + bs <= num_tokens {
            let end = start + bs;
            let keys = if start == first_start {
                first_keys.as_ref()
            } else {
                later_keys.as_ref()
            };
            let extra_keys = if lookahead > 0 {
                ExtraKeys::Lookahead {
                    keys,
                    tokens: &all_token_ids[end..end + lookahead],
                }
            } else {
                ExtraKeys::Plain(keys)
            };
            let hash = hash_block_tokens_into(
                &mut buf,
                self.algo,
                parent.as_ref(),
                &self.none_hash,
                &all_token_ids[start..end],
                extra_keys,
            );
            hashes.push(hash.clone());
            parent = Some(hash);
            start = end;
        }
        hashes
    }
}

#[cfg(test)]
mod tests;
