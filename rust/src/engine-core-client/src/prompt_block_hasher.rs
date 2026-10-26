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
    use super::*;
    use crate::protocol::lora::LoraRequest;

    const SHA256_NONE_HASH: &str =
        "066b1e8987e027138e98701c46128929378935f9b6cd1549105ae6b4f0d71198";
    const SHA256_CBOR_NONE_HASH: &str =
        "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0";

    /// Token ids at every encoding-width boundary of both formats.
    const EDGE_TOKENS: [u32; 10] = [0, 1, 23, 24, 255, 256, 65535, 65536, 151_645, 0x7fff_ffff];

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

    /// First and last hash and the count, as hex.
    fn summary(hashes: &[BlockHash]) -> (String, String, usize) {
        (
            hex::encode(&hashes[0].0),
            hex::encode(&hashes.last().unwrap().0),
            hashes.len(),
        )
    }

    /// Plain prompts at several block sizes, against `hash_block_tokens`
    /// chained from the default-seed `NONE_HASH` with the real `sha256` and
    /// `sha256_cbor`.
    #[test]
    fn plain_prompts_match_python_engine() {
        let cases: [(u64, Vec<u32>); 5] = [
            (16, [EDGE_TOKENS.repeat(3), vec![7, 8]].concat()),
            (1, vec![0, 300, 70000]),
            (3, EDGE_TOKENS.to_vec()),
            (24, (100..150).collect()),
            (300, (0..650).map(|i| i * 7919 % 200_003).collect()),
        ];
        let expected = [
            (
                "sha256",
                [
                    (
                        "00e88e4087ec88315eb29afe9c001411b274ead7ffbc56e35b824d2c8817b280",
                        "70de07ec6eed1e78ee32aa5f2c17e7eddda8068507c29e9562b43e0f0a872fe0",
                        2,
                    ),
                    (
                        "1876a790824e1c62cc68ce0472a1542ccc7bf2d1d5c9e53827fc516bb9f88601",
                        "e4b0a447a032eac68f97ed77776d9f506446b12d46ec513801b12717ecaf3c8c",
                        3,
                    ),
                    (
                        "588c1268ece56d895e2730c39f59ae5f370f7ea121e4fdaa9569c7bca0ea6a00",
                        "bcfc515ddd274ab841f7c2a32d5b90a6dc4efb881d1c237b89437edafaedb691",
                        3,
                    ),
                    (
                        "160d21b10f24f083b8b07babc68f413df4dbedaaf45c1c171057b12b5f6599c7",
                        "823d4823d3b4dcc76fe5735ec2cf07ab4d3030d095050db13e783bfb69d7a742",
                        2,
                    ),
                    (
                        "60fc9719c922de9484e11daebe846969dde25dbbfc8975ad9d5a5f52f5be7cd2",
                        "a4fb5f1cf7802dcd1ed08aa20ff6f20ce522547ed9931aca73bd94fcb5f2aa2f",
                        2,
                    ),
                ],
            ),
            (
                "sha256_cbor",
                [
                    (
                        "b4e3d7f45cf989fa34b8f87e41e8da20296ed03ac04537e363d47ede8f6910aa",
                        "ab9339ba27613a3dbdd8ad445cd43a3f8175baca60c547911cc500c3ec5b409d",
                        2,
                    ),
                    (
                        "493f37dcda994a276ed9f9f27341b870a0da995cb22a36dc0e43fbf55d0cd662",
                        "5b6d403bb7875016220be9de56c3271affb85e0a77f6e4b6ac442523277bad78",
                        3,
                    ),
                    (
                        "a8419a131ade760fe7db6f193dd8ea7b0d898acd29aa35ab89176221482e24ac",
                        "ddc37216453bcfcd78ceb2903a6a2fe1e9694b11892ecff48cdad33f9a3aeccd",
                        3,
                    ),
                    (
                        "4100327c81de908bc85378ee3ed53eb3a25961ff3cad70b6f45a8a2973426f63",
                        "4962a3803c42d379a6976da0ccf67f1f0e17e9cc72d57c5080804866e5e6c2e1",
                        2,
                    ),
                    (
                        "f8d904340e7699f0e56074debe137e5c25c91fae8e2c20b77d0a34f9dd0aee23",
                        "5c579410ecd94807db1dd87e335a1c9467b20bafbbc50f5d981e8abd9bfc1eaf",
                        2,
                    ),
                ],
            ),
        ];
        for (hash_algo, expected) in expected {
            for ((block_size, tokens), (first, last, count)) in cases.iter().zip(expected) {
                let hashes = hasher(hash_algo, *block_size, 0)
                    .hash_request(&request(tokens.clone()))
                    .unwrap();
                assert_eq!(
                    summary(&hashes),
                    (first.to_string(), last.to_string(), count),
                    "{hash_algo} block_size={block_size}"
                );
            }
        }
    }

    /// LoRA, cache-salt and lookahead requests, against `Request.block_hashes`
    /// from the real `get_request_block_hasher` with block size 5 and a
    /// 32-token prompt. The last case has non-ASCII LoRA strings and a salt
    /// long enough for pickle's 4-byte string length.
    #[test]
    fn extra_keys_match_python_engine() {
        let prompt = [EDGE_TOKENS.repeat(3), vec![7, 8]].concat();
        let long_salt = "s".repeat(300);
        let requests: [(&str, EngineCoreRequest); 4] = [
            ("plain", request(prompt.clone())),
            (
                "lora",
                EngineCoreRequest {
                    lora_request: Some(lora_request("adapter-a", "/models/adapter-a")),
                    ..request(prompt.clone())
                },
            ),
            (
                "salt",
                EngineCoreRequest {
                    cache_salt: Some("tenant-1".to_string()),
                    ..request(prompt.clone())
                },
            ),
            (
                "lora_salt",
                EngineCoreRequest {
                    lora_request: Some(lora_request("adapter-é", "/m/é")),
                    cache_salt: Some(long_salt),
                    ..request(prompt.clone())
                },
            ),
        ];
        let expected = [
            (
                "sha256",
                0,
                "plain",
                6,
                "56b141f4a969e34bcc1da4ad6cb476ebcb2b7ae507e856a475f0b53f87859e20",
                "975f6dfd455769cfba2621b0645c8ba660460ceba9162c2cb5a14bf0d25e3e02",
            ),
            (
                "sha256",
                0,
                "lora",
                6,
                "354f46da231d7fbbe8387f35c2d3b314d827e063c365ebdcd5366d621de464e1",
                "82aa728271e8aaf4228f737222281aefc7e8bb96dfe2507f760295cc646b4a92",
            ),
            (
                "sha256",
                0,
                "salt",
                6,
                "06241d30c0c0700c890353e7487fbad2355e3a3f4fc34c8bb54d75e8e3005a7f",
                "f46b061b047585f338cc8ea37c19f3fe634ea9b23d35985d759ba143ff19bc91",
            ),
            (
                "sha256",
                0,
                "lora_salt",
                6,
                "a4c783bc6d449adcdafdbfd6d59b15f9c23b92eaac1efad5abe94e6f48f97dd5",
                "d94930748344c58edfd4a63ebff6c2515a93dd9a952f6ed8d0f0e210e8652093",
            ),
            (
                "sha256",
                1,
                "plain",
                6,
                "3c7c5bb68aa46add1384d9ea79ddb81fc222eada29c779ff6a3a50485fc44e28",
                "158d6ea13b7352c92eb0cd1fc2ce1683b2b3ab280351d2901a4f500ee4c032e8",
            ),
            (
                "sha256",
                1,
                "lora",
                6,
                "462d89fbe1de8e64755b339cd9b1f28d6d8fe37fd46f2c7cb500e55b967439f8",
                "1809ea41c760099ad91a5b028e2fd78bce9207f882df3a8e52616ab247db4baf",
            ),
            (
                "sha256",
                1,
                "salt",
                6,
                "b75ac67378c80b6c465aa2a1a8819d594035819511971dedf5f3edfb5347d4e9",
                "e4fa8639fc4ed687dbca1b7b398621403d4945efb58802f1bcb49825d339cd53",
            ),
            (
                "sha256",
                1,
                "lora_salt",
                6,
                "281a977700036131e70708dc04067eb00031476da6fc270eeb91ef3bec4af9fd",
                "3109559a165cba4f5adda0d5d9f54c4472ca23df2a354d19fe687df14d6c9be7",
            ),
            (
                "sha256",
                3,
                "plain",
                5,
                "befa92d3904be5c3845960b68edee89434b3f3bd4098c3ec2a40f320203e94cc",
                "158e0f4eb727ddc8eded92a45b77e328220b892f296fe8c6d304b3bd3cadb67b",
            ),
            (
                "sha256",
                3,
                "lora",
                5,
                "51725f059c14065470452d9f3027f992f601bbf622fa83864c78a0efbd063ba2",
                "e644ef52f092cc1f4b27d340e1a114df1d9d65ebd54be016303b61f6d6c65df4",
            ),
            (
                "sha256",
                3,
                "salt",
                5,
                "070f8fe284fa36a9dbe70492aa9ee5f8bbfcf185de439f3826f0f59a06d221fc",
                "08d9d549413cdfe0a3f552adddb1ca025928b2ec4eca1d76365d872370506ea0",
            ),
            (
                "sha256",
                3,
                "lora_salt",
                5,
                "d8ee74ffd008d0c7065a877ad2455f910511261a79be770ca9bb444c856126c0",
                "db3c933bf37e992760e0ce1083e9944ea25e218491bf06a75c2dd284e541fe3c",
            ),
            (
                "sha256_cbor",
                0,
                "plain",
                6,
                "339f6de80342500080b21b7e3c61c46e5efdecfe290287e0bd565a9d42fdbcbd",
                "f60a875b5ea76b5d48f036138ee14ecd08c723e23756794bd57295cb229fe5a8",
            ),
            (
                "sha256_cbor",
                0,
                "lora",
                6,
                "613539868e6bca2d39892c001f93b6883795c683f171e7108e1044215318a23a",
                "5663159167f01a066ce2482a2155247542adaebe7eab0ed01f29d71317176693",
            ),
            (
                "sha256_cbor",
                0,
                "salt",
                6,
                "f578cca646674a49a675fe07f79f5126c7cf9f103e9e2676b484e5ce8ea6e48f",
                "7acf3ac9e258df746a17015fc5b8aeac00511f599028d2b670fe20b2b92eaaa7",
            ),
            (
                "sha256_cbor",
                0,
                "lora_salt",
                6,
                "5d8fc62a8373282ad491aa6c56943ed885da3f29e5fff16640f797f013ad0746",
                "a01f56a47079cb79fa8d8edf9acf11aafbea3f417baf708b0fd06d5c64b75da9",
            ),
            (
                "sha256_cbor",
                1,
                "plain",
                6,
                "2cd1c535592d7e8f81d1153412b0c7fb319d247d1c27098daee7261bd53392c1",
                "a7f254fe782288d4a742c4fb9057e94edb970e8584c3e3b958baae0b259614ad",
            ),
            (
                "sha256_cbor",
                1,
                "lora",
                6,
                "05788a34017c1104e7f55f0ea41670eb8011d0aa1e5884f86462382adc7c3c1f",
                "1f4d67a4d69c9d6b9ba2f374c6a84db0a81893195fdf527d9e7715c4b073aebf",
            ),
            (
                "sha256_cbor",
                1,
                "salt",
                6,
                "d0932ff71fb9dd44ee41671687167b8c0fdc842cc9c32805fe30fdc2423c7418",
                "d172f715798145183e406ab09bb7aaaaa484aec2099653e192cc6ed94d14dbeb",
            ),
            (
                "sha256_cbor",
                1,
                "lora_salt",
                6,
                "5322c6f9418a576fbb8888cc4b19bf0c1364e70f819e94b9e68d979f3c837568",
                "43ec7a88990016de65d464768e3265259c41c550b5af2a56360413db52e1894b",
            ),
            (
                "sha256_cbor",
                3,
                "plain",
                5,
                "a108c7ac9e75b566b45c0fbc271641e9796a1c4d519594e4a7eb6956903ffbba",
                "d5d56edf84af5784f508a9157f1979425c3e491f1c1824abf0a9986c4307041e",
            ),
            (
                "sha256_cbor",
                3,
                "lora",
                5,
                "41ad34e840c0acd2077d256363a6bc6be245458955ab89587911ced23931cc2b",
                "9d5d87932ca2c41207ecadc63d39c9335d48b84e6be33a64aa02453768624cb5",
            ),
            (
                "sha256_cbor",
                3,
                "salt",
                5,
                "ec7caf3a4a17c4680fa13870361419627219bf4b90f6c678370584de10016ec8",
                "798b35d9c87d9e14182d410b87757a5c90fd3b3054204af3b0a0e014646deb01",
            ),
            (
                "sha256_cbor",
                3,
                "lora_salt",
                5,
                "8e7c16de0fa89e4d029c9cf08f9d1924298da3e483afc2d7e1ef584b707db4ab",
                "04ef0e5cc97bb4382dba452903f6af733972ff16582a1e6ca446269b52122d14",
            ),
        ];
        for (hash_algo, lookahead, name, count, first, last) in expected {
            let request = &requests.iter().find(|(case, _)| *case == name).unwrap().1;
            let hashes = hasher(hash_algo, 5, lookahead).hash_request(request).unwrap();
            assert_eq!(
                summary(&hashes),
                (first.to_string(), last.to_string(), count),
                "{hash_algo} lookahead={lookahead} {name}"
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
