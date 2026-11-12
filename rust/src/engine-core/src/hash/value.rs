// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The subset of Python values that appear in block-hash keys.

/// Identity of a Python object (e.g. its `id()`), used to reproduce pickle's
/// memoization of repeated objects.
pub type ObjectId = usize;

/// A Python value as passed to the prefix-caching hash functions.
///
/// Pickle memoizes `bytes`, `str` and non-empty `tuple` objects by identity:
/// when the same object is saved twice, the second occurrence is a memo
/// reference (`BINGET`) instead of a copy. Plain nodes are treated as distinct
/// objects; wrap a node in [`Value::Ident`] to tell the pickler which nodes
/// are the same Python object. CBOR ignores identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    None,
    Bool(bool),
    /// Python `int`, limited to the `i128` range.
    Int(i128),
    Bytes(Vec<u8>),
    /// Python `str` without lone surrogates.
    Str(String),
    Tuple(Vec<Value>),
    /// A memoizable node (`Bytes`, `Str` or `Tuple`) together with the
    /// identity of the Python object it came from. Nodes with equal ids must
    /// be equal values.
    Ident(ObjectId, Box<Value>),
}

impl Value {
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }

    pub fn tuple(items: impl IntoIterator<Item = Value>) -> Self {
        Self::Tuple(items.into_iter().collect())
    }
}

/// The extra keys of a block, i.e. the third element of the hashed
/// `(parent_block_hash, block_token_ids, extra_keys)` tuple.
#[derive(Clone, Copy, Debug)]
pub enum ExtraKeys<'a> {
    /// `extra_keys` as returned by `generate_block_hash_extra_keys`; `None`
    /// when the block has no extra keys.
    Plain(Option<&'a Value>),
    /// `(keys, *lookahead_tokens)`, as built by `get_request_block_hasher`
    /// when `num_hash_lookahead_tokens > 0`.
    Lookahead {
        keys: Option<&'a Value>,
        tokens: &'a [i32],
    },
}
