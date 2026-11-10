// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Byte-exact reproduction of `cbor2.dumps(obj, canonical=True)` for
//! [`Value`]s.
//!
//! Canonical mode only affects floats, maps and sets, none of which appear in
//! [`Value`]. `cbor2` shares no values or strings by default, so identity is
//! irrelevant.

use super::extend_encoded;
use super::value::{ExtraKeys, Value};

const MAJOR_UINT: u8 = 0;
const MAJOR_NEGINT: u8 = 1;
const MAJOR_BYTES: u8 = 2;
const MAJOR_TEXT: u8 = 3;
const MAJOR_ARRAY: u8 = 4;
const MAJOR_TAG: u8 = 6;
const FALSE: u8 = 0xf4;
const TRUE: u8 = 0xf5;
const NULL: u8 = 0xf6;
const TAG_POS_BIGNUM: u64 = 2;
const TAG_NEG_BIGNUM: u64 = 3;

/// Serialize `value` like `cbor2.dumps(value, canonical=True)` into `out`
/// (cleared first).
pub fn dumps_into(out: &mut Vec<u8>, value: &Value) {
    out.clear();
    encode(out, value);
}

/// Serialize the block key `(parent, tuple(tokens), extra_keys)` like
/// `cbor2.dumps(..., canonical=True)` into `out` (cleared first).
pub fn dumps_block_key_into(
    out: &mut Vec<u8>,
    parent: &[u8],
    tokens: &[i32],
    extra_keys: ExtraKeys<'_>,
) {
    out.clear();
    out.reserve(tokens.len() * 5 + parent.len() + 32);
    encode_length(out, MAJOR_ARRAY, 3);
    encode_length(out, MAJOR_BYTES, parent.len() as u64);
    out.extend_from_slice(parent);
    encode_ints(out, tokens);
    match extra_keys {
        ExtraKeys::Plain(keys) => encode_opt(out, keys),
        ExtraKeys::Lookahead { keys, tokens } => {
            encode_length(out, MAJOR_ARRAY, 1 + tokens.len() as u64);
            encode_opt(out, keys);
            extend_encoded(out, tokens, encode_i32);
        }
    }
}

fn encode_ints(out: &mut Vec<u8>, tokens: &[i32]) {
    encode_length(out, MAJOR_ARRAY, tokens.len() as u64);
    extend_encoded(out, tokens, encode_i32);
}

/// [`encode_int`] for an `i32`: the bytes and their count (1, 2, 3 or 5).
///
/// Branch-free, since token ids of random widths would mispredict.
#[inline(always)]
fn encode_i32(v: i32) -> ([u8; 5], usize) {
    // `-(v + 1)` is `!v` for negative values.
    let major = if v < 0 { MAJOR_NEGINT } else { MAJOR_UINT };
    let arg = (v ^ (v >> 31)) as u32;
    // 0: inline, 1-3: 1, 2 or 4 argument bytes.
    let class = usize::from(arg >= 24) + usize::from(arg > 0xff) + usize::from(arg > 0xffff);
    let width = [0, 1, 2, 4][class];
    let info = if class == 0 {
        arg as u8
    } else {
        23 + class as u8
    };
    // The argument's `width` big-endian bytes, moved to the front.
    let b = ((u64::from(arg) << (32 - 8 * width)) as u32).to_be_bytes();
    ([(major << 5) | info, b[0], b[1], b[2], b[3]], 1 + width)
}

fn encode_opt(out: &mut Vec<u8>, value: Option<&Value>) {
    match value {
        Some(v) => encode(out, v),
        None => out.push(NULL),
    }
}

fn encode(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::None => out.push(NULL),
        Value::Bool(b) => out.push(if *b { TRUE } else { FALSE }),
        Value::Int(v) => encode_int(out, *v),
        Value::Bytes(b) => {
            encode_length(out, MAJOR_BYTES, b.len() as u64);
            out.extend_from_slice(b);
        }
        Value::Str(s) => {
            encode_length(out, MAJOR_TEXT, s.len() as u64);
            out.extend_from_slice(s.as_bytes());
        }
        Value::Tuple(items) => {
            encode_length(out, MAJOR_ARRAY, items.len() as u64);
            for item in items {
                encode(out, item);
            }
        }
        Value::Ident(_, inner) => encode(out, inner),
    }
}

/// `CBOREncoder.encode_int`: integers outside `[-2**64, 2**64)` become
/// bignums (tags 2 and 3) with a minimal big-endian magnitude.
#[inline]
fn encode_int(out: &mut Vec<u8>, v: i128) {
    if let Ok(u) = u64::try_from(v) {
        encode_length(out, MAJOR_UINT, u);
    } else if let Ok(n) = u64::try_from(-1 - v) {
        encode_length(out, MAJOR_NEGINT, n);
    } else {
        let (tag, magnitude) = if v >= 0 {
            (TAG_POS_BIGNUM, v as u128)
        } else {
            (TAG_NEG_BIGNUM, (-1 - v) as u128)
        };
        let be = magnitude.to_be_bytes();
        let skip = (magnitude.leading_zeros() / 8) as usize;
        encode_length(out, MAJOR_TAG, tag);
        encode_length(out, MAJOR_BYTES, (16 - skip) as u64);
        out.extend_from_slice(&be[skip..]);
    }
}

/// `CBOREncoder.encode_length`: the shortest argument encoding.
#[inline]
fn encode_length(out: &mut Vec<u8>, major: u8, len: u64) {
    let m = major << 5;
    if len < 24 {
        out.push(m | len as u8);
    } else if len <= 0xff {
        out.extend_from_slice(&[m | 24, len as u8]);
    } else if len <= 0xffff {
        out.push(m | 25);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else if len <= 0xffff_ffff {
        out.push(m | 26);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&len.to_be_bytes());
    }
}
