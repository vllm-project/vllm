// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Byte-exact reproduction of CPython's `pickle.dumps(obj, protocol=5)` (the C
//! `_pickle` implementation) for [`Value`]s.

use rustc_hash::FxHashMap;

use super::extend_encoded;
use super::value::{ExtraKeys, ObjectId, Value};

const PROTO: u8 = 0x80;
const FRAME: u8 = 0x95;
const STOP: u8 = b'.';
const NONE: u8 = b'N';
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const BININT: u8 = b'J';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const LONG1: u8 = 0x8a;
const SHORT_BINBYTES: u8 = b'C';
const BINBYTES: u8 = b'B';
const BINBYTES8: u8 = 0x8e;
const SHORT_BINUNICODE: u8 = 0x8c;
const BINUNICODE: u8 = b'X';
const BINUNICODE8: u8 = 0x8d;
const EMPTY_TUPLE: u8 = b')';
const MARK: u8 = b'(';
const TUPLE: u8 = b't';
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const MEMOIZE: u8 = 0x94;
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';

/// `FRAME_SIZE_TARGET` in `_pickle.c`.
const FRAME_SIZE_TARGET: usize = 64 * 1024;
/// Frames shorter than this are emitted without a `FRAME` header.
const FRAME_SIZE_MIN: usize = 4;
const FRAME_HEADER_SIZE: usize = 9;

/// Serialize `value` like `pickle.dumps(value, protocol=5)` into `out`
/// (cleared first).
pub fn dumps_into(out: &mut Vec<u8>, value: &Value) {
    let mut p = Pickler::new(out);
    p.save(value);
    p.finish();
}

/// Serialize the block key `(parent, tuple(tokens), extra_keys)` like
/// `pickle.dumps(..., protocol=5)` into `out` (cleared first), without
/// materializing the token tuple.
pub fn dumps_block_key_into(
    out: &mut Vec<u8>,
    parent: &[u8],
    tokens: &[i32],
    extra_keys: ExtraKeys<'_>,
) {
    // Upper bound for the common case: 5 bytes per token plus framing.
    out.reserve(tokens.len() * 5 + parent.len() + 64);
    let mut p = Pickler::new(out);
    p.save_bytes(parent);
    p.save_ints(tokens);
    match extra_keys {
        ExtraKeys::Plain(keys) => p.save_opt(keys),
        ExtraKeys::Lookahead { keys, tokens } => {
            p.boundary();
            let len = 1 + tokens.len();
            if len > 3 {
                p.write(&[MARK]);
            }
            p.save_opt(keys);
            for &t in tokens {
                p.save_int(t.into());
            }
            p.end_tuple(len);
        }
    }
    p.end_tuple(3);
    p.finish();
}

/// A pickler mirroring the framing and memo of `_pickle.c`'s `PicklerObject`.
struct Pickler<'a> {
    out: &'a mut Vec<u8>,
    framing: bool,
    /// Offset of the reserved header of the open frame, if any.
    frame_start: Option<usize>,
    /// Number of memo entries, i.e. the index of the next `MEMOIZE`.
    memo_len: u32,
    /// Memo indices of the identified objects saved so far.
    memo: FxHashMap<ObjectId, u32>,
}

impl<'a> Pickler<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        out.clear();
        out.extend_from_slice(&[PROTO, 5]);
        Self {
            out,
            framing: true,
            frame_start: None,
            memo_len: 0,
            memo: FxHashMap::default(),
        }
    }

    fn finish(mut self) {
        self.write(&[STOP]);
        self.commit_frame();
    }

    /// `_Pickler_Write`: append, opening a new frame if none is open.
    #[inline]
    fn write(&mut self, data: &[u8]) {
        if self.framing && self.frame_start.is_none() {
            self.frame_start = Some(self.out.len());
            self.out.extend_from_slice(&[0; FRAME_HEADER_SIZE]);
        }
        self.out.extend_from_slice(data);
    }

    /// `_Pickler_CommitFrame`.
    fn commit_frame(&mut self) {
        let Some(start) = self.frame_start.take() else {
            return;
        };
        let len = self.out.len() - start - FRAME_HEADER_SIZE;
        if len >= FRAME_SIZE_MIN {
            self.out[start] = FRAME;
            self.out[start + 1..start + FRAME_HEADER_SIZE]
                .copy_from_slice(&(len as u64).to_le_bytes());
        } else {
            self.out.drain(start..start + FRAME_HEADER_SIZE);
        }
    }

    /// `_Pickler_OpcodeBoundary`: commit the open frame once it reaches the
    /// size target. `_pickle.c` runs it at the start of every `save()`, so a
    /// frame may end between an object's opcodes but never inside one.
    #[inline]
    fn boundary(&mut self) {
        if let Some(start) = self.frame_start
            && self.out.len() - start - FRAME_HEADER_SIZE >= FRAME_SIZE_TARGET
        {
            self.commit_frame();
        }
    }

    /// `_Pickler_write_bytes`: payloads of at least the frame size target are
    /// written outside of any frame.
    fn write_large(&mut self, header: &[u8], payload: &[u8]) {
        if payload.len() >= FRAME_SIZE_TARGET {
            self.commit_frame();
            self.framing = false;
            self.write(header);
            self.write(payload);
            self.framing = true;
        } else {
            self.write(header);
            self.write(payload);
        }
    }

    fn memoize(&mut self) {
        self.write(&[MEMOIZE]);
        self.memo_len += 1;
    }

    /// `save()`, which every object, including each tuple element, goes
    /// through.
    fn save(&mut self, value: &Value) {
        self.boundary();
        self.save_body(value);
    }

    fn save_body(&mut self, value: &Value) {
        match value {
            Value::None => self.write(&[NONE]),
            Value::Bool(true) => self.write(&[NEWTRUE]),
            Value::Bool(false) => self.write(&[NEWFALSE]),
            Value::Int(v) => self.write_int(*v),
            Value::Bytes(b) => self.write_bytes(b),
            Value::Str(s) => self.write_str(s),
            Value::Tuple(items) => self.write_tuple(items),
            Value::Ident(id, inner) => {
                if let Some(&idx) = self.memo.get(id) {
                    self.write_get(idx);
                } else {
                    let memo_len = self.memo_len;
                    self.save_body(inner);
                    // Empty tuples (and non-memoizable values) aren't memoized.
                    if self.memo_len != memo_len {
                        self.memo.insert(*id, self.memo_len - 1);
                    }
                }
            }
        }
    }

    fn save_opt(&mut self, value: Option<&Value>) {
        match value {
            Some(v) => self.save(v),
            None => {
                self.boundary();
                self.write(&[NONE]);
            }
        }
    }

    #[inline]
    fn save_int(&mut self, v: i128) {
        self.boundary();
        self.write_int(v);
    }

    fn save_bytes(&mut self, b: &[u8]) {
        self.boundary();
        self.write_bytes(b);
    }

    /// A tuple of ints, as `save_tuple` with every element going through
    /// `save()`.
    fn save_ints(&mut self, tokens: &[i32]) {
        self.boundary();
        if tokens.is_empty() {
            self.write(&[EMPTY_TUPLE]);
            return;
        }
        if tokens.len() > 3 {
            self.write(&[MARK]);
        }
        let mut rest = tokens;
        while let Some((&first, tail)) = rest.split_first() {
            self.save_int(first.into());
            // The frame is open now. While it stays below the size target
            // (each int takes at most 5 bytes), `boundary()` is a no-op, so
            // the following ints can be written without checking.
            let start = self.frame_start.expect("write() opens a frame");
            let len = self.out.len() - start - FRAME_HEADER_SIZE;
            let n = (FRAME_SIZE_TARGET.saturating_sub(len) / 5).min(tail.len());
            extend_encoded(self.out, &tail[..n], binint);
            rest = &tail[n..];
        }
        self.end_tuple(tokens.len());
    }

    fn write_get(&mut self, idx: u32) {
        if idx < 256 {
            self.write(&[BINGET, idx as u8]);
        } else {
            let mut buf = [LONG_BINGET, 0, 0, 0, 0];
            buf[1..].copy_from_slice(&idx.to_le_bytes());
            self.write(&buf);
        }
    }

    /// `save_long`: `BININT*` for values in the `i32` range, else `LONG1`.
    #[inline]
    fn write_int(&mut self, v: i128) {
        if let Ok(v) = i32::try_from(v) {
            let (buf, n) = binint(v);
            self.write(&buf[..n]);
        } else {
            // Minimal little-endian two's complement, as `encode_long`.
            let mut bytes = [if v < 0 { 0xff } else { 0 }; 17];
            bytes[..16].copy_from_slice(&v.to_le_bytes());
            let bit_length = 128 - v.unsigned_abs().leading_zeros();
            let mut n = (bit_length >> 3) as usize + 1;
            if v < 0 && n > 1 && bytes[n - 1] == 0xff && bytes[n - 2] & 0x80 != 0 {
                n -= 1;
            }
            let mut buf = [0u8; 2 + 17];
            buf[0] = LONG1;
            buf[1] = n as u8;
            buf[2..2 + n].copy_from_slice(&bytes[..n]);
            self.write(&buf[..2 + n]);
        }
    }

    fn write_bytes(&mut self, b: &[u8]) {
        let len = b.len();
        if len <= 0xff {
            self.write_large(&[SHORT_BINBYTES, len as u8], b);
        } else if let Ok(len) = u32::try_from(len) {
            let l = len.to_le_bytes();
            self.write_large(&[BINBYTES, l[0], l[1], l[2], l[3]], b);
        } else {
            let mut h = [BINBYTES8; 9];
            h[1..].copy_from_slice(&(len as u64).to_le_bytes());
            self.write_large(&h, b);
        }
        self.memoize();
    }

    fn write_str(&mut self, s: &str) {
        let b = s.as_bytes();
        let len = b.len();
        if len <= 0xff {
            self.write_large(&[SHORT_BINUNICODE, len as u8], b);
        } else if let Ok(len) = u32::try_from(len) {
            let l = len.to_le_bytes();
            self.write_large(&[BINUNICODE, l[0], l[1], l[2], l[3]], b);
        } else {
            let mut h = [BINUNICODE8; 9];
            h[1..].copy_from_slice(&(len as u64).to_le_bytes());
            self.write_large(&h, b);
        }
        self.memoize();
    }

    fn write_tuple(&mut self, items: &[Value]) {
        if items.is_empty() {
            self.write(&[EMPTY_TUPLE]);
            return;
        }
        if items.len() > 3 {
            self.write(&[MARK]);
        }
        for item in items {
            self.save(item);
        }
        self.end_tuple(items.len());
    }

    /// Close a non-empty tuple whose `len` elements were saved.
    fn end_tuple(&mut self, len: usize) {
        match len {
            1 => self.write(&[TUPLE1]),
            2 => self.write(&[TUPLE2]),
            3 => self.write(&[TUPLE3]),
            _ => self.write(&[TUPLE]),
        }
        self.memoize();
    }
}

/// The `BININT*` opcode of an `i32` and its length (2, 3 or 5 bytes).
///
/// Branch-free, since token ids of random widths would mispredict.
#[inline(always)]
fn binint(v: i32) -> ([u8; 5], usize) {
    let u = v as u32;
    // Negative values are written as 4-byte `BININT`s.
    let le_ff = usize::from(u <= 0xff);
    let le_ffff = usize::from(u <= 0xffff);
    let op = [BININT, BININT2, BININT1][le_ff + le_ffff];
    let b = u.to_le_bytes();
    ([op, b[0], b[1], b[2], b[3]], 5 - 2 * le_ffff - le_ff)
}
