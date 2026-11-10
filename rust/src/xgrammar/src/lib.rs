// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Safe bindings to the xgrammar C++ engine.
//!
//! The C++ sources under `vendor/xgrammar` are upstream xgrammar v0.2.8
//! (`cpp/`, `include/`, and the `picojson`/`dlpack` headers), i.e. the same
//! engine as the `xgrammar==0.2.8` Python wheel, so tokenizer info, compiled
//! grammars, token bitmasks, accept/rollback, and termination are identical
//! to Python's. Only the C ABI shim in `cpp/shim.cc` is ours.
//!
//! The API mirrors the subset of the Python package (`xgrammar.TokenizerInfo`,
//! `GrammarCompiler`, `GrammarMatcher`) that vLLM uses.

use std::ffi::{CStr, c_char};
use std::ptr::{self, NonNull};

mod ffi;

/// An error raised by the C++ engine; `message` is its `what()`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    pub message: String,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Calls `f` with an error out-parameter and converts a set error into `Err`.
fn call<T>(f: impl FnOnce(*mut *mut c_char) -> T) -> Result<T> {
    let mut err: *mut c_char = ptr::null_mut();
    let out = f(&mut err);
    if err.is_null() {
        Ok(out)
    } else {
        // SAFETY: the shim sets `err` to a malloc'ed NUL-terminated string.
        let message = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
        unsafe { ffi::xgr_string_free(err) };
        Err(Error { message })
    }
}

/// Takes ownership of a malloc'ed string returned by the shim.
fn take_string(s: *mut c_char) -> String {
    // SAFETY: `s` is a non-null malloc'ed NUL-terminated string from the shim.
    let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
    unsafe { ffi::xgr_string_free(s) };
    out
}

fn non_null<T>(p: *mut T) -> Result<NonNull<T>> {
    NonNull::new(p).ok_or_else(|| Error {
        message: "xgrammar returned a null object".to_owned(),
    })
}

/// `xgr.get_bitmask_size`: the number of int32 words per bitmask row.
pub fn bitmask_size(vocab_size: usize) -> usize {
    // SAFETY: pure function.
    unsafe { ffi::xgr_get_bitmask_size(vocab_size as i32) as usize }
}

/// `xgr.VocabType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum VocabType {
    Raw = 0,
    ByteFallback = 1,
    ByteLevel = 2,
}

impl VocabType {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Raw),
            1 => Some(Self::ByteFallback),
            2 => Some(Self::ByteLevel),
            _ => None,
        }
    }
}

/// Pointers and lengths of a vocabulary, borrowed for one FFI call.
struct VocabView {
    ptrs: Vec<*const c_char>,
    lens: Vec<usize>,
}

impl VocabView {
    fn new<S: AsRef<[u8]>>(vocab: &[S]) -> Self {
        Self {
            ptrs: vocab.iter().map(|t| t.as_ref().as_ptr().cast()).collect(),
            lens: vocab.iter().map(|t| t.as_ref().len()).collect(),
        }
    }
}

/// `xgr.TokenizerInfo`.
pub struct TokenizerInfo(NonNull<ffi::TokenizerInfo>);

// SAFETY: the C++ object is immutable after construction.
unsafe impl Send for TokenizerInfo {}
unsafe impl Sync for TokenizerInfo {}

impl Drop for TokenizerInfo {
    fn drop(&mut self) {
        unsafe { ffi::xgr_tokenizer_info_free(self.0.as_ptr()) }
    }
}

impl TokenizerInfo {
    /// `xgr.TokenizerInfo(encoded_vocab, vocab_type, vocab_size=,
    /// stop_token_ids=, add_prefix_space=)`.
    pub fn new<S: AsRef<[u8]>>(
        encoded_vocab: &[S],
        vocab_type: VocabType,
        vocab_size: Option<usize>,
        stop_token_ids: Option<&[i32]>,
        add_prefix_space: bool,
    ) -> Result<Self> {
        let view = VocabView::new(encoded_vocab);
        let stop = stop_token_ids.unwrap_or(&[]);
        let p = call(|err| unsafe {
            ffi::xgr_tokenizer_info_new(
                view.ptrs.as_ptr(),
                view.lens.as_ptr(),
                encoded_vocab.len(),
                vocab_type as i32,
                vocab_size.unwrap_or(0) as i32,
                vocab_size.is_some(),
                stop.as_ptr(),
                stop.len(),
                stop_token_ids.is_some(),
                add_prefix_space,
                err,
            )
        })?;
        non_null(p).map(Self)
    }

    /// `xgr.TokenizerInfo.from_vocab_and_metadata`.
    pub fn from_vocab_and_metadata<S: AsRef<[u8]>>(
        encoded_vocab: &[S],
        metadata: &str,
    ) -> Result<Self> {
        let view = VocabView::new(encoded_vocab);
        let p = call(|err| unsafe {
            ffi::xgr_tokenizer_info_from_vocab_and_metadata(
                view.ptrs.as_ptr(),
                view.lens.as_ptr(),
                encoded_vocab.len(),
                metadata.as_ptr().cast(),
                metadata.len(),
                err,
            )
        })?;
        non_null(p).map(Self)
    }

    /// `_core.TokenizerInfo._detect_metadata_from_hf`: the JSON metadata
    /// (`{"vocab_type": int, "add_prefix_space": bool}`) detected from a
    /// serialized Hugging Face fast tokenizer (`backend_tokenizer.to_str()`).
    pub fn detect_metadata_from_hf(backend_str: &str) -> Result<String> {
        let s = call(|err| unsafe {
            ffi::xgr_detect_metadata_from_hf(backend_str.as_ptr().cast(), backend_str.len(), err)
        })?;
        Ok(take_string(non_null(s)?.as_ptr()))
    }

    pub fn vocab_size(&self) -> usize {
        unsafe { ffi::xgr_tokenizer_info_vocab_size(self.0.as_ptr()) as usize }
    }

    pub fn vocab_type(&self) -> VocabType {
        let raw = unsafe { ffi::xgr_tokenizer_info_vocab_type(self.0.as_ptr()) };
        VocabType::from_i32(raw).expect("xgrammar returned an invalid vocab type")
    }

    pub fn add_prefix_space(&self) -> bool {
        unsafe { ffi::xgr_tokenizer_info_add_prefix_space(self.0.as_ptr()) }
    }

    pub fn stop_token_ids(&self) -> Vec<i32> {
        let mut len = 0;
        let p = unsafe { ffi::xgr_tokenizer_info_stop_token_ids(self.0.as_ptr(), &mut len) };
        slice_to_vec(p, len)
    }

    pub fn special_token_ids(&self) -> Vec<i32> {
        let mut len = 0;
        let p = unsafe { ffi::xgr_tokenizer_info_special_token_ids(self.0.as_ptr(), &mut len) };
        slice_to_vec(p, len)
    }

    /// `TokenizerInfo.dump_metadata()`.
    pub fn dump_metadata(&self) -> Result<String> {
        let s = call(|err| unsafe { ffi::xgr_tokenizer_info_dump_metadata(self.0.as_ptr(), err) })?;
        Ok(take_string(non_null(s)?.as_ptr()))
    }
}

impl TokenizerInfo {
    /// `TokenizerInfo.serialize_json()`, including the decoded vocabulary.
    pub fn serialize_json(&self) -> Result<String> {
        let s =
            call(|err| unsafe { ffi::xgr_tokenizer_info_serialize_json(self.0.as_ptr(), err) })?;
        Ok(take_string(non_null(s)?.as_ptr()))
    }
}

fn slice_to_vec(p: *const i32, len: usize) -> Vec<i32> {
    if len == 0 {
        return Vec::new();
    }
    // SAFETY: the shim returns a pointer to `len` ints owned by a live object.
    unsafe { std::slice::from_raw_parts(p, len) }.to_vec()
}

/// Options of `GrammarCompiler.compile_json_schema`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonSchemaOptions {
    pub any_whitespace: bool,
    pub indent: Option<i32>,
    /// `(item_separator, key_separator)`, as in `json.dumps`.
    pub separators: Option<(String, String)>,
    pub strict_mode: bool,
    pub max_whitespace_cnt: Option<i32>,
    pub any_order: bool,
}

impl Default for JsonSchemaOptions {
    /// The Python keyword defaults.
    fn default() -> Self {
        Self {
            any_whitespace: true,
            indent: None,
            separators: None,
            strict_mode: true,
            max_whitespace_cnt: None,
            any_order: false,
        }
    }
}

/// `xgr.GrammarCompiler`. Compilation is thread safe and cached by the
/// engine (`cache_enabled`, `cache_limit_bytes`).
pub struct GrammarCompiler(NonNull<ffi::Compiler>);

// SAFETY: `GrammarCompiler` is internally synchronized (Python calls it from a
// thread pool).
unsafe impl Send for GrammarCompiler {}
unsafe impl Sync for GrammarCompiler {}

impl Drop for GrammarCompiler {
    fn drop(&mut self) {
        unsafe { ffi::xgr_compiler_free(self.0.as_ptr()) }
    }
}

impl GrammarCompiler {
    /// `xgr.GrammarCompiler(tokenizer_info, max_threads=, cache_enabled=,
    /// cache_limit_bytes=)`; `cache_limit_bytes` of -1 is unlimited.
    pub fn new(
        tokenizer_info: &TokenizerInfo,
        max_threads: i32,
        cache_enabled: bool,
        cache_limit_bytes: i64,
    ) -> Result<Self> {
        let p = call(|err| unsafe {
            ffi::xgr_compiler_new(
                tokenizer_info.0.as_ptr(),
                max_threads,
                cache_enabled,
                cache_limit_bytes,
                err,
            )
        })?;
        non_null(p).map(Self)
    }

    pub fn cache_size_bytes(&self) -> i64 {
        unsafe { ffi::xgr_compiler_cache_size_bytes(self.0.as_ptr()) }
    }

    fn wrap(p: Result<*mut ffi::CompiledGrammar>) -> Result<CompiledGrammar> {
        non_null(p?).map(CompiledGrammar)
    }

    pub fn compile_json_schema(
        &self,
        schema: &str,
        options: &JsonSchemaOptions,
    ) -> Result<CompiledGrammar> {
        let (item, key) =
            options.separators.as_ref().map_or(("", ""), |(i, k)| (i.as_str(), k.as_str()));
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_json_schema(
                self.0.as_ptr(),
                schema.as_ptr().cast(),
                schema.len(),
                options.any_whitespace,
                options.indent.unwrap_or(0),
                options.indent.is_some(),
                item.as_ptr().cast(),
                item.len(),
                key.as_ptr().cast(),
                key.len(),
                options.separators.is_some(),
                options.strict_mode,
                options.max_whitespace_cnt.unwrap_or(0),
                options.max_whitespace_cnt.is_some(),
                options.any_order,
                err,
            )
        }))
    }

    pub fn compile_builtin_json_grammar(&self) -> Result<CompiledGrammar> {
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_builtin_json(self.0.as_ptr(), err)
        }))
    }

    /// `compile_grammar(ebnf_str, root_rule_name="root")`.
    pub fn compile_grammar(&self, ebnf: &str, root_rule_name: &str) -> Result<CompiledGrammar> {
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_ebnf(
                self.0.as_ptr(),
                ebnf.as_ptr().cast(),
                ebnf.len(),
                root_rule_name.as_ptr().cast(),
                root_rule_name.len(),
                err,
            )
        }))
    }

    /// `compile_lark(lark_string)` without named grammars.
    pub fn compile_lark(&self, lark: &str) -> Result<CompiledGrammar> {
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_lark(self.0.as_ptr(), lark.as_ptr().cast(), lark.len(), err)
        }))
    }

    pub fn compile_regex(&self, regex: &str) -> Result<CompiledGrammar> {
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_regex(self.0.as_ptr(), regex.as_ptr().cast(), regex.len(), err)
        }))
    }

    /// `compile_structural_tag(structural_tag_json)`.
    pub fn compile_structural_tag(&self, structural_tag_json: &str) -> Result<CompiledGrammar> {
        Self::wrap(call(|err| unsafe {
            ffi::xgr_compile_structural_tag(
                self.0.as_ptr(),
                structural_tag_json.as_ptr().cast(),
                structural_tag_json.len(),
                err,
            )
        }))
    }
}

/// `xgr.CompiledGrammar`.
pub struct CompiledGrammar(NonNull<ffi::CompiledGrammar>);

// SAFETY: the C++ object is immutable after compilation and shared by matchers.
unsafe impl Send for CompiledGrammar {}
unsafe impl Sync for CompiledGrammar {}

impl Drop for CompiledGrammar {
    fn drop(&mut self) {
        unsafe { ffi::xgr_compiled_grammar_free(self.0.as_ptr()) }
    }
}

impl CompiledGrammar {
    pub fn memory_size_bytes(&self) -> usize {
        unsafe { ffi::xgr_compiled_grammar_memory_size_bytes(self.0.as_ptr()) as usize }
    }
}

/// `xgr.GrammarMatcher`.
pub struct GrammarMatcher {
    ptr: NonNull<ffi::Matcher>,
    bitmask_size: usize,
}

// SAFETY: a matcher owns its state; `&mut self` serializes mutation, and the
// `&self` methods only read it.
unsafe impl Send for GrammarMatcher {}
unsafe impl Sync for GrammarMatcher {}

impl Drop for GrammarMatcher {
    fn drop(&mut self) {
        unsafe { ffi::xgr_matcher_free(self.ptr.as_ptr()) }
    }
}

impl GrammarMatcher {
    /// `xgr.GrammarMatcher(compiled_grammar, override_stop_tokens=,
    /// terminate_without_stop_token=, max_rollback_tokens=)`.
    ///
    /// `vocab_size` is the tokenizer info's vocab size, used to check
    /// bitmask row lengths.
    pub fn new(
        compiled_grammar: &CompiledGrammar,
        override_stop_tokens: Option<&[i32]>,
        terminate_without_stop_token: bool,
        max_rollback_tokens: i32,
        vocab_size: usize,
    ) -> Result<Self> {
        let stop = override_stop_tokens.unwrap_or(&[]);
        let p = call(|err| unsafe {
            ffi::xgr_matcher_new(
                compiled_grammar.0.as_ptr(),
                stop.as_ptr(),
                stop.len(),
                override_stop_tokens.is_some(),
                terminate_without_stop_token,
                max_rollback_tokens,
                err,
            )
        })?;
        Ok(Self {
            ptr: non_null(p)?,
            bitmask_size: bitmask_size(vocab_size),
        })
    }

    /// The number of int32 words of a bitmask row.
    pub fn bitmask_size(&self) -> usize {
        self.bitmask_size
    }

    pub fn accept_token(&mut self, token_id: i32) -> Result<bool> {
        let r =
            call(|err| unsafe { ffi::xgr_matcher_accept_token(self.ptr.as_ptr(), token_id, err) })?;
        Ok(r == 1)
    }

    pub fn accept_string(&mut self, s: &[u8]) -> Result<bool> {
        let r = call(|err| unsafe {
            ffi::xgr_matcher_accept_string(self.ptr.as_ptr(), s.as_ptr().cast(), s.len(), err)
        })?;
        Ok(r == 1)
    }

    /// `fill_next_token_bitmask` into one row of `bitmask_size()` words.
    /// Returns whether the mask needs to be applied (is not all-true).
    pub fn fill_next_token_bitmask(&mut self, row: &mut [i32]) -> Result<bool> {
        let r = call(|err| unsafe {
            ffi::xgr_matcher_fill_next_token_bitmask(
                self.ptr.as_ptr(),
                row.as_mut_ptr(),
                row.len() as i64,
                err,
            )
        })?;
        Ok(r == 1)
    }

    pub fn rollback(&mut self, num_tokens: usize) -> Result<()> {
        call(|err| unsafe {
            ffi::xgr_matcher_rollback(self.ptr.as_ptr(), num_tokens as i32, err)
        })?;
        Ok(())
    }

    pub fn is_terminated(&self) -> bool {
        unsafe { ffi::xgr_matcher_is_terminated(self.ptr.as_ptr()) }
    }

    pub fn is_completed(&self) -> bool {
        unsafe { ffi::xgr_matcher_is_completed(self.ptr.as_ptr()) }
    }

    pub fn reset(&mut self) {
        unsafe { ffi::xgr_matcher_reset(self.ptr.as_ptr()) }
    }

    pub fn find_jump_forward_string(&mut self) -> Result<String> {
        let s = call(|err| unsafe {
            ffi::xgr_matcher_find_jump_forward_string(self.ptr.as_ptr(), err)
        })?;
        Ok(take_string(non_null(s)?.as_ptr()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_info() -> TokenizerInfo {
        let vocab = ["{", "}", "\"", "a", ":", " ", "1", "</s>"];
        TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(&[7]), false).unwrap()
    }

    #[test]
    fn json_object_matcher() {
        let info = tiny_info();
        assert_eq!(info.vocab_size(), 8);
        assert_eq!(info.stop_token_ids(), vec![7]);
        let compiler = GrammarCompiler::new(&info, 1, true, -1).unwrap();
        let cg = compiler
            .compile_json_schema(r#"{"type": "object"}"#, &JsonSchemaOptions::default())
            .unwrap();
        let mut m = GrammarMatcher::new(&cg, None, false, -1, info.vocab_size()).unwrap();
        let mut row = vec![0i32; m.bitmask_size()];
        assert!(m.fill_next_token_bitmask(&mut row).unwrap());
        // Only `{` (and whitespace) can start an object.
        assert_eq!(row[0] & 1, 1);
        assert_eq!(row[0] & (1 << 1), 0);
        assert!(m.accept_token(0).unwrap());
        assert!(!m.accept_token(6).unwrap());
        assert!(m.accept_token(1).unwrap());
        assert!(!m.is_terminated());
        assert!(m.accept_token(7).unwrap());
        assert!(m.is_terminated());
        m.rollback(2).unwrap();
        assert!(!m.is_terminated());
        assert!(m.accept_token(1).unwrap());
    }

    #[test]
    fn errors_carry_engine_message() {
        let info = tiny_info();
        let compiler = GrammarCompiler::new(&info, 1, true, -1).unwrap();
        let err = compiler.compile_grammar("root ::= ", "root").err().unwrap();
        assert!(!err.message.is_empty());
        let err = compiler
            .compile_json_schema("{not json", &JsonSchemaOptions::default())
            .err()
            .unwrap();
        assert!(!err.message.is_empty());
    }
}
