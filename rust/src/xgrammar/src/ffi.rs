// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Raw declarations of the C ABI in `cpp/shim.cc`.

use std::ffi::c_char;

#[repr(C)]
pub struct TokenizerInfo {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Compiler {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CompiledGrammar {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Matcher {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn xgr_string_free(s: *mut c_char);
    pub fn xgr_get_bitmask_size(vocab_size: i32) -> i32;

    pub fn xgr_tokenizer_info_new(
        tokens: *const *const c_char,
        lens: *const usize,
        num_tokens: usize,
        vocab_type: i32,
        vocab_size: i32,
        has_vocab_size: bool,
        stop_token_ids: *const i32,
        num_stop_token_ids: usize,
        has_stop_token_ids: bool,
        add_prefix_space: bool,
        err: *mut *mut c_char,
    ) -> *mut TokenizerInfo;
    pub fn xgr_tokenizer_info_from_vocab_and_metadata(
        tokens: *const *const c_char,
        lens: *const usize,
        num_tokens: usize,
        metadata: *const c_char,
        metadata_len: usize,
        err: *mut *mut c_char,
    ) -> *mut TokenizerInfo;
    pub fn xgr_detect_metadata_from_hf(
        backend_str: *const c_char,
        len: usize,
        err: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn xgr_tokenizer_info_free(p: *mut TokenizerInfo);
    pub fn xgr_tokenizer_info_vocab_size(p: *const TokenizerInfo) -> i32;
    pub fn xgr_tokenizer_info_vocab_type(p: *const TokenizerInfo) -> i32;
    pub fn xgr_tokenizer_info_add_prefix_space(p: *const TokenizerInfo) -> bool;
    pub fn xgr_tokenizer_info_stop_token_ids(
        p: *const TokenizerInfo,
        len: *mut usize,
    ) -> *const i32;
    pub fn xgr_tokenizer_info_special_token_ids(
        p: *const TokenizerInfo,
        len: *mut usize,
    ) -> *const i32;
    pub fn xgr_tokenizer_info_dump_metadata(
        p: *const TokenizerInfo,
        err: *mut *mut c_char,
    ) -> *mut c_char;

    pub fn xgr_tokenizer_info_serialize_json(
        p: *const TokenizerInfo,
        err: *mut *mut c_char,
    ) -> *mut c_char;

    pub fn xgr_compiler_new(
        info: *const TokenizerInfo,
        max_threads: i32,
        cache_enabled: bool,
        cache_limit_bytes: i64,
        err: *mut *mut c_char,
    ) -> *mut Compiler;
    pub fn xgr_compiler_free(p: *mut Compiler);
    pub fn xgr_compiler_cache_size_bytes(p: *const Compiler) -> i64;
    pub fn xgr_compile_json_schema(
        c: *mut Compiler,
        schema: *const c_char,
        schema_len: usize,
        any_whitespace: bool,
        indent: i32,
        has_indent: bool,
        item_separator: *const c_char,
        item_separator_len: usize,
        key_separator: *const c_char,
        key_separator_len: usize,
        has_separators: bool,
        strict_mode: bool,
        max_whitespace_cnt: i32,
        has_max_whitespace_cnt: bool,
        any_order: bool,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compile_builtin_json(
        c: *mut Compiler,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compile_ebnf(
        c: *mut Compiler,
        ebnf: *const c_char,
        ebnf_len: usize,
        root_rule_name: *const c_char,
        root_rule_name_len: usize,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compile_lark(
        c: *mut Compiler,
        lark: *const c_char,
        lark_len: usize,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compile_regex(
        c: *mut Compiler,
        regex: *const c_char,
        regex_len: usize,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compile_structural_tag(
        c: *mut Compiler,
        tag_json: *const c_char,
        tag_json_len: usize,
        err: *mut *mut c_char,
    ) -> *mut CompiledGrammar;
    pub fn xgr_compiled_grammar_free(p: *mut CompiledGrammar);
    pub fn xgr_compiled_grammar_memory_size_bytes(p: *const CompiledGrammar) -> i64;

    pub fn xgr_matcher_new(
        cg: *const CompiledGrammar,
        override_stop_tokens: *const i32,
        num_override_stop_tokens: usize,
        has_override_stop_tokens: bool,
        terminate_without_stop_token: bool,
        max_rollback_tokens: i32,
        err: *mut *mut c_char,
    ) -> *mut Matcher;
    pub fn xgr_matcher_free(p: *mut Matcher);
    pub fn xgr_matcher_accept_token(m: *mut Matcher, token_id: i32, err: *mut *mut c_char) -> i32;
    pub fn xgr_matcher_accept_string(
        m: *mut Matcher,
        s: *const c_char,
        len: usize,
        err: *mut *mut c_char,
    ) -> i32;
    pub fn xgr_matcher_fill_next_token_bitmask(
        m: *mut Matcher,
        row: *mut i32,
        len: i64,
        err: *mut *mut c_char,
    ) -> i32;
    pub fn xgr_matcher_rollback(m: *mut Matcher, num_tokens: i32, err: *mut *mut c_char) -> i32;
    pub fn xgr_matcher_is_terminated(m: *const Matcher) -> bool;
    pub fn xgr_matcher_is_completed(m: *const Matcher) -> bool;
    pub fn xgr_matcher_reset(m: *mut Matcher);
    pub fn xgr_matcher_find_jump_forward_string(
        m: *mut Matcher,
        err: *mut *mut c_char,
    ) -> *mut c_char;
}
