// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/structured_output/backend_xgrammar.py`: `XgrammarBackend`
//! and `XgrammarGrammar`.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{Value, json};
use vllm_xgrammar::{
    CompiledGrammar, GrammarCompiler, GrammarMatcher, JsonSchemaOptions, TokenizerInfo,
};

use super::utils::{choice_as_grammar, grammar_is_likely_lark};
use super::{Error, StructuredOutputOptions};

/// The vLLM settings `XgrammarBackend` reads.
#[derive(Clone, Debug)]
pub struct XgrammarBackendConfig {
    /// `structured_outputs_config.disable_any_whitespace`.
    pub disable_any_whitespace: bool,
    /// `speculative_config.num_speculative_tokens`, or 0.
    pub num_speculative_tokens: usize,
    /// `VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024`.
    pub cache_limit_bytes: i64,
    /// `VLLM_REGEX_COMPILATION_TIMEOUT_S`; `None` (Python: `<= 0`) disables
    /// the timeout.
    pub regex_compilation_timeout: Option<Duration>,
}

impl Default for XgrammarBackendConfig {
    /// vLLM's defaults.
    fn default() -> Self {
        Self {
            disable_any_whitespace: false,
            num_speculative_tokens: 0,
            cache_limit_bytes: 512 * 1024 * 1024,
            regex_compilation_timeout: Some(Duration::from_secs(5)),
        }
    }
}

/// `XgrammarBackend`: compiles grammars against the model's tokenizer.
pub struct XgrammarBackend {
    config: XgrammarBackendConfig,
    vocab_size: usize,
    /// Compiler for json. Must NOT accept any special tokens.
    compiler: Arc<GrammarCompiler>,
    /// Compiler for structural tag. May accept some of the special tokens.
    full_vocab_compiler: Arc<GrammarCompiler>,
}

impl XgrammarBackend {
    /// `max_threads` of each `xgr.GrammarCompiler`.
    const COMPILER_MAX_THREADS: i32 = 8;

    /// Creates the backend from the tokenizer info for json (no special
    /// tokens) and, if it differs, the full-vocabulary one for structural
    /// tags and Lark grammars.
    ///
    /// FIXME(arpera) in Python: two compilers are needed because xgrammar
    /// does not properly handle special tokens
    /// (https://github.com/mlc-ai/xgrammar/issues/832).
    pub fn new(
        config: XgrammarBackendConfig,
        vocab_size: usize,
        tokenizer_info: &TokenizerInfo,
        full_vocab_tokenizer_info: Option<&TokenizerInfo>,
    ) -> Result<Self, Error> {
        let create = |info: &TokenizerInfo| {
            GrammarCompiler::new(
                info,
                Self::COMPILER_MAX_THREADS,
                true,
                config.cache_limit_bytes,
            )
            .map(Arc::new)
        };
        let compiler = create(tokenizer_info)?;
        let full_vocab_compiler = match full_vocab_tokenizer_info {
            Some(info) => create(info)?,
            None => compiler.clone(),
        };
        Ok(Self {
            config,
            vocab_size,
            compiler,
            full_vocab_compiler,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// `compile_grammar`: compiles `grammar_spec` and creates its matcher.
    ///
    /// `stop_token_ids` (`sampling_params.all_stop_token_ids`) override the
    /// tokenizer's stop tokens when non-empty.
    pub fn compile_grammar(
        &self,
        request_type: StructuredOutputOptions,
        grammar_spec: &str,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<XgrammarGrammar, Error> {
        let ctx = self.compile(request_type, grammar_spec)?;
        let matcher = GrammarMatcher::new(
            &ctx,
            stop_token_ids.filter(|ids| !ids.is_empty()),
            false,
            self.config.num_speculative_tokens as i32,
            self.vocab_size,
        )?;
        Ok(XgrammarGrammar::new(matcher))
    }

    fn compile(
        &self,
        request_type: StructuredOutputOptions,
        grammar_spec: &str,
    ) -> Result<CompiledGrammar, Error> {
        // Note(arpera): `disable_any_whitespace` does NOT map directly to
        // xgrammar's `any_whitespace`; the separators must be given too.
        //
        // FIXME(arpera): xgrammar v0.2.8 DOES emit spaces after comma even if
        // we specify it in separators list
        // (https://github.com/mlc-ai/xgrammar/issues/945).
        let json_options = || JsonSchemaOptions {
            any_whitespace: !self.config.disable_any_whitespace,
            separators: self
                .config
                .disable_any_whitespace
                .then(|| (",".to_owned(), ":".to_owned())),
            ..JsonSchemaOptions::default()
        };
        let ctx = match request_type {
            StructuredOutputOptions::Json => {
                self.compiler.compile_json_schema(grammar_spec, &json_options())?
            }
            StructuredOutputOptions::JsonObject => {
                self.compiler.compile_json_schema(r#"{"type": "object"}"#, &json_options())?
            }
            StructuredOutputOptions::Grammar => {
                if grammar_is_likely_lark(grammar_spec) {
                    self.full_vocab_compiler.compile_lark(grammar_spec)?
                } else {
                    self.compiler.compile_grammar(grammar_spec, "root")?
                }
            }
            StructuredOutputOptions::Regex => self.compile_regex_with_timeout(grammar_spec)?,
            StructuredOutputOptions::Choice => {
                // xgrammar has no choice type. Python request validation rewrites
                // `choice` into this grammar, but other frontends send it as is.
                let choice: Vec<String> =
                    serde_json::from_str(grammar_spec).map_err(|e| Error {
                        message: format!("invalid choice: {e}"),
                    })?;
                self.compiler.compile_grammar(&choice_as_grammar(&choice), "root")?
            }
            StructuredOutputOptions::StructuralTag => {
                let s_tag: Value = serde_json::from_str(grammar_spec).map_err(|e| Error {
                    message: format!("invalid structural tag: {e}"),
                })?;
                if s_tag.get("structures").is_some() {
                    // Falling back to deprecated method of compiling structural tag
                    let converted = legacy_structural_tag_json(&s_tag)?;
                    self.full_vocab_compiler.compile_structural_tag(&converted)?
                } else {
                    self.full_vocab_compiler.compile_structural_tag(grammar_spec)?
                }
            }
        };
        Ok(ctx)
    }

    /// `compile_regex_with_timeout(self.compiler.compile_regex, pattern)`.
    ///
    /// Prevents ReDoS attacks where adversarial regex patterns (e.g. nested
    /// quantifiers like `(a+)+b`) cause exponential DFA state-space explosion,
    /// hanging the inference worker indefinitely. As in Python, a timed-out
    /// compilation keeps running in the background.
    fn compile_regex_with_timeout(&self, pattern: &str) -> Result<CompiledGrammar, Error> {
        let Some(timeout) = self.config.regex_compilation_timeout else {
            return Ok(self.compiler.compile_regex(pattern)?);
        };
        let (tx, rx) = mpsc::channel();
        let compiler = self.compiler.clone();
        let owned = pattern.to_owned();
        std::thread::spawn(move || {
            let _ = tx.send(compiler.compile_regex(&owned));
        });
        match rx.recv_timeout(timeout) {
            Ok(result) => Ok(result?),
            Err(_) => {
                let head: String = pattern.chars().take(200).collect();
                Err(Error {
                    message: format!(
                        "Regex compilation timed out after {}s. The pattern may be too \
                         complex or contain constructs that cause exponential state-space \
                         explosion (e.g. nested quantifiers). Pattern: {head}",
                        timeout.as_secs()
                    ),
                })
            }
        }
    }
}

/// The JSON xgrammar's Python package sends to C++ for
/// `compile_structural_tag([StructuralTagItem(begin=s["begin"],
/// schema=json.dumps(s["schema"]), end=s["end"]) for s in s_tag["structures"]],
/// s_tag["triggers"])`: `StructuralTag.from_legacy_structural_tag(...)
/// .model_dump_json()`, with the pydantic defaults spelled out.
pub(super) fn legacy_structural_tag_json(s_tag: &Value) -> Result<String, Error> {
    let invalid = |what: &str| Error {
        message: format!("invalid structural tag: {what}"),
    };
    let structures = s_tag["structures"]
        .as_array()
        .ok_or_else(|| invalid("`structures` must be a list"))?;
    let triggers = s_tag
        .get("triggers")
        .and_then(Value::as_array)
        .filter(|t| t.iter().all(Value::is_string))
        .ok_or_else(|| invalid("`triggers` must be a list of strings"))?;
    let tags = structures
        .iter()
        .map(|s| {
            let begin = s.get("begin").filter(|v| v.is_string());
            let end = s.get("end").filter(|v| v.is_string());
            let schema = s.get("schema").filter(|v| v.is_object() || v.is_boolean());
            match (begin, schema, end) {
                (Some(begin), Some(schema), Some(end)) => Ok(json!({
                    "type": "tag",
                    "begin": begin,
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema,
                        "style": "json",
                        "any_order": false,
                        "max_whitespace_cnt": null,
                        "excludes": [],
                    },
                    "end": end,
                })),
                _ => Err(invalid(
                    "each structure needs string `begin`/`end` and an object `schema`",
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": triggers,
            "tags": tags,
            "at_least_one": false,
            "stop_after_first": false,
            "excludes": [],
        },
    });
    Ok(tag.to_string())
}

/// `XgrammarGrammar`: a request's matcher plus vLLM's bookkeeping.
pub struct XgrammarGrammar {
    matcher: GrammarMatcher,
    num_processed_tokens: usize,
    is_terminated: bool,
}

impl XgrammarGrammar {
    fn new(matcher: GrammarMatcher) -> Self {
        Self {
            matcher,
            num_processed_tokens: 0,
            is_terminated: false,
        }
    }

    /// The number of int32 words of a bitmask row.
    pub fn bitmask_size(&self) -> usize {
        self.matcher.bitmask_size()
    }

    pub fn num_processed_tokens(&self) -> usize {
        self.num_processed_tokens
    }

    fn accept_token(&mut self, token: i32) -> bool {
        self.matcher
            .accept_token(token)
            .unwrap_or_else(|e| panic!("xgrammar accept_token failed: {e}"))
    }

    fn rollback_matcher(&mut self, num_tokens: usize) {
        self.matcher
            .rollback(num_tokens)
            .unwrap_or_else(|e| panic!("xgrammar rollback failed: {e}"));
    }

    /// Accepts a list of tokens and advances the FSM.
    ///
    /// Returns True if all grammar-constrained tokens were accepted.
    /// Tokens after termination are ignored. Returns False if the FSM
    /// failed to advance.
    pub fn accept_tokens(&mut self, tokens: &[i32]) -> bool {
        if self.is_terminated {
            return true;
        }
        for (num_accepted, &token) in tokens.iter().enumerate() {
            if !self.accept_token(token) {
                if num_accepted > 0 {
                    self.rollback_matcher(num_accepted);
                    self.num_processed_tokens -= num_accepted;
                    self.is_terminated = self.matcher.is_terminated();
                }
                return false;
            }
            self.num_processed_tokens += 1;
            self.is_terminated = self.matcher.is_terminated();
            if self.is_terminated {
                break;
            }
        }
        true
    }

    /// Checks if the list of tokens are accepted by the FSM in sequence.
    /// Will not advance the FSM.
    ///
    /// Returns the length of the prefix of tokens that are accepted by the FSM.
    pub fn validate_tokens(&mut self, tokens: &[i32]) -> usize {
        if self.is_terminated {
            return 0;
        }
        let mut num_accepted = 0;
        for &token in tokens {
            if !self.accept_token(token) {
                break;
            }
            num_accepted += 1;
            if self.matcher.is_terminated() {
                break;
            }
        }
        if num_accepted > 0 {
            // Rollback the FSM to the initial state
            self.rollback_matcher(num_accepted);
        }
        num_accepted
    }

    pub fn rollback(&mut self, num_tokens: usize) {
        self.rollback_matcher(num_tokens);
        self.num_processed_tokens -= num_tokens;
        self.is_terminated = self.matcher.is_terminated();
    }

    /// `fill_bitmask(bitmask, idx)` on the row `bitmask[idx]`.
    pub fn fill_bitmask(&mut self, row: &mut [i32]) {
        self.matcher
            .fill_next_token_bitmask(row)
            .unwrap_or_else(|e| panic!("xgrammar fill_next_token_bitmask failed: {e}"));
    }

    pub fn is_terminated(&self) -> bool {
        self.is_terminated
    }

    pub fn reset(&mut self) {
        self.matcher.reset();
        self.num_processed_tokens = 0;
        self.is_terminated = false;
    }
}
