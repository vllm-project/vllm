// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bindings for the native xgrammar structured output manager
//! (`vllm_engine_core::structured_output`).
//!
//! Requests are passed as plain token lists rather than scheduler requests:
//! the Rust scheduler (`RustSchedulerCore.set_native_structured_output`)
//! shares the manager ([`RustStructuredOutputManager::shared`]) and calls it
//! directly through the `StructuredOutput` trait, while `grammar_init` stays
//! a method here.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rustc_hash::FxHashMap;
use vllm_engine_core::sched::scheduler::GrammarStatus;
use vllm_engine_core::structured_output::reasoner::{Reasoner, ReasoningEnd, ThinkTokenReasoner};
use vllm_engine_core::structured_output::tokenizer_info::HfTokenizerParts;
use vllm_engine_core::structured_output::{
    RequestView, StructuredOutputConfig, StructuredOutputManager, StructuredOutputOptions,
    XgrammarBackend, XgrammarBackendConfig,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustStructuredOutputManager>()?;
    m.add_function(wrap_pyfunction!(xgrammar_tokenizer_info_json, m)?)?;
    Ok(())
}

fn parse_request_type(name: &str) -> PyResult<StructuredOutputOptions> {
    Ok(match name {
        "JSON" => StructuredOutputOptions::Json,
        "JSON_OBJECT" => StructuredOutputOptions::JsonObject,
        "REGEX" => StructuredOutputOptions::Regex,
        "GRAMMAR" => StructuredOutputOptions::Grammar,
        "CHOICE" => StructuredOutputOptions::Choice,
        "STRUCTURAL_TAG" => StructuredOutputOptions::StructuralTag,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown request type: {name}"
            )));
        }
    })
}

/// The values `xgr.TokenizerInfo.from_huggingface` reads from a Hugging Face
/// fast tokenizer.
fn hf_tokenizer_parts(tokenizer: &Bound<'_, PyAny>) -> PyResult<HfTokenizerParts> {
    let py = tokenizer.py();
    let Ok(backend) = tokenizer.getattr("backend_tokenizer") else {
        return Err(PyNotImplementedError::new_err(
            "only Hugging Face fast tokenizers are supported",
        ));
    };
    let vocab_dict = tokenizer.call_method0("get_vocab")?.cast_into::<PyDict>()?;
    let vocab = vocab_dict
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<i32>()?)))
        .collect::<PyResult<Vec<_>>>()?;
    let backend_str = backend.call_method0("to_str")?.extract()?;
    let eos_token_id = match tokenizer.getattr("eos_token_id") {
        Ok(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let first_token_of_a = (|| -> PyResult<Option<String>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("add_special_tokens", false)?;
        let ids = tokenizer.call_method("encode", ("a",), Some(&kwargs))?;
        let tokens = tokenizer.call_method1("convert_ids_to_tokens", (ids,))?;
        let tokens = tokens.cast_into::<PyList>()?;
        Ok(match tokens.iter().next() {
            Some(t) if !t.is_none() => Some(t.extract()?),
            _ => None,
        })
    })()
    .unwrap_or(None);
    let mut special_token_ids = Vec::new();
    if let Ok(decoder) = tokenizer.getattr("added_tokens_decoder") {
        for (id, token) in decoder.cast_into::<PyDict>()?.iter() {
            if token.getattr("special")?.is_truthy()? {
                special_token_ids.push(id.extract()?);
            }
        }
    }
    Ok(HfTokenizerParts {
        vocab,
        backend_str,
        eos_token_id,
        first_token_of_a,
        special_token_ids,
    })
}

/// `serialize_json()` of the full-vocabulary and the special-token-filtered
/// tokenizer infos `XgrammarBackend` builds for `tokenizer`.
#[pyfunction]
fn xgrammar_tokenizer_info_json(
    tokenizer: &Bound<'_, PyAny>,
    vocab_size: usize,
) -> PyResult<(String, String)> {
    let infos = hf_tokenizer_parts(tokenizer)?
        .backend_tokenizer_infos(vocab_size)
        .map_err(|e| PyRuntimeError::new_err(e.message))?;
    let json = |info: &vllm_engine_core::structured_output::xgrammar::TokenizerInfo| {
        info.serialize_json().map_err(|e| PyRuntimeError::new_err(e.message))
    };
    Ok((json(&infos.full_vocab)?, json(&infos.filtered)?))
}

/// A Python `vllm.reasoning.ReasoningParser` instance as a [`Reasoner`].
struct PyReasoner {
    parser: Py<PyAny>,
    reasoning_end_token_ids: Vec<i32>,
    implicit_reasoning_end_token_ids: Vec<i32>,
}

impl PyReasoner {
    fn new(parser: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ids = |name: &str| -> PyResult<Vec<i32>> {
            let mut ids: Vec<i32> = parser
                .getattr(name)?
                .try_iter()?
                .map(|v| v?.extract())
                .collect::<PyResult<_>>()?;
            ids.sort_unstable();
            Ok(ids)
        };
        Ok(Self {
            parser: parser.clone().unbind(),
            reasoning_end_token_ids: ids("reasoning_end_token_ids")?,
            implicit_reasoning_end_token_ids: ids("implicit_reasoning_end_token_ids")?,
        })
    }

    fn call<T>(&self, f: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> T {
        Python::attach(|py| {
            f(self.parser.bind(py)).unwrap_or_else(|e| panic!("reasoning parser raised: {e}"))
        })
    }
}

impl Reasoner for PyReasoner {
    fn reasoning_end_token_ids(&self) -> &[i32] {
        &self.reasoning_end_token_ids
    }

    fn implicit_reasoning_end_token_ids(&self) -> &[i32] {
        &self.implicit_reasoning_end_token_ids
    }

    fn find_reasoning_end(&mut self, token_ids: &[i32]) -> Option<ReasoningEnd> {
        self.call(|p| {
            let end = p.call_method1("find_reasoning_end", (token_ids.to_vec(),))?;
            if end.is_none() {
                return Ok(None);
            }
            Ok(Some(ReasoningEnd {
                offset: end.getattr("offset")?.extract()?,
                implicit: end.getattr("implicit")?.is_truthy()?,
            }))
        })
    }

    fn is_reasoning_end(&mut self, input_ids: &[i32]) -> bool {
        self.call(|p| p.call_method1("is_reasoning_end", (input_ids.to_vec(),))?.is_truthy())
    }

    fn is_reasoning_end_streaming(&mut self, input_ids: &[i32], delta_ids: &[i32]) -> bool {
        self.call(|p| {
            p.call_method1(
                "is_reasoning_end_streaming",
                (input_ids.to_vec(), delta_ids.to_vec()),
            )?
            .is_truthy()
        })
    }
}

/// A request as `(request_id, prompt_token_ids | None, all_token_ids)`.
type PyRequest = (String, Option<Vec<i32>>, Vec<i32>);

fn view(request: &PyRequest) -> RequestView<'_> {
    RequestView {
        request_id: &request.0,
        use_structured_output: true,
        prompt_token_ids: request.1.as_deref(),
        all_token_ids: &request.2,
    }
}

/// `StructuredOutputManager` with the xgrammar backend, in Rust.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct RustStructuredOutputManager {
    inner: Arc<Mutex<StructuredOutputManager>>,
}

impl RustStructuredOutputManager {
    /// The manager, shared with the Rust scheduler.
    pub fn shared(&self) -> Arc<Mutex<StructuredOutputManager>> {
        self.inner.clone()
    }

    fn lock(&self) -> MutexGuard<'_, StructuredOutputManager> {
        self.inner.lock().unwrap()
    }

    fn with_grammar<R>(
        &self,
        request_id: &str,
        f: impl FnOnce(&mut vllm_engine_core::structured_output::XgrammarGrammar) -> R,
    ) -> PyResult<R> {
        let mut inner = self.lock();
        let grammar = inner
            .grammar_mut(request_id)
            .ok_or_else(|| PyKeyError::new_err(request_id.to_owned()))?;
        Ok(f(grammar))
    }
}

#[pymethods]
impl RustStructuredOutputManager {
    /// Builds the manager and its `XgrammarBackend` from a Hugging Face fast
    /// tokenizer, like `XgrammarBackend.__post_init__`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        tokenizer,
        vocab_size,
        *,
        max_num_seqs,
        num_speculative_tokens = 0,
        is_diffusion = false,
        enable_in_reasoning = false,
        disable_any_whitespace = false,
        async_grammar_compilation = true,
        cache_limit_bytes = 512 * 1024 * 1024,
        regex_compilation_timeout_s = 5,
        compile_threads = None,
        fill_bitmask_threads = None,
    ))]
    fn new(
        py: Python<'_>,
        tokenizer: &Bound<'_, PyAny>,
        vocab_size: usize,
        max_num_seqs: usize,
        num_speculative_tokens: usize,
        is_diffusion: bool,
        enable_in_reasoning: bool,
        disable_any_whitespace: bool,
        async_grammar_compilation: bool,
        cache_limit_bytes: i64,
        regex_compilation_timeout_s: i64,
        compile_threads: Option<usize>,
        fill_bitmask_threads: Option<usize>,
    ) -> PyResult<Self> {
        let parts = hf_tokenizer_parts(tokenizer)?;
        let (default_compile, default_fill) = StructuredOutputConfig::default_threads();
        let backend = py
            .detach(|| {
                let infos = parts.backend_tokenizer_infos(vocab_size)?;
                XgrammarBackend::new(
                    XgrammarBackendConfig {
                        disable_any_whitespace,
                        num_speculative_tokens,
                        cache_limit_bytes,
                        regex_compilation_timeout: (regex_compilation_timeout_s > 0)
                            .then(|| Duration::from_secs(regex_compilation_timeout_s as u64)),
                    },
                    vocab_size,
                    &infos.filtered,
                    Some(&infos.full_vocab),
                )
            })
            .map_err(|e| PyRuntimeError::new_err(e.message))?;
        let config = StructuredOutputConfig {
            max_num_seqs,
            num_speculative_tokens,
            is_diffusion,
            enable_in_reasoning,
            async_grammar_compilation,
            compile_threads: compile_threads.unwrap_or(default_compile),
            fill_bitmask_threads: fill_bitmask_threads.unwrap_or(default_fill),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(StructuredOutputManager::new(config, backend))),
        })
    }

    /// `grammar_init`. `request_type` is a `StructuredOutputOptions` name.
    /// `reasoner` is a Python reasoning parser instance, or `think_tokens`
    /// `(start_token_id, end_token_id)` selects the native
    /// `BaseThinkingReasoningParser` port.
    #[pyo3(signature = (request_id, request_type, grammar_spec, stop_token_ids=None, reasoner=None, think_tokens=None))]
    fn grammar_init(
        &self,
        request_id: String,
        request_type: &str,
        grammar_spec: String,
        stop_token_ids: Option<Vec<i32>>,
        reasoner: Option<&Bound<'_, PyAny>>,
        think_tokens: Option<(i32, i32)>,
    ) -> PyResult<()> {
        let request_type = parse_request_type(request_type)?;
        let reasoner: Option<Box<dyn Reasoner>> = match (reasoner, think_tokens) {
            (Some(parser), _) => Some(Box::new(PyReasoner::new(parser)?)),
            (None, Some((start_token_id, end_token_id))) => Some(Box::new(ThinkTokenReasoner {
                start_token_id,
                end_token_id,
            })),
            (None, None) => None,
        };
        self.lock().grammar_init(
            request_id,
            (request_type, grammar_spec),
            stop_token_ids,
            reasoner,
        );
        Ok(())
    }

    /// `"pending"`, `"ready"`, or `("failed", message)`.
    fn grammar_status<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let status = self.lock().grammar_status(request_id);
        Ok(match status {
            GrammarStatus::Pending => "pending".into_pyobject(py)?.into_any(),
            GrammarStatus::Ready => "ready".into_pyobject(py)?.into_any(),
            GrammarStatus::Failed { message } => {
                PyTuple::new(py, ["failed".to_owned(), message])?.into_any()
            }
        })
    }

    fn remove_request(&self, request_id: &str) {
        self.lock().remove_request(request_id);
    }

    fn has_request(&self, request_id: &str) -> bool {
        self.lock().has_request(request_id)
    }

    fn num_requests(&self) -> usize {
        self.lock().num_requests()
    }

    /// `accept_tokens(request, new_token_ids)`; `request` is
    /// `(request_id, prompt_token_ids, all_token_ids)`.
    fn accept_tokens(&self, request: PyRequest, new_token_ids: Vec<i32>) -> bool {
        self.lock().accept_tokens(view(&request), &new_token_ids)
    }

    /// `trim_spec_tokens(request, spec_tokens)`.
    fn trim_spec_tokens(&self, request: PyRequest, spec_tokens: Vec<i32>) -> Vec<i32> {
        self.lock().trim_spec_tokens(view(&request), spec_tokens)
    }

    /// `grammar_bitmask(requests, structured_output_request_ids,
    /// scheduled_spec_decode_tokens)` with `requests` in bitmask row order.
    /// Returns `(num_rows, row_words, bytes)` of the int32 array, or `None`.
    fn grammar_bitmask<'py>(
        &self,
        py: Python<'py>,
        requests: Vec<PyRequest>,
        scheduled_spec_decode_tokens: FxHashMap<String, Vec<i32>>,
    ) -> Option<(usize, usize, Bound<'py, PyBytes>)> {
        let views: Vec<_> = requests.iter().map(view).collect();
        py.detach(|| {
            let mut inner = self.lock();
            let bitmask = inner.grammar_bitmask(&views, |id| {
                scheduled_spec_decode_tokens.get(id).map(Vec::as_slice)
            })?;
            Some((
                bitmask.num_rows,
                bitmask.row_words,
                bytes_of(bitmask.data).to_vec(),
            ))
        })
        .map(|(rows, words, bytes)| (rows, words, PyBytes::new(py, &bytes)))
    }

    /// Times `grammar_bitmask` in Rust, excluding the Python conversion.
    /// Returns the elapsed seconds.
    fn time_grammar_bitmask(
        &self,
        py: Python<'_>,
        requests: Vec<PyRequest>,
        scheduled_spec_decode_tokens: FxHashMap<String, Vec<i32>>,
    ) -> f64 {
        let views: Vec<_> = requests.iter().map(view).collect();
        py.detach(|| {
            let mut inner = self.lock();
            let start = std::time::Instant::now();
            let rows = inner
                .grammar_bitmask(&views, |id| {
                    scheduled_spec_decode_tokens.get(id).map(Vec::as_slice)
                })
                .map_or(0, |b| b.num_rows);
            std::hint::black_box(rows);
            start.elapsed().as_secs_f64()
        })
    }

    // `XgrammarGrammar` methods of a request's compiled grammar.

    fn grammar_accept_tokens(&self, request_id: &str, tokens: Vec<i32>) -> PyResult<bool> {
        self.with_grammar(request_id, |g| g.accept_tokens(&tokens))
    }

    fn grammar_validate_tokens(&self, request_id: &str, tokens: Vec<i32>) -> PyResult<Vec<i32>> {
        let n = self.with_grammar(request_id, |g| g.validate_tokens(&tokens))?;
        Ok(tokens[..n].to_vec())
    }

    fn grammar_rollback(&self, request_id: &str, num_tokens: usize) -> PyResult<()> {
        self.with_grammar(request_id, |g| g.rollback(num_tokens))
    }

    fn grammar_is_terminated(&self, request_id: &str) -> PyResult<bool> {
        self.with_grammar(request_id, |g| g.is_terminated())
    }

    fn grammar_num_processed_tokens(&self, request_id: &str) -> PyResult<usize> {
        self.with_grammar(request_id, |g| g.num_processed_tokens())
    }

    fn grammar_reset(&self, request_id: &str) -> PyResult<()> {
        self.with_grammar(request_id, |g| g.reset())
    }

    /// `fill_bitmask` into a fresh all-zero row; returns its bytes.
    fn grammar_fill_bitmask<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let row = self.with_grammar(request_id, |g| {
            let mut row = vec![0i32; g.bitmask_size()];
            g.fill_bitmask(&mut row);
            row
        })?;
        Ok(PyBytes::new(py, bytes_of(&row)))
    }
}

pub fn bytes_of(words: &[i32]) -> &[u8] {
    // SAFETY: any initialized i32 slice is a valid byte slice.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
