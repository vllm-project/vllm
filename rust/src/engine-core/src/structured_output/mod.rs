// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/structured_output` for the xgrammar backend: the
//! engine-level `StructuredOutputManager` and `XgrammarBackend`.
//!
//! Grammar compilation and matching run in xgrammar's C++ engine
//! (`vllm-xgrammar`, the same version as the Python wheel), so bitmasks,
//! accept/rollback, and termination are bitwise identical to Python.
//!
//! What stays with the caller:
//! - Tokenizer loading: [`tokenizer_info`] builds xgrammar's tokenizer info
//!   from the values Python reads from a Hugging Face fast tokenizer.
//! - Reasoning parsers, which are model specific: the manager takes one
//!   [`Reasoner`] per request (see [`reasoner`]).
//! - Request validation (`validate_xgrammar_grammar`), which runs in the
//!   frontend.
//!
//! TODO: the guidance, outlines, and lm-format-enforcer backends are not
//! ported.

mod backend_xgrammar;
pub mod reasoner;
#[cfg(test)]
mod tests;
pub mod tokenizer_info;
pub mod utils;

use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub use self::backend_xgrammar::{XgrammarBackend, XgrammarBackendConfig, XgrammarGrammar};
use self::reasoner::Reasoner;
use self::utils::strip_speculative_padding;
use crate::sched::request::Request;
use crate::sched::scheduler::{GrammarStatus, StructuredOutput};
pub use vllm_xgrammar as xgrammar;

/// An error from grammar compilation or tokenizer setup; `message` is what
/// Python's `str(exception)` would be for the same cause where the engine
/// raises it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    pub message: String,
}

impl From<vllm_xgrammar::Error> for Error {
    fn from(e: vllm_xgrammar::Error) -> Self {
        Self { message: e.message }
    }
}

/// `StructuredOutputOptions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StructuredOutputOptions {
    Json,
    JsonObject,
    Regex,
    Grammar,
    Choice,
    StructuralTag,
}

/// `StructuredOutputKey`: `(request_type, grammar_spec)`, as
/// `get_structured_output_key` builds it from `StructuredOutputsParams`.
pub type StructuredOutputKey = (StructuredOutputOptions, String);

/// The vLLM settings `StructuredOutputManager` reads.
#[derive(Clone, Debug)]
pub struct StructuredOutputConfig {
    /// `scheduler_config.max_num_seqs`.
    pub max_num_seqs: usize,
    /// `vllm_config.num_speculative_tokens`: speculative tokens, or the
    /// canvas length of diffusion LLMs.
    pub num_speculative_tokens: usize,
    /// `model_config.is_diffusion`.
    pub is_diffusion: bool,
    /// `structured_outputs_config.enable_in_reasoning`.
    pub enable_in_reasoning: bool,
    /// Compile grammars on a thread pool. Python disables this for the
    /// `external_launcher` executor, whose per-rank schedulers must see
    /// grammars become ready at the same step.
    pub async_grammar_compilation: bool,
    /// Grammar compilation threads; Python uses `(cpu_count + 1) // 2`.
    pub compile_threads: usize,
    /// Bitmask filling threads, used for batches of at least 32 requests
    /// (Python: non-spec batches above 128 requests, with
    /// `max(1, min(cpu_count // 2, 8))` threads); 1 fills serially.
    pub fill_bitmask_threads: usize,
}

impl StructuredOutputConfig {
    /// Python's thread counts for this machine.
    pub fn default_threads() -> (usize, usize) {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        (cpus.div_ceil(2), (cpus / 2).clamp(1, 8))
    }
}

/// Where tokens start being fed to the grammar and being masked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConstraintBounds {
    grammar_start: usize,
    constraint_start: usize,
}

const CONSTRAINED_FROM_START: ConstraintBounds = ConstraintBounds {
    grammar_start: 0,
    constraint_start: 0,
};
const NO_SPEC_TOKENS: ConstraintBounds = ConstraintBounds {
    grammar_start: 1,
    constraint_start: 1,
};

/// The token state of a request that the manager reads.
#[derive(Clone, Copy, Debug)]
pub struct RequestView<'a> {
    pub request_id: &'a str,
    pub use_structured_output: bool,
    pub prompt_token_ids: Option<&'a [i32]>,
    pub all_token_ids: &'a [i32],
}

impl<'a> From<&'a Request> for RequestView<'a> {
    fn from(request: &'a Request) -> Self {
        Self {
            request_id: &request.request_id,
            use_structured_output: request.use_structured_output(),
            prompt_token_ids: request.prompt_token_ids.as_deref(),
            all_token_ids: request.all_token_ids(),
        }
    }
}

/// The result slot of an in-flight compilation, set by the compile pool.
type PendingGrammar = Arc<Mutex<Option<Result<XgrammarGrammar, Error>>>>;

enum GrammarSlot {
    /// Compilation is in flight on the compile pool.
    Pending(PendingGrammar),
    Ready(Box<XgrammarGrammar>),
    Failed(Error),
}

impl GrammarSlot {
    /// `StructuredOutputRequest._check_grammar_completion`.
    fn poll(&mut self) {
        if let Self::Pending(slot) = self {
            let Some(result) = slot.lock().unwrap().take() else {
                return;
            };
            *self = match result {
                Ok(grammar) => Self::Ready(Box::new(grammar)),
                Err(e) => Self::Failed(e),
            };
        }
    }
}

/// `StructuredOutputRequest`: the per-request structured output state.
struct RequestState {
    grammar: GrammarSlot,
    reasoning_ended: Option<bool>,
    /// Request-local; reasoning parsers can depend on per-request kwargs.
    reasoner: Option<Box<dyn Reasoner>>,
    /// Implicit reasoning-end tokens the grammar rejects as the first content
    /// token; masked while the request reasons. Computed on first use.
    masked_reasoning_end_ids: Option<Vec<i32>>,
}

impl RequestState {
    fn grammar(&mut self) -> &mut XgrammarGrammar {
        self.grammar.poll();
        match &mut self.grammar {
            GrammarSlot::Ready(grammar) => grammar,
            GrammarSlot::Pending(_) => panic!("structured output grammar is not compiled yet"),
            GrammarSlot::Failed(e) => panic!("structured output grammar failed to compile: {e}"),
        }
    }
}

/// The scheduled bitmask: `num_rows` rows of `row_words` int32 words,
/// row-major. The `np.int32` array `grammar_bitmask` returns.
#[derive(Debug)]
pub struct GrammarBitmask<'a> {
    pub data: &'a [i32],
    pub num_rows: usize,
    pub row_words: usize,
}

/// Engine-level manager for structured output requests.
pub struct StructuredOutputManager {
    config: StructuredOutputConfig,
    backend: Arc<XgrammarBackend>,
    requests: FxHashMap<String, RequestState>,
    /// `_grammar_bitmask`: `max_num_seqs * (1 + num_speculative_tokens)`
    /// rows, allocated on first use. Like Python, rows a step does not write
    /// keep their previous contents.
    bitmask: Vec<i32>,
    compile_pool: Option<rayon::ThreadPool>,
    fill_pool: Option<rayon::ThreadPool>,
}

impl StructuredOutputManager {
    const FILL_BITMASK_PARALLEL_THRESHOLD: usize = 128;
    const FILL_BITMASK_PARALLEL_BATCH_SIZE: usize = 16;

    pub fn new(config: StructuredOutputConfig, backend: XgrammarBackend) -> Self {
        let pool = |n: usize, name: &'static str| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .thread_name(move |i| format!("{name}-{i}"))
                .build()
                .expect("failed to build structured output thread pool")
        };
        let compile_pool = config
            .async_grammar_compilation
            .then(|| pool(config.compile_threads, "so-compile"));
        let fill_pool =
            (config.fill_bitmask_threads > 1).then(|| pool(config.fill_bitmask_threads, "so-fill"));
        Self {
            config,
            backend: Arc::new(backend),
            requests: FxHashMap::default(),
            bitmask: Vec::new(),
            compile_pool,
            fill_pool,
        }
    }

    pub fn backend(&self) -> &XgrammarBackend {
        &self.backend
    }

    /// `grammar_init`: starts compiling the grammar of a new request.
    ///
    /// `stop_token_ids` is `sampling_params.all_stop_token_ids`; `reasoner`
    /// is the request's reasoning parser if the engine has one configured.
    pub fn grammar_init(
        &mut self,
        request_id: String,
        key: StructuredOutputKey,
        stop_token_ids: Option<Vec<i32>>,
        reasoner: Option<Box<dyn Reasoner>>,
    ) {
        let compile = {
            let backend = self.backend.clone();
            move || {
                let (request_type, grammar_spec) = key;
                // TODO: Python logs "Failed to compile grammar for request %s"
                // with the traceback.
                backend.compile_grammar(request_type, &grammar_spec, stop_token_ids.as_deref())
            }
        };
        let grammar = match &self.compile_pool {
            Some(pool) => {
                let slot = PendingGrammar::default();
                let result = slot.clone();
                pool.spawn(move || {
                    let grammar = compile();
                    *result.lock().unwrap() = Some(grammar);
                });
                GrammarSlot::Pending(slot)
            }
            None => match compile() {
                Ok(grammar) => GrammarSlot::Ready(Box::new(grammar)),
                Err(e) => GrammarSlot::Failed(e),
            },
        };
        self.requests.insert(
            request_id,
            RequestState {
                grammar,
                reasoning_ended: None,
                reasoner,
                masked_reasoning_end_ids: None,
            },
        );
    }

    /// Drops the state of a finished request (Python frees it with the
    /// request object).
    pub fn remove_request(&mut self, request_id: &str) {
        self.requests.remove(request_id);
    }

    /// Whether the manager holds the state of `request_id`.
    pub fn has_request(&self, request_id: &str) -> bool {
        self.requests.contains_key(request_id)
    }

    pub fn num_requests(&self) -> usize {
        self.requests.len()
    }

    /// The grammar state of a request, read by the scheduler while the
    /// request is in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`.
    pub fn grammar_status(&mut self, request_id: &str) -> GrammarStatus {
        let Some(state) = self.requests.get_mut(request_id) else {
            return GrammarStatus::Pending;
        };
        state.grammar.poll();
        match &state.grammar {
            GrammarSlot::Pending(_) => GrammarStatus::Pending,
            GrammarSlot::Ready(_) => GrammarStatus::Ready,
            GrammarSlot::Failed(e) => GrammarStatus::Failed {
                message: e.message.clone(),
            },
        }
    }

    /// Mutable access to a request's grammar, e.g. for tests.
    pub fn grammar_mut(&mut self, request_id: &str) -> Option<&mut XgrammarGrammar> {
        self.requests.get_mut(request_id).map(RequestState::grammar)
    }

    fn state(&mut self, request_id: &str) -> &mut RequestState {
        self.requests
            .get_mut(request_id)
            .unwrap_or_else(|| panic!("no structured output state for request {request_id}"))
    }

    /// Implicit reasoning-end tokens the grammar rejects as first content.
    ///
    /// Such a token (e.g. a tool-call start) ends reasoning and is also the
    /// first content token, so sampling one where the grammar cannot start
    /// with it would make the output invalid. These stay masked while the
    /// request reasons. The grammar does not advance before reasoning ends,
    /// so the set is fixed per request.
    fn masked_reasoning_ends(state: &mut RequestState) -> &[i32] {
        if state.masked_reasoning_end_ids.is_none() {
            let mut masked = Vec::new();
            let implicit = state
                .reasoner
                .as_ref()
                .map(|r| {
                    let mut ids = r.implicit_reasoning_end_token_ids().to_vec();
                    ids.sort_unstable();
                    ids.dedup();
                    ids
                })
                .unwrap_or_default();
            if !implicit.is_empty() {
                let grammar = state.grammar();
                masked
                    .extend(implicit.into_iter().filter(|&id| grammar.validate_tokens(&[id]) == 0));
            }
            state.masked_reasoning_end_ids = Some(masked);
        }
        state.masked_reasoning_end_ids.as_deref().unwrap()
    }

    /// Return the index into `spec_tokens` where tokens should start being
    /// constrained by the grammar.
    /// Assumes `spec_tokens` are stripped of -1 padding tokens.
    /// Returns `len(spec_tokens) + 1` if no token should be constrained after
    /// accepting all `spec_tokens`.
    /// This is `constraint_start`; `grammar_start` is one less if reasoning ends
    /// implicitly on a content token (e.g. a tool-call start).
    ///
    /// `spec_tokens_committed` is True if spec_tokens is already in
    /// `request.all_token_ids`.
    fn get_constraint_bounds(
        enable_in_reasoning: bool,
        state: &mut RequestState,
        request: RequestView<'_>,
        spec_tokens: &[i32],
        spec_tokens_committed: bool,
    ) -> ConstraintBounds {
        if enable_in_reasoning || state.reasoning_ended == Some(true) {
            return CONSTRAINED_FROM_START;
        }
        let Some(reasoner) = state.reasoner.as_deref_mut() else {
            return CONSTRAINED_FROM_START;
        };

        if state.reasoning_ended.is_none() {
            // This should be removed here, but since `openai_gptoss`
            // is an independent code path, it is kept for now.
            // After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
            // it can be removed.
            if reasoner.is_reasoning_end(request.prompt_token_ids.unwrap_or(&[])) {
                return CONSTRAINED_FROM_START;
            }
            state.reasoning_ended = Some(false);
        }

        let num_spec_tokens = spec_tokens.len();
        if num_spec_tokens == 0 {
            return NO_SPEC_TOKENS;
        }
        let bounds = |grammar_start, constraint_start| ConstraintBounds {
            grammar_start,
            constraint_start,
        };

        // Parsers that know their end tokens also report whether one is content.
        if !reasoner.reasoning_end_token_ids().is_empty()
            && let Some(end) = reasoner.find_reasoning_end(spec_tokens)
        {
            let constraint_start = end.offset + 1;
            if end.implicit {
                return bounds(end.offset, constraint_start);
            }
            return bounds(constraint_start, constraint_start);
        }

        // Fallback: the grammar starts after the end token.
        let input_ids: Vec<i32>;
        let mut input_ids: &[i32] = if spec_tokens_committed {
            request.all_token_ids
        } else {
            input_ids = [request.all_token_ids, spec_tokens].concat();
            &input_ids
        };
        let mut delta_ids = spec_tokens;
        if !reasoner.is_reasoning_end_streaming(input_ids, delta_ids) {
            return bounds(num_spec_tokens + 1, num_spec_tokens + 1);
        }
        for i in (1..num_spec_tokens).rev() {
            input_ids = &input_ids[..input_ids.len() - 1];
            delta_ids = &delta_ids[..delta_ids.len() - 1];
            if !reasoner.is_reasoning_end_streaming(input_ids, delta_ids) {
                return bounds(i + 1, i + 1);
            }
        }
        bounds(1, 1)
    }

    /// Trim structured-output padding and grammar-invalid suffixes.
    pub fn trim_spec_tokens(
        &mut self,
        request: RequestView<'_>,
        spec_tokens: Vec<i32>,
    ) -> Vec<i32> {
        if !request.use_structured_output {
            // Synchronous unstructured decoding uses -1 values as scheduler-side
            // markers for speculative slots, so preserve their length.
            return spec_tokens;
        }
        let enable_in_reasoning = self.config.enable_in_reasoning;
        let state = self.state(request.request_id);
        let mut spec_tokens = spec_tokens;
        let unpadded = strip_speculative_padding(&spec_tokens).len();
        spec_tokens.truncate(unpadded);

        let ConstraintBounds { grammar_start, .. } =
            Self::get_constraint_bounds(enable_in_reasoning, state, request, &spec_tokens, false);
        if grammar_start >= spec_tokens.len() {
            return spec_tokens;
        }
        let grammar = state.grammar();
        if grammar.is_terminated() {
            spec_tokens.truncate(grammar_start);
            return spec_tokens;
        }
        let len_validated = grammar.validate_tokens(&spec_tokens[grammar_start..]);
        spec_tokens.truncate((grammar_start + len_validated + 1).min(spec_tokens.len()));
        spec_tokens
    }

    /// Advance grammar with accepted tokens. Returns False on rejection.
    pub fn accept_tokens(&mut self, request: RequestView<'_>, new_token_ids: &[i32]) -> bool {
        if !request.use_structured_output || new_token_ids.is_empty() {
            return true;
        }
        let enable_in_reasoning = self.config.enable_in_reasoning;
        let state = self.state(request.request_id);
        let ConstraintBounds { grammar_start, .. } =
            Self::get_constraint_bounds(enable_in_reasoning, state, request, new_token_ids, true);
        // Early return only when no reasoning-end token is present.
        // Otherwise, latch structured_req.reasoning_ended.
        if grammar_start > new_token_ids.len() {
            return true;
        }
        state.reasoning_ended = Some(true);
        state.grammar().accept_tokens(&new_token_ids[grammar_start..])
    }

    /// `grammar_bitmask`: the bitmask rows for the scheduled structured
    /// output requests, in the order of `requests`, or `None` if there are
    /// none.
    ///
    /// `scheduled_spec_decode_tokens` maps a request id to its scheduled
    /// draft tokens (missing: none).
    pub fn grammar_bitmask<'r, F>(
        &mut self,
        requests: &[RequestView<'r>],
        scheduled_spec_decode_tokens: F,
    ) -> Option<GrammarBitmask<'_>>
    where
        F: Fn(&str) -> Option<&'r [i32]>,
    {
        // Prepare the structured output bitmask for this batch.
        if requests.is_empty() {
            return None;
        }

        // Covers both speculative decoding and diffusion LLMs (canvas_length).
        let max_num_spec_tokens = self.config.num_speculative_tokens;
        let is_diffusion = self.config.is_diffusion;
        let mask_stride = max_num_spec_tokens + usize::from(!is_diffusion);
        let row_words = vllm_xgrammar::bitmask_size(self.backend.vocab_size());

        if self.bitmask.is_empty() {
            // Allocate a bitmask for each token needing to be checked:
            // one for each speculative position, and one more for the
            // bonus token / non-speculative token.
            // `xgr.allocate_token_bitmask` fills it with -1.
            let rows = self.config.max_num_seqs * (1 + max_num_spec_tokens);
            self.bitmask = vec![-1; rows * row_words];
        }

        // Python fills non-spec batches above the threshold on a thread pool
        // and ignores `scheduled_spec_decode_tokens` there; the rows are the
        // same as the serial path's for empty drafts.
        let ignore_drafts =
            requests.len() > Self::FILL_BITMASK_PARALLEL_THRESHOLD && max_num_spec_tokens == 0;

        // Reasoning bounds are computed serially, in request order, since
        // they call the request's reasoner.
        let enable_in_reasoning = self.config.enable_in_reasoning;
        let mut plans: FxHashMap<&str, RowsPlan<'r>> = FxHashMap::default();
        for (grammar_idx, &request) in requests.iter().enumerate() {
            let state = self.state(request.request_id);
            let req_tokens = if ignore_drafts {
                &[][..]
            } else {
                scheduled_spec_decode_tokens(request.request_id).unwrap_or(&[])
            };
            let unpadded_tokens = strip_speculative_padding(req_tokens);
            let bounds = Self::get_constraint_bounds(
                enable_in_reasoning,
                state,
                request,
                unpadded_tokens,
                false,
            );
            // Rows before constraint_start are still reasoning.
            let masked_ids = if bounds.constraint_start > 0 {
                Self::masked_reasoning_ends(state).to_vec()
            } else {
                Vec::new()
            };
            // Diffusion LLMs don't sample a bonus token after the
            // scheduled positions, so skip its bitmask in that case.
            let bonus = !is_diffusion || req_tokens.is_empty();
            plans.insert(
                request.request_id,
                RowsPlan {
                    base_index: grammar_idx * mask_stride,
                    req_tokens,
                    bonus,
                    bounds,
                    masked_ids,
                },
            );
        }

        // Each request then writes its own rows with its own grammar. When the
        // blocks of rows are disjoint (always, unless a request has more
        // drafts than `num_speculative_tokens`), requests run in parallel;
        // the result does not depend on the order.
        let disjoint = plans.values().all(|p| p.num_rows() <= mask_stride);
        let parallel = disjoint
            && requests.len() >= Self::FILL_BITMASK_PARALLEL_BATCH_SIZE * 2
            && self.fill_pool.is_some();
        let mut jobs: Vec<(&mut RequestState, RowsPlan<'r>)> = self
            .requests
            .iter_mut()
            .filter_map(|(id, state)| Some((state, plans.remove(id.as_str())?)))
            .collect();
        if parallel {
            let mut blocks: Vec<Option<&mut [i32]>> =
                self.bitmask.chunks_mut(mask_stride * row_words).map(Some).collect();
            let mut jobs: Vec<_> = jobs
                .into_iter()
                .map(|(state, plan)| {
                    let block = blocks[plan.base_index / mask_stride]
                        .take()
                        .expect("duplicate structured output request");
                    (state, plan, block)
                })
                .collect();
            self.fill_pool.as_ref().unwrap().install(|| {
                jobs.par_chunks_mut(Self::FILL_BITMASK_PARALLEL_BATCH_SIZE).for_each(|chunk| {
                    for (state, plan, block) in chunk {
                        plan.fill(state.grammar(), block, row_words);
                    }
                });
            });
        } else {
            for (state, plan) in &mut jobs {
                let start = plan.base_index * row_words;
                plan.fill(state.grammar(), &mut self.bitmask[start..], row_words);
            }
        }

        let num_rows = (requests.len() * mask_stride).min(self.bitmask.len() / row_words);
        Some(GrammarBitmask {
            data: &self.bitmask[..num_rows * row_words],
            num_rows,
            row_words,
        })
    }
}

/// The bitmask rows of one request in `grammar_bitmask`.
struct RowsPlan<'r> {
    base_index: usize,
    /// The scheduled drafts, with -1 padding.
    req_tokens: &'r [i32],
    /// Whether a row follows the drafts for the bonus token.
    bonus: bool,
    bounds: ConstraintBounds,
    masked_ids: Vec<i32>,
}

impl RowsPlan<'_> {
    fn num_rows(&self) -> usize {
        self.req_tokens.len() + usize::from(self.bonus)
    }

    /// Fills the request's rows, starting at the first row of `rows`, by
    /// advancing the grammar through the drafts and rolling it back.
    fn fill(&self, grammar: &mut XgrammarGrammar, rows: &mut [i32], row_words: usize) {
        let ConstraintBounds {
            grammar_start,
            constraint_start,
        } = self.bounds;
        let bonus: &[i32] = if self.bonus { &[-1] } else { &[] };
        let mut state_advancements = 0;
        let mut padding_or_grammar_invalid = false;
        for (i, &token) in self.req_tokens.iter().chain(bonus).enumerate() {
            let row = &mut rows[i * row_words..(i + 1) * row_words];
            if padding_or_grammar_invalid {
                fill_row(grammar, row, false, &[]);
                continue;
            }

            let apply_bitmask = i >= constraint_start;
            let row_masked: &[i32] = if i < constraint_start {
                &self.masked_ids
            } else {
                &[]
            };
            fill_row(grammar, row, apply_bitmask, row_masked);
            if token == -1 {
                padding_or_grammar_invalid = true;
            } else if i >= grammar_start {
                if !grammar.is_terminated() && grammar.accept_tokens(&[token]) {
                    state_advancements += 1;
                } else {
                    padding_or_grammar_invalid = true;
                }
            }
        }

        if state_advancements > 0 {
            grammar.rollback(state_advancements);
        }
    }
}

/// `_fill_bitmasks` for one row.
///
/// A row the grammar does not constrain allows every token except
/// `masked_ids`, the implicit reasoning ends masked while reasoning.
fn fill_row(
    grammar: &mut XgrammarGrammar,
    row: &mut [i32],
    apply_bitmask: bool,
    masked_ids: &[i32],
) {
    if apply_bitmask && !grammar.is_terminated() {
        grammar.fill_bitmask(row);
    } else {
        // Note that for thinking support, we will need to
        // reset the relevant part of the bitmask for consequent
        // requests here.
        row.fill(-1);
        for &token_id in masked_ids {
            let word = &mut row[(token_id >> 5) as usize];
            *word = (*word as u32 & !(1u32 << (token_id & 31))) as i32;
        }
    }
}

impl StructuredOutput for StructuredOutputManager {
    fn grammar_status(&mut self, request: &Request) -> GrammarStatus {
        StructuredOutputManager::grammar_status(self, &request.request_id)
    }

    fn accept_tokens(&mut self, request: &Request, new_token_ids: &[i32]) -> bool {
        StructuredOutputManager::accept_tokens(self, request.into(), new_token_ids)
    }

    fn trim_spec_tokens(&mut self, request: &Request, spec_token_ids: Vec<i32>) -> Vec<i32> {
        StructuredOutputManager::trim_spec_tokens(self, request.into(), spec_token_ids)
    }
}
