#![feature(coroutines)]

//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

pub use backend::{DynTextBackend, SamplingHints, TextBackend};
pub use error::{Error, Result};

mod backend;
pub mod backends;
mod error;
pub mod output;
