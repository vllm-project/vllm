use std::sync::Arc;

use crate::error::Result;

/// Tokenizer/model-derived hints used to enrich text-generation requests before they are lowered
/// into engine-core.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: std::collections::BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
    /// Model context window size (`max_position_embeddings` from `config.json`).
    pub max_model_len: Option<u32>,
}

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return the newly decoded text chunk, if any.
    ///
    /// Returns `Ok(None)` when the token does not yet produce a stable text fragment (e.g. in the
    /// middle of a multi-byte UTF-8 sequence).
    fn step(&mut self, token_id: u32) -> Result<Option<String>>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out incomplete fragments.
    fn flush(&mut self) -> Result<Option<String>>;
}

/// Minimal text-processing backend needed by `vllm-text`.
pub trait TextBackend: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Create a stateful incremental decoder primed with the given prompt tokens.
    ///
    /// The prompt tokens provide left context for the first generated token; the decoder does not
    /// re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream {
            backend: self,
            skip_special_tokens,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
        })
    }

    /// Return the backend model ID when available.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Return tokenizer/model-derived hints used to enrich southbound sampling parameters.
    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints::default())
    }
}

/// Shared trait-object form of [`TextBackend`].
pub type DynTextBackend = Arc<dyn TextBackend>;

/// [`IncrementalDecoder`] built on [`TextBackend::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream` and
/// `fastokens::DecodeStream`.
struct DecodeStream<'a, B: TextBackend + ?Sized> {
    backend: &'a B,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl<B: TextBackend + ?Sized> IncrementalDecoder for DecodeStream<'_, B> {
    fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            if !new_prefix.ends_with('\u{FFFD}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(token_id);
        let string = self.backend.decode(&self.ids, self.skip_special_tokens)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{FFFD}') {
            let new_text = string[self.prefix.len()..].to_string();
            self.ids.drain(..self.prefix_index);
            self.prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            self.prefix_index = self.ids.len();
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    fn flush(&mut self) -> Result<Option<String>> {
        if self.ids.is_empty() {
            return Ok(None);
        }
        let text = self.backend.decode(&self.ids, self.skip_special_tokens)?;
        let remaining = &text[self.prefix.len()..];
        self.ids.clear();
        self.prefix.clear();
        if remaining.is_empty() {
            Ok(None)
        } else {
            Ok(Some(remaining.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend that treats each token ID as a raw byte, producing lossy UTF-8.
    #[derive(Debug)]
    struct Utf8Backend;

    impl TextBackend for Utf8Backend {
        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    #[test]
    fn holds_incomplete_utf8_until_complete() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false);

        // 你 = U+4F60 = 0xE4 0xBD 0xA0
        assert_eq!(decoder.step(0xe4).unwrap(), None);
        assert_eq!(decoder.step(0xbd).unwrap(), None);
        assert_eq!(decoder.step(0xa0).unwrap().as_deref(), Some("你"));
    }

    #[test]
    fn emits_ascii_immediately() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false);

        assert_eq!(decoder.step(b'o' as u32).unwrap().as_deref(), Some("o"));
        assert_eq!(decoder.step(b'k' as u32).unwrap().as_deref(), Some("k"));
    }

    #[test]
    fn flush_returns_none_when_fully_consumed() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false);

        assert_eq!(decoder.step(b'o' as u32).unwrap().as_deref(), Some("o"));
        assert_eq!(decoder.step(b'k' as u32).unwrap().as_deref(), Some("k"));
        assert_eq!(decoder.flush().unwrap(), None);
    }

    #[test]
    fn flush_emits_buffered_incomplete_utf8() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false);

        // Push incomplete multi-byte sequence — step returns None.
        assert_eq!(decoder.step(0xe4).unwrap(), None);
        assert_eq!(decoder.step(0xbd).unwrap(), None);

        // Flush forces out whatever the decoder can produce (lossy replacement).
        let flushed = decoder.flush().unwrap();
        assert!(flushed.is_some());
    }

    /// Backend where token 0 is a special token.
    #[derive(Debug)]
    struct SpecialTokenBackend;

    impl TextBackend for SpecialTokenBackend {
        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            let mut text = String::new();
            for &token_id in token_ids {
                match token_id {
                    0 if !skip_special_tokens => text.push_str("<special>"),
                    0 => {}
                    1 => text.push('a'),
                    _ => {}
                }
            }
            Ok(text)
        }
    }

    #[test]
    fn respects_skip_special_tokens() {
        let backend = SpecialTokenBackend;
        let mut skip_decoder = backend.create_decode_stream(&[], true);
        let mut keep_decoder = backend.create_decode_stream(&[], false);

        assert_eq!(skip_decoder.step(0).unwrap(), None);
        assert_eq!(keep_decoder.step(0).unwrap().as_deref(), Some("<special>"));
    }

    #[test]
    fn prompt_tokens_provide_context_without_re_emission() {
        let backend = Utf8Backend;
        let prompt = &[b'H' as u32, b'i' as u32];
        let mut decoder = backend.create_decode_stream(prompt, false);

        // First generated token should not re-emit "Hi".
        let chunk = decoder.step(b'!' as u32).unwrap();
        assert_eq!(chunk.as_deref(), Some("!"));
    }

    #[test]
    fn chunks_concatenate_to_full_text() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false);

        let input = b"Hello, world!";
        let mut full = String::new();
        for &byte in input {
            if let Some(chunk) = decoder.step(byte as u32).unwrap() {
                full.push_str(&chunk);
            }
        }
        if let Some(chunk) = decoder.flush().unwrap() {
            full.push_str(&chunk);
        }
        assert_eq!(full, "Hello, world!");
    }
}
