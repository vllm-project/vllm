use crate::backend::DynChatBackend;
use crate::error::Result;

/// Small left-context window retained when initializing the decoder from prompt tokens.
const INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 5;

/// Minimal chat-local incremental decoder built on top of `ChatBackend::decode()`.
pub(crate) struct IncrementalTextDecoder {
    /// Backend used for full-slice detokenization.
    backend: DynChatBackend,
    /// Whether special tokens should be suppressed while decoding.
    skip_special_tokens: bool,

    /// Prompt tokens followed by every generated token observed so far.
    all_token_ids: Vec<u32>,
    /// Start of the left-context window used for incremental diffing.
    prefix_offset: usize,
    /// Exclusive end of the last successfully emitted decode window.
    read_offset: usize,
}

impl IncrementalTextDecoder {
    /// Create one incremental decoder primed with the request prompt tokens.
    ///
    /// The decoder retains only a small suffix of the prompt as left context for the first
    /// generated token; it does not attempt to re-emit prompt text.
    pub(crate) fn new(
        backend: DynChatBackend,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Self {
        Self {
            backend,
            skip_special_tokens,
            all_token_ids: prompt_token_ids.to_vec(),
            prefix_offset: prompt_token_ids
                .len()
                .saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET),
            read_offset: prompt_token_ids.len(),
        }
    }

    /// Push one newly generated token and, if possible, return the newly decoded text chunk.
    ///
    /// Returning `Ok(None)` means the token did not yet produce a stable text fragment. This
    /// commonly happens when the tokenizer is still in the middle of decoding a multi-byte UTF-8
    /// sequence.
    pub(crate) fn push_token(&mut self, token_id: u32) -> Result<Option<String>> {
        self.all_token_ids.push(token_id);

        // Decode the previously stable window first, then decode the same window extended with
        // the new token. The difference between them is the newly available text.
        let prefix_text = self.backend.decode(
            &self.all_token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;
        let new_text = self.backend.decode(
            &self.all_token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        if new_text.len() > prefix_text.len() && !new_text.ends_with('\u{fffd}') {
            let mut split_at = prefix_text.len();
            // `prefix_text.len()` may land in the middle of a UTF-8 codepoint, so walk back to a
            // valid boundary before slicing.
            while split_at > 0 && !new_text.is_char_boundary(split_at) {
                split_at -= 1;
            }

            // Once a stable chunk is emitted, the newly extended window becomes the next baseline.
            self.prefix_offset = self.read_offset;
            self.read_offset = self.all_token_ids.len();

            Ok(Some(new_text[split_at..].to_string()))
        } else {
            Ok(None)
        }
    }

    /// Flush any remaining decodable text that has been buffered but not yet emitted.
    ///
    /// This is mainly relevant at terminal completion, where the caller wants to force out the
    /// last chunk after all generated tokens have been observed.
    pub(crate) fn flush(&mut self) -> Result<Option<String>> {
        if self.read_offset >= self.all_token_ids.len() {
            return Ok(None);
        }

        let remaining = self.backend.decode(
            &self.all_token_ids[self.read_offset..],
            self.skip_special_tokens,
        )?;
        self.read_offset = self.all_token_ids.len();

        if remaining.is_empty() {
            Ok(None)
        } else {
            Ok(Some(remaining))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::IncrementalTextDecoder;
    use crate::backend::ChatBackend;
    use crate::request::ChatRequest;

    #[derive(Debug)]
    struct Utf8Backend;

    impl ChatBackend for Utf8Backend {
        fn apply_chat_template(&self, _request: &ChatRequest) -> crate::Result<String> {
            unreachable!()
        }

        fn encode(&self, _text: &str) -> crate::Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> crate::Result<String> {
            let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    #[test]
    fn incremental_decoder_holds_utf8_until_complete() {
        let backend: Arc<dyn ChatBackend> = Arc::new(Utf8Backend);
        let mut decoder = IncrementalTextDecoder::new(backend, &[], false);

        assert_eq!(decoder.push_token(0xe4).unwrap(), None);
        assert_eq!(decoder.push_token(0xbd).unwrap(), None);
        assert_eq!(decoder.push_token(0xa0).unwrap().as_deref(), Some("你"));
    }

    #[test]
    fn incremental_decoder_flushes_remaining_text() {
        let backend: Arc<dyn ChatBackend> = Arc::new(Utf8Backend);
        let mut decoder = IncrementalTextDecoder::new(backend, &[], false);

        assert_eq!(
            decoder.push_token(b'o' as u32).unwrap().as_deref(),
            Some("o")
        );
        assert_eq!(
            decoder.push_token(b'k' as u32).unwrap().as_deref(),
            Some("k")
        );
        assert_eq!(decoder.flush().unwrap(), None);
    }

    #[derive(Debug)]
    struct SpecialTokenBackend;

    impl ChatBackend for SpecialTokenBackend {
        fn apply_chat_template(&self, _request: &ChatRequest) -> crate::Result<String> {
            unreachable!()
        }

        fn encode(&self, _text: &str) -> crate::Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> crate::Result<String> {
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
    fn incremental_decoder_respects_skip_special_tokens() {
        let backend: Arc<dyn ChatBackend> = Arc::new(SpecialTokenBackend);
        let mut skip_decoder = IncrementalTextDecoder::new(backend.clone(), &[], true);
        let mut keep_decoder = IncrementalTextDecoder::new(backend, &[], false);

        assert_eq!(skip_decoder.push_token(0).unwrap(), None);
        assert_eq!(
            keep_decoder.push_token(0).unwrap().as_deref(),
            Some("<special>")
        );
    }
}
