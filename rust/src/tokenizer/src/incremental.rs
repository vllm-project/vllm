use std::mem::take;

use crate::{Result, Tokenizer};

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return how many new string bytes were
    /// added.
    fn push_token(&mut self, token_id: u32) -> Result<usize>;

    /// Consume any text which is currently ready.
    fn next_chunk(&mut self) -> Option<String>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out buffered/incomplete
    /// fragments.
    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)>;

    /// Return cumulative decoded text so far.
    fn output(&self) -> &str;
}

/// [`IncrementalDecoder`] built on [`Tokenizer::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream`
pub(crate) struct DecodeStream<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    // mutated state
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    cumulative_output: String,
    output_index: usize,
}

impl<'a, T: Tokenizer + ?Sized> DecodeStream<'a, T> {
    pub(crate) fn new(
        tokenizer: &'a T,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            min_bytes_to_buffer,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            cumulative_output: String::new(),
            output_index: 0,
        }
    }
}

/// Try a short tail suffix first (covers a CJK glyph straddling 1-2 token
/// boundaries); beyond 6 tokens the fallback full-prompt decode is no worse
/// than baseline so widening the sweep just adds overhead.
const SAFE_SUFFIX_MIN: usize = 4;
const SAFE_SUFFIX_MAX: usize = 6;

impl<T: Tokenizer + ?Sized> DecodeStream<'_, T> {
    /// Seed `self.prefix` from the shortest trailing suffix whose decoded text
    /// has no U+FFFD — a clean decode means the suffix starts and ends at
    /// valid UTF-8/token boundaries, so priming from it is equivalent to
    /// priming from the full prompt.
    fn seed_prefix(&mut self) -> Result<()> {
        let prompt_len = self.ids.len();
        if prompt_len > SAFE_SUFFIX_MIN {
            let max_try = SAFE_SUFFIX_MAX.min(prompt_len - 1);
            for suffix_len in SAFE_SUFFIX_MIN..=max_try {
                let start = prompt_len - suffix_len;
                let decoded =
                    self.tokenizer.decode(&self.ids[start..], self.skip_special_tokens)?;
                if !decoded.contains('\u{FFFD}') {
                    self.prefix = decoded;
                    self.ids.drain(..start);
                    self.prefix_index = self.ids.len();
                    return Ok(());
                }
            }
        }
        let decoded = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if !decoded.ends_with('\u{FFFD}') {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }
}

impl<T: Tokenizer + ?Sized> IncrementalDecoder for DecodeStream<'_, T> {
    fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            self.seed_prefix()?;
        }

        self.ids.push(token_id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        if string.len() <= prefix_len || string.ends_with('\u{FFFD}') {
            return Ok(0);
        }
        // Ensure we split at a utf-8 char boundary.
        let new_chunk = &string[string.floor_char_boundary(prefix_len)..];
        self.cumulative_output.push_str(new_chunk);
        self.ids.drain(..self.prefix_index);
        self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        self.prefix_index = self.ids.len();
        Ok(new_chunk.len())
    }

    fn next_chunk(&mut self) -> Option<String> {
        let cutoff = self.cumulative_output.len().saturating_sub(self.min_bytes_to_buffer);
        (cutoff > self.output_index).then(|| {
            let chunk = self.cumulative_output[self.output_index..cutoff].to_string();
            self.output_index = cutoff;
            chunk
        })
    }

    fn flush(&mut self, truncate_output_to: Option<usize>) -> Result<(Option<String>, String)> {
        if !self.ids.is_empty() {
            let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            self.ids.clear();
            self.prefix.clear();
            self.prefix_index = 0;
            // Ensure we split at a utf-8 char boundary.
            self.cumulative_output
                .push_str(&string[string.floor_char_boundary(prefix_len)..]);
        }
        if let Some(truncate_output_to) = truncate_output_to {
            self.cumulative_output.truncate(truncate_output_to);
        }
        let last_chunk = (self.output_index < self.cumulative_output.len())
            .then(|| self.cumulative_output[self.output_index..].to_string());
        self.output_index = 0;
        Ok((last_chunk, take(&mut self.cumulative_output)))
    }

    fn output(&self) -> &str {
        &self.cumulative_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend that treats each token ID as a raw byte, producing lossy UTF-8.
    #[derive(Debug)]
    struct Utf8Backend;

    impl Tokenizer for Utf8Backend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }
    }

    #[test]
    fn holds_incomplete_utf8_until_complete() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // 你 = U+4F60 = 0xE4 0xBD 0xA0
        assert_eq!(decoder.push_token(0xe4).unwrap(), 0);
        assert_eq!(decoder.push_token(0xbd).unwrap(), 0);
        assert_eq!(decoder.push_token(0xa0).unwrap(), 3); // "你" is 3 bytes
        assert_eq!(decoder.output(), "你");
    }

    #[test]
    fn emits_ascii_immediately() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap(), 1);
        assert_eq!(decoder.push_token(b'k' as u32).unwrap(), 1);
        assert_eq!(decoder.output(), "ok");
    }

    #[test]
    fn flush_returns_none_when_fully_consumed() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap(), 1);
        assert_eq!(decoder.next_chunk().as_deref(), Some("o"));
        assert_eq!(decoder.push_token(b'k' as u32).unwrap(), 1);
        assert_eq!(decoder.next_chunk().as_deref(), Some("k"));
        // All text already consumed via next_chunk
        let (last_chunk, full_text) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None);
        assert_eq!(full_text, "ok");
    }

    #[test]
    fn flush_emits_buffered_incomplete_utf8() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // Push incomplete multi-byte sequence — step returns 0 bytes.
        assert_eq!(decoder.push_token(0xe4).unwrap(), 0);
        assert_eq!(decoder.push_token(0xbd).unwrap(), 0);

        // Flush forces out whatever the decoder can produce (lossy replacement).
        let (last_chunk, _full_text) = decoder.flush(None).unwrap();
        assert!(last_chunk.is_some());
    }

    /// Backend where token 0 is a special token.
    #[derive(Debug)]
    struct SpecialTokenBackend;

    impl Tokenizer for SpecialTokenBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
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

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }
    }

    #[test]
    fn respects_skip_special_tokens() {
        let backend = SpecialTokenBackend;
        let mut skip_decoder = backend.create_decode_stream(&[], true, 0);
        let mut keep_decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(skip_decoder.push_token(0).unwrap(), 0);
        assert_eq!(keep_decoder.push_token(0).unwrap(), 9); // "<special>" is 9 bytes
        assert_eq!(keep_decoder.output(), "<special>");
    }

    #[test]
    fn prompt_tokens_provide_context_without_re_emission() {
        let backend = Utf8Backend;
        let prompt = &[b'H' as u32, b'i' as u32];
        let mut decoder = backend.create_decode_stream(prompt, false, 0);

        // First generated token should not re-emit "Hi".
        let added = decoder.push_token(b'!' as u32).unwrap();
        assert_eq!(added, 1);
        assert_eq!(decoder.output(), "!");
    }

    #[test]
    fn chunks_concatenate_to_full_text() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        let input = b"Hello, world!";
        let mut full = String::new();
        for &byte in input {
            decoder.push_token(byte as u32).unwrap();
            if let Some(chunk) = decoder.next_chunk() {
                full.push_str(&chunk);
            }
        }
        let (last_chunk, full_text) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None); // all consumed via next_chunk
        assert_eq!(full, "Hello, world!");
        assert_eq!(full_text, "Hello, world!");
    }

    /// Backend simulating non-monotonic decode where adding a token changes how
    /// earlier tokens decode (context-dependent normalization), causing
    /// prefix_len to land mid-UTF-8. Reproduces the class of bug from
    /// vllm-project/vllm#17448.
    #[derive(Debug)]
    struct NonMonotonicBackend;

    impl Tokenizer for NonMonotonicBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            match token_ids {
                [1] => Ok("abc".into()),
                [1, 2] => Ok("ab".into()),
                // Token 3 triggers a normalization change: "ab" becomes emoji + "d".
                // prefix_len=3 ("abc") lands inside the 4-byte emoji 🎉.
                [1, 2, 3] => Ok("🎉d".into()), // 🎉 is 4 bytes + d = 5 bytes
                [2, 3] => Ok("🎉d".into()),    // prefix recompute after drain
                [3] => Ok("d".into()),         // after drain
                _ => panic!("unexpected decode: {:?}", token_ids),
            }
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }
    }

    /// Without the char-boundary fix, this panics slicing mid-emoji.
    #[test]
    fn non_monotonic_decode_does_not_panic() {
        let backend = NonMonotonicBackend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // Token 1: "abc", prefix="abc"
        assert_eq!(decoder.push_token(1).unwrap(), 3);
        // Token 2: "ab" (shorter), no emit
        assert_eq!(decoder.push_token(2).unwrap(), 0);
        // Token 3: "🎉d" — prefix_len=3 is mid-emoji. Without fix this panics.
        let added = decoder.push_token(3).unwrap();
        assert!(added > 0);
    }

    #[test]
    fn next_chunk_with_hold_back() {
        let backend = Utf8Backend;
        // hold_back_bytes: 3 means we buffer the last 3 bytes
        let mut decoder = backend.create_decode_stream(&[], false, 3);

        let input = b"Hello!";
        let mut chunks = String::new();
        for &byte in input {
            decoder.push_token(byte as u32).unwrap();
            if let Some(chunk) = decoder.next_chunk() {
                chunks.push_str(&chunk);
            }
        }
        // With hold_back_bytes=3, last 3 bytes ("lo!") are held back
        assert_eq!(chunks, "Hel");
        // Flush returns the rest
        let (last_chunk, full_text) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk.as_deref(), Some("lo!"));
        assert_eq!(full_text, "Hello!");
    }
}
