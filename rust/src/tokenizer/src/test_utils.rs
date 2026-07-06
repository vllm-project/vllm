use std::collections::BTreeMap;

use crate::{Result, Tokenizer, TokenizerError};

const FIRST_CONFIGURED_TOKEN_ID: u32 = 256;

/// Whether a configured test token should be treated as special.
///
/// Special tokens are skipped by [`Tokenizer::decode`] when
/// `skip_special_tokens` is set. Regular configured tokens are always emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestTokenKind {
    /// Token is skipped when `skip_special_tokens = true`.
    Special,
    /// Token is emitted regardless of `skip_special_tokens`.
    Regular,
}

impl TestTokenKind {
    fn is_special(self) -> bool {
        matches!(self, Self::Special)
    }
}

/// Decode behavior for token ids that are neither configured tokens nor byte ids.
///
/// The default is [`UnknownDecode::Error`] so tests notice missing tokenizer
/// fixtures instead of silently accepting impossible ids. Individual tests can
/// opt into empty or replacement output when they are explicitly modeling a
/// lenient detokenization path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownDecode {
    /// Return a tokenizer error on the first unknown id.
    Error,
    /// Drop unknown ids from decoded output.
    Empty,
    /// Emit U+FFFD for each unknown id.
    Replacement,
}

#[derive(Debug, Clone)]
struct TestToken {
    text: String,
    kind: TestTokenKind,
}

/// Configurable tokenizer for Rust frontend tests.
///
/// `TestTokenizer` is intentionally small, but its methods obey the same basic
/// contract as production tokenizers:
///
/// - ordinary text encodes as UTF-8 byte ids;
/// - configured token ids start at 256, leaving `0..=255` for byte fallback;
/// - configured token ids and token text are unique;
/// - configured tokens are matched before ordinary bytes, using longest-prefix matching so
///   multi-character markers such as `<think>` work naturally;
/// - `token_to_id` and `id_to_token` are consistent for configured tokens;
/// - `decode` is strict by default for ids outside the byte range and the configured token table;
/// - `vocab_size` is an exclusive upper bound covering byte ids and configured token ids unless a
///   test sets it explicitly.
///
/// Prefer this helper over ad-hoc fake tokenizers for tests that rely on
/// tokenizer semantics. Keep dedicated tiny fakes for error injection or for
/// tests that deliberately need a degenerate tokenizer.
#[derive(Debug, Clone)]
pub struct TestTokenizer {
    token_to_id: BTreeMap<String, u32>,
    id_to_token: BTreeMap<u32, TestToken>,
    unknown_decode: UnknownDecode,
    vocab_size: Option<usize>,
    bos_token_id: Option<u32>,
}

impl Default for TestTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TestTokenizer {
    /// Create a byte-level test tokenizer with strict unknown-id decode.
    pub fn new() -> Self {
        Self {
            token_to_id: BTreeMap::new(),
            id_to_token: BTreeMap::new(),
            unknown_decode: UnknownDecode::Error,
            vocab_size: None,
            bos_token_id: None,
        }
    }

    /// Add a configured token and return the updated tokenizer.
    ///
    /// Configured tokens must use ids outside the byte range and may be marked
    /// special or regular.
    pub fn with_token(mut self, token: impl Into<String>, id: u32, kind: TestTokenKind) -> Self {
        self.insert_token(token, id, kind);
        self
    }

    /// Add a special configured token and return the updated tokenizer.
    pub fn with_special_token(self, token: impl Into<String>, id: u32) -> Self {
        self.with_token(token, id, TestTokenKind::Special)
    }

    /// Add a regular configured token and return the updated tokenizer.
    pub fn with_regular_token(self, token: impl Into<String>, id: u32) -> Self {
        self.with_token(token, id, TestTokenKind::Regular)
    }

    /// Add a special BOS token inserted by `encode(..., true)`.
    ///
    /// This also registers the token in the normal token/id maps so
    /// `token_to_id`, `id_to_token`, `decode`, and `is_special_id` stay
    /// consistent for the inserted id.
    pub fn with_bos_token(mut self, token: impl Into<String>, id: u32) -> Self {
        self.insert_token(token, id, TestTokenKind::Special);
        self.bos_token_id = Some(id);
        self
    }

    /// Set decode behavior for unknown non-byte ids.
    pub fn with_unknown_decode(mut self, behavior: UnknownDecode) -> Self {
        self.unknown_decode = behavior;
        self
    }

    /// Set an explicit vocabulary size.
    ///
    /// Use this when a test needs a model-like vocabulary bound that differs
    /// from the highest configured token id plus one.
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    fn insert_token(&mut self, token: impl Into<String>, id: u32, kind: TestTokenKind) {
        let token = token.into();
        assert!(
            !token.is_empty(),
            "configured test token text must be non-empty"
        );
        assert!(
            id >= FIRST_CONFIGURED_TOKEN_ID,
            "configured test token id {id} overlaps byte fallback range 0..=255"
        );
        assert!(
            token.len() > 1,
            "configured test token text {token:?} overlaps byte fallback token text"
        );
        if self.token_to_id.insert(token.clone(), id).is_some() {
            panic!("configured test token text {token:?} was registered more than once");
        }
        if self.id_to_token.insert(id, TestToken { text: token, kind }).is_some() {
            panic!("configured test token id {id} was registered more than once");
        }
    }

    fn byte_to_token(id: u32) -> Option<String> {
        u8::try_from(id).ok().map(|byte| String::from_utf8_lossy(&[byte]).into_owned())
    }

    fn flush_bytes(bytes: &mut Vec<u8>, output: &mut String) {
        if !bytes.is_empty() {
            output.push_str(&String::from_utf8_lossy(bytes));
            bytes.clear();
        }
    }

    fn configured_token_prefix(&self, text: &str) -> Option<(&str, u32)> {
        self.token_to_id
            .iter()
            .filter_map(|(token, &id)| text.starts_with(token).then_some((token.as_str(), id)))
            .max_by_key(|(token, _)| token.len())
    }

    fn inferred_vocab_size(&self) -> usize {
        let max_configured =
            self.id_to_token.last_key_value().map(|(&id, _)| id as usize + 1).unwrap_or(0);
        256.max(max_configured)
    }
}

impl Tokenizer for TestTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_special_tokens && let Some(bos_token_id) = self.bos_token_id {
            ids.push(bos_token_id);
        }

        let mut rest = text;
        while !rest.is_empty() {
            if let Some((token, id)) = self.configured_token_prefix(rest) {
                ids.push(id);
                rest = &rest[token.len()..];
                continue;
            }

            let ch = rest.chars().next().expect("rest is not empty");
            let mut buf = [0_u8; 4];
            ids.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
            rest = &rest[ch.len_utf8()..];
        }

        Ok(ids)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut output = String::new();
        let mut pending_bytes = Vec::new();
        for &id in token_ids {
            if let Some(token) = self.id_to_token.get(&id) {
                Self::flush_bytes(&mut pending_bytes, &mut output);
                if !(skip_special_tokens && token.kind.is_special()) {
                    output.push_str(&token.text);
                }
            } else if let Ok(byte) = u8::try_from(id) {
                pending_bytes.push(byte);
            } else {
                Self::flush_bytes(&mut pending_bytes, &mut output);
                match self.unknown_decode {
                    UnknownDecode::Error => {
                        return Err(TokenizerError(format!(
                            "test tokenizer cannot decode unknown token id {id}"
                        )));
                    }
                    UnknownDecode::Empty => {}
                    UnknownDecode::Replacement => output.push('\u{FFFD}'),
                }
            }
        }
        Self::flush_bytes(&mut pending_bytes, &mut output);
        Ok(output)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied().or_else(|| {
            let bytes = token.as_bytes();
            (bytes.len() == 1).then(|| u32::from(bytes[0]))
        })
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token
            .get(&id)
            .map(|token| token.text.clone())
            .or_else(|| Self::byte_to_token(id))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size.unwrap_or_else(|| self.inferred_vocab_size())
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.id_to_token.get(&token_id).is_some_and(|token| token.kind.is_special())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_text_roundtrips_and_reports_byte_ids() {
        let tokenizer = TestTokenizer::new();

        let ids = tokenizer.encode("hi", false).unwrap();
        assert_eq!(ids, vec![b'h' as u32, b'i' as u32]);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), "hi");
        assert_eq!(tokenizer.token_to_id("h"), Some(b'h' as u32));
        assert_eq!(tokenizer.id_to_token(b'h' as u32).as_deref(), Some("h"));
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn configured_tokens_use_longest_prefix_matching() {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<image>", 999)
            .with_regular_token("<image></image>", 1000);

        assert_eq!(
            tokenizer.encode("a<image></image>b", false).unwrap(),
            vec![b'a' as u32, 1000, b'b' as u32,]
        );
        assert_eq!(
            tokenizer.decode(&[b'a' as u32, 1000, b'b' as u32], false).unwrap(),
            "a<image></image>b"
        );
        assert_eq!(tokenizer.token_to_id("<image>"), Some(999));
        assert_eq!(
            tokenizer.id_to_token(1000).as_deref(),
            Some("<image></image>")
        );
        assert_eq!(tokenizer.vocab_size(), 1001);
    }

    #[test]
    fn non_ascii_text_roundtrips_through_buffered_byte_decode() {
        let tokenizer = TestTokenizer::new();
        let text = "你好, café, 🚀";

        let ids = tokenizer.encode(text, false).unwrap();
        assert_eq!(
            ids,
            text.as_bytes().iter().copied().map(u32::from).collect::<Vec<_>>()
        );
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }

    #[test]
    fn buffered_byte_decode_flushes_around_configured_tokens() {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<image>", 999)
            .with_special_token("<skip>", 1000);

        assert_eq!(
            tokenizer.encode("你<image>好<skip>🚀", false).unwrap(),
            vec![228, 189, 160, 999, 229, 165, 189, 1000, 240, 159, 154, 128]
        );
        assert_eq!(
            tokenizer
                .decode(
                    &[228, 189, 160, 999, 229, 165, 189, 1000, 240, 159, 154, 128],
                    false
                )
                .unwrap(),
            "你<image>好<skip>🚀"
        );
        assert_eq!(
            tokenizer
                .decode(
                    &[228, 189, 160, 999, 229, 165, 189, 1000, 240, 159, 154, 128],
                    true
                )
                .unwrap(),
            "你<image>好🚀"
        );
    }

    #[test]
    fn invalid_utf8_bytes_decode_lossily_as_a_sequence() {
        let tokenizer = TestTokenizer::new();

        assert_eq!(tokenizer.decode(&[0xE4, 0xBD], false).unwrap(), "\u{FFFD}");
        assert_eq!(
            tokenizer.decode(&[0xFF, b'a' as u32], false).unwrap(),
            "\u{FFFD}a"
        );
    }

    #[test]
    fn special_tokens_respect_skip_special_tokens() {
        let tokenizer = TestTokenizer::new()
            .with_bos_token("<bos>", 256)
            .with_special_token("<think>", 0xF001)
            .with_regular_token("</think>", 0xF002);

        assert_eq!(
            tokenizer.encode("<think>x</think>", true).unwrap(),
            vec![256, 0xF001, b'x' as u32, 0xF002,]
        );
        assert_eq!(
            tokenizer.decode(&[256, 0xF001, b'x' as u32, 0xF002], false).unwrap(),
            "<bos><think>x</think>"
        );
        assert_eq!(
            tokenizer.decode(&[256, 0xF001, b'x' as u32, 0xF002], true).unwrap(),
            "x</think>"
        );
        assert!(tokenizer.is_special_id(0xF001));
        assert!(!tokenizer.is_special_id(0xF002));
    }

    #[test]
    #[should_panic(expected = "configured test token id 255 overlaps byte fallback range 0..=255")]
    fn configured_token_id_must_stay_outside_byte_range() {
        let _ = TestTokenizer::new().with_regular_token("<token>", 255);
    }

    #[test]
    #[should_panic(expected = "configured test token text \"a\" overlaps byte fallback token text")]
    fn configured_token_text_must_not_shadow_byte_tokens() {
        let _ = TestTokenizer::new().with_regular_token("a", 256);
    }

    #[test]
    #[should_panic(
        expected = "configured test token text \"<token>\" was registered more than once"
    )]
    fn configured_token_text_must_be_unique() {
        let _ = TestTokenizer::new()
            .with_regular_token("<token>", 256)
            .with_regular_token("<token>", 257);
    }

    #[test]
    #[should_panic(expected = "configured test token id 256 was registered more than once")]
    fn configured_token_id_must_be_unique() {
        let _ = TestTokenizer::new()
            .with_regular_token("<token-a>", 256)
            .with_regular_token("<token-b>", 256);
    }

    #[test]
    fn unknown_decode_is_strict_by_default_and_configurable() {
        let strict = TestTokenizer::new();
        assert!(strict.decode(&[300], false).is_err());
        assert_eq!(
            TestTokenizer::new()
                .with_unknown_decode(UnknownDecode::Empty)
                .decode(&[b'a' as u32, 300, b'b' as u32], false)
                .unwrap(),
            "ab"
        );
        assert_eq!(
            TestTokenizer::new()
                .with_unknown_decode(UnknownDecode::Replacement)
                .decode(&[300], false)
                .unwrap(),
            "\u{FFFD}"
        );
        assert_eq!(strict.id_to_token(300), None);
    }

    #[test]
    fn explicit_vocab_size_overrides_inferred_bound() {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<high>", 10_000)
            .with_vocab_size(20_000);

        assert_eq!(tokenizer.vocab_size(), 20_000);
        assert_eq!(tokenizer.id_to_token(10_000).as_deref(), Some("<high>"));
    }
}
