// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming resolution of `open_pattern` / `close_pattern` boundaries.
//!
//! Transformers relies on the Python `regex` module's `partial=True` search to
//! decide whether a match at the end of the buffer is final, could still appear,
//! or is impossible. The Rust `regex` crate has no partial-match API, so this
//! module answers that question by stepping a `regex-automata` DFA, and uses a
//! `meta::Regex` for the actual match span and captures.

use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::meta::Regex;
use regex_automata::nfa::thompson::{self, WhichCaptures};
use regex_automata::util::primitives::StateID;
use regex_automata::{Anchored, Input, MatchKind, PatternID};
use regex_syntax::hir::literal::{ExtractKind, Extractor};
use regex_syntax::hir::{Hir, HirKind};
use serde_json::{Map, Value};
use thiserror_ext::AsReport as _;

use super::{Result, unsupported};

/// Upper bound for the memory of one boundary DFA.
const DFA_SIZE_LIMIT: usize = 16 << 20;

/// Named capture groups of a boundary match, as string values: the scope a
/// transform template reads them from.
pub(super) type Captures = Map<String, Value>;

/// Outcome of resolving a boundary at a candidate position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Resolution {
    /// The boundary matches `len` bytes; more input cannot change the match.
    Matched { len: usize, captures: Captures },
    /// More input could still produce or change a match.
    Pending,
    /// No match starts here, whatever input follows.
    NoMatch,
}

/// A regex boundary that can be resolved against a growing buffer.
#[derive(Debug, Clone)]
pub(super) struct StreamingPattern {
    /// Finite set of non-empty literals, one of which starts every match.
    prefixes: Vec<String>,
    /// Anchored leftmost-first DFA of the pattern: decides whether the
    /// leftmost-first match is final. Unicode word boundaries are handled
    /// heuristically, quitting on non-ASCII bytes.
    first: dense::DFA<Vec<u32>>,
    /// Anchored DFA with every look-around assertion removed and
    /// `MatchKind::All`, used when `first` quits. Relaxing an assertion only
    /// enlarges the language, so once this DFA is dead no continuation can
    /// produce a later-ending match of the real pattern. `MatchKind::All` is
    /// required for that argument: leftmost-first pruning could discard the
    /// thread the real pattern takes when an assertion fails (`a\b|ab` on `ab`).
    fallback: Option<dense::DFA<Vec<u32>>>,
    /// The pattern itself, for the match span and named captures.
    exact: Regex,
}

/// Result of walking a DFA over the available bytes.
enum Walk {
    /// No continuation can change the result.
    Final,
    /// More input could still change the result.
    Alive,
    /// The DFA gave up (non-ASCII byte near a Unicode word boundary).
    Quit,
}

impl StreamingPattern {
    /// Compile a boundary pattern in Python `regex` syntax, with `re.DOTALL`.
    ///
    /// Streaming resolution only starts at occurrences of the pattern's literal
    /// prefixes, so every match must start with one of a finite set of non-empty
    /// literals.
    pub fn new(scope: &str, pattern: &str) -> Result<Self> {
        let hir = regex_syntax::ParserBuilder::new()
            .dot_matches_new_line(true)
            .build()
            .parse(pattern)
            .map_err(|error| {
                unsupported!(
                    "{scope}: regex {pattern:?} is not supported by the Rust regex engine: {}",
                    error.as_report()
                )
            })?;

        let prefixes = literal_prefixes(&hir).ok_or_else(|| {
            unsupported!(
                "{scope}: pattern {pattern:?} must start with a literal prefix for streaming \
                 matching (e.g. `<tool_call>` in `<tool_call>(?P<name>\\w+)`)"
            )
        })?;

        let first = build_dfa(scope, pattern, &hir, MatchKind::LeftmostFirst)?;
        let fallback = hir
            .properties()
            .look_set()
            .contains_word_unicode()
            .then(|| build_dfa(scope, pattern, &relax_assertions(&hir), MatchKind::All))
            .transpose()?;
        let exact = Regex::builder().build_from_hir(&hir).map_err(|error| {
            unsupported!(
                "{scope}: failed to compile regex {pattern:?}: {}",
                error.as_report()
            )
        })?;

        Ok(Self {
            prefixes,
            first,
            fallback,
            exact,
        })
    }

    /// Literals, one of which starts every match.
    pub fn prefixes(&self) -> &[String] {
        &self.prefixes
    }

    /// Named capture groups of the pattern.
    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        self.exact.group_info().pattern_names(PatternID::ZERO).flatten()
    }

    /// Resolve a match starting at the beginning of `text`.
    ///
    /// `eof` means no more input will arrive.
    pub fn resolve(&self, text: &str, eof: bool) -> Resolution {
        let walk = match walk(&self.first, text) {
            Walk::Quit => walk(
                self.fallback.as_ref().expect("quits imply a fallback"),
                text,
            ),
            walk => walk,
        };
        if matches!(walk, Walk::Alive) && !eof {
            return Resolution::Pending;
        }

        let input = Input::new(text).anchored(Anchored::Yes);
        let mut captures = self.exact.create_captures();
        self.exact.search_captures(&input, &mut captures);
        let Some(found) = captures.get_match() else {
            return Resolution::NoMatch;
        };
        let names = self.exact.group_info().pattern_names(PatternID::ZERO);
        let captures = names
            .enumerate()
            .filter_map(|(index, name)| {
                let span = captures.get_group(index)?;
                Some((
                    name?.to_string(),
                    Value::String(text[span.range()].to_string()),
                ))
            })
            .collect();
        Resolution::Matched {
            len: found.end(),
            captures,
        }
    }
}

/// Walk `dfa` from an anchored start over `text`.
///
/// The walk is final once the DFA is dead, or once it is in a match state whose
/// transitions are all dead: that removes the one byte of holdback that delayed
/// DFA match reporting would otherwise add. A quit transition counts as live.
fn walk(dfa: &dense::DFA<Vec<u32>>, text: &str) -> Walk {
    let input = Input::new(text).anchored(Anchored::Yes);
    let Ok(mut state) = dfa.start_state_forward(&input) else {
        return Walk::Quit;
    };
    for byte in text.bytes() {
        state = dfa.next_state(state, byte);
        if dfa.is_quit_state(state) {
            return Walk::Quit;
        }
        if dfa.is_dead_state(state) || is_terminal_match(dfa, state) {
            return Walk::Final;
        }
    }
    Walk::Alive
}

/// Return whether `state` reports a match and cannot continue.
fn is_terminal_match(dfa: &dense::DFA<Vec<u32>>, state: StateID) -> bool {
    dfa.is_match_state(state)
        && (0..=u8::MAX).all(|byte| dfa.is_dead_state(dfa.next_state(state, byte)))
}

/// Build an anchored dense DFA for `hir`.
fn build_dfa(
    scope: &str,
    pattern: &str,
    hir: &Hir,
    kind: MatchKind,
) -> Result<dense::DFA<Vec<u32>>> {
    let nfa = thompson::Compiler::new()
        .configure(thompson::Config::new().which_captures(WhichCaptures::None))
        .build_from_hir(hir)
        .map_err(|error| {
            unsupported!(
                "{scope}: failed to compile regex {pattern:?}: {}",
                error.as_report()
            )
        })?;
    dense::Builder::new()
        .configure(
            dense::Config::new()
                .match_kind(kind)
                .start_kind(StartKind::Anchored)
                .unicode_word_boundary(true)
                .dfa_size_limit(Some(DFA_SIZE_LIMIT))
                .determinize_size_limit(Some(DFA_SIZE_LIMIT)),
        )
        .build_from_nfa(&nfa)
        .map_err(|error| {
            unsupported!(
                "{scope}: regex {pattern:?} cannot be matched incrementally: {}",
                error.as_report()
            )
        })
}

/// Replace every look-around assertion with the empty regex.
fn relax_assertions(hir: &Hir) -> Hir {
    match hir.kind() {
        HirKind::Look(_) => Hir::empty(),
        HirKind::Repetition(repetition) => {
            let mut repetition = repetition.clone();
            repetition.sub = Box::new(relax_assertions(&repetition.sub));
            Hir::repetition(repetition)
        }
        HirKind::Capture(capture) => {
            let mut capture = capture.clone();
            capture.sub = Box::new(relax_assertions(&capture.sub));
            Hir::capture(capture)
        }
        HirKind::Concat(subs) => Hir::concat(subs.iter().map(relax_assertions).collect()),
        HirKind::Alternation(subs) => Hir::alternation(subs.iter().map(relax_assertions).collect()),
        HirKind::Empty | HirKind::Literal(_) | HirKind::Class(_) => hir.clone(),
    }
}

/// Extract the finite set of non-empty literal prefixes starting every match.
fn literal_prefixes(hir: &Hir) -> Option<Vec<String>> {
    let seq = Extractor::new().kind(ExtractKind::Prefix).extract(hir);
    let mut prefixes: Vec<String> = Vec::new();
    for literal in seq.literals()? {
        let bytes = literal.as_bytes();
        // Truncated inexact literals may end inside a UTF-8 sequence; any valid
        // prefix of a prefix is still a prefix.
        let valid = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => std::str::from_utf8(&bytes[..error.valid_up_to()]).ok()?,
        };
        if valid.is_empty() {
            return None;
        }
        if !prefixes.iter().any(|prefix| prefix == valid) {
            prefixes.push(valid.to_string());
        }
    }
    (!prefixes.is_empty()).then_some(prefixes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MUSE_INVOKE: &str = r#"<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">"#;
    const GEMMA4_CALL: &str = r"<\|tool_call>call:(?P<name>\w+)";
    const GPT_OSS_CALL: &str =
        r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>";

    fn pattern(source: &str) -> StreamingPattern {
        StreamingPattern::new("test", source).unwrap()
    }

    fn matched(len: usize, name: &str) -> Resolution {
        Resolution::Matched {
            len,
            captures: Captures::from_iter([("name".to_string(), name.into())]),
        }
    }

    #[test]
    fn extracts_literal_prefixes() {
        assert_eq!(pattern(GEMMA4_CALL).prefixes(), ["<|tool_call>call:"]);
        assert_eq!(pattern(MUSE_INVOKE).prefixes(), ["<atem:invoke"]);
        assert_eq!(
            pattern(r"to=(?:user|commentary)<\|message\|>").prefixes(),
            ["to=user<|message|>", "to=commentary<|message|>"]
        );
        assert_eq!(
            pattern(r"\n?</response>").prefixes(),
            ["\n</response>", "</response>"]
        );
        for source in [r"(?:^|<think>\s*)", r"(?:<\|m\|>)?[^<]*<\|c\|>", r".*x"] {
            let result = StreamingPattern::new("test", source);
            assert!(
                matches!(
                    result,
                    Err(super::super::HfTemplateError::Unsupported { .. })
                ),
                "{source}"
            );
        }
    }

    #[test]
    fn word_boundaries_follow_unicode_semantics() {
        let muse = pattern(MUSE_INVOKE);
        assert_eq!(
            muse.resolve(r#"<atem:invoke name="get">rest"#, false),
            matched(24, "get")
        );
        let unicode_attr = r#"<atem:invoke title="天气" name="获取">rest"#;
        assert_eq!(
            muse.resolve(unicode_attr, false),
            matched(unicode_attr.len() - 4, "获取")
        );
        assert_eq!(
            muse.resolve(r#"<atem:invoke name="get"#, false),
            Resolution::Pending
        );
        assert_eq!(
            muse.resolve(r#"<atem:invoker name="x">r"#, false),
            Resolution::NoMatch
        );
        assert_eq!(
            muse.resolve(r#"<atem:invoke xname="x">r"#, false),
            Resolution::NoMatch
        );
        assert_eq!(
            muse.resolve(r#"<atem:invoke é name="x">r"#, false),
            matched(25, "x")
        );
        assert_eq!(
            muse.resolve(r#"<atem:invoke éname="x">r"#, false),
            Resolution::NoMatch
        );
        assert_eq!(
            muse.resolve("<atem:invoke 使用标签", false),
            Resolution::Pending
        );
        assert_eq!(
            muse.resolve("<atem:invoke 使用标签", true),
            Resolution::NoMatch
        );
        assert_eq!(
            muse.resolve("<atem:invoke 使用> 正文", false),
            Resolution::NoMatch
        );

        let alternation = pattern(r"a\b|ab");
        assert_eq!(
            alternation.resolve("ab!", true),
            Resolution::Matched {
                len: 2,
                captures: Captures::new()
            }
        );
    }

    #[test]
    fn greedy_and_lazy_tails_commit_like_transformers() {
        let gemma4 = pattern(GEMMA4_CALL);
        assert_eq!(
            gemma4.resolve("<|tool_call>call:get_wea", false),
            Resolution::Pending
        );
        assert_eq!(
            gemma4.resolve("<|tool_call>call:get_weather{", false),
            matched(28, "get_weather")
        );
        assert_eq!(
            gemma4.resolve("<|tool_call>call:天气{", false),
            matched(23, "天气")
        );
        assert_eq!(
            gemma4.resolve("<|tool_call>call:{", false),
            Resolution::NoMatch
        );
        assert_eq!(
            gemma4.resolve("<|tool_call>call:get", true),
            matched(20, "get")
        );

        let gpt_oss = pattern(GPT_OSS_CALL);
        let opener = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>";
        assert_eq!(gpt_oss.resolve(opener, false), Resolution::Pending);
        assert_eq!(
            gpt_oss.resolve(&format!("{opener}{{\"a\""), false),
            matched(opener.len(), "get_weather")
        );
    }
}
