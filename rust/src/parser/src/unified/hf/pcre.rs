// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Template regexes on PCRE2, a backtracking engine like Python's `regex`.
//!
//! Transformers compiles template patterns with `re.DOTALL` and decides
//! streaming boundaries with `regex`'s partial matching, which reports a
//! complete match anywhere before a partial one (`PCRE2_PARTIAL_SOFT`). The
//! dialects mostly agree; [`translate`] rewrites the escapes that differ.

use std::ops::Range;

use pcre2::bytes::{MatchOptions, Partial, RegexBuilder};

use super::{Result, unsupported};

/// A compiled template regex.
#[derive(Debug)]
pub(super) struct Regex {
    regex: pcre2::bytes::Regex,
    /// Named groups with their indices, in group order.
    names: Vec<(usize, String)>,
}

/// A match, or with partial matching, a match that more input may complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Found {
    pub start: usize,
    pub end: usize,
    /// The match reached the end of the subject and needs more input.
    pub partial: bool,
    /// The span of each group, by index; group 0 is the match. Only group 0
    /// is set for a partial match, or for a pattern without named groups.
    pub groups: Vec<Option<Range<usize>>>,
}

/// A matching failure other than no match, such as an exceeded match limit.
#[derive(Debug, thiserror::Error)]
#[error("regex match failed: {0}")]
pub(super) struct MatchError(#[from] pcre2::Error);

impl Regex {
    /// Compile a template regex in Python's dialect with `re.DOTALL`.
    pub fn new(scope: &str, pattern: &str) -> Result<Self> {
        let regex = RegexBuilder::new()
            .ucp(true)
            .dotall(true)
            .jit_if_available(true)
            .jit_partial(true)
            .build(&translate(pattern))
            .map_err(|error| {
                unsupported!("{scope}: regex {pattern:?} is not supported: {error}")
            })?;
        let names = regex
            .capture_names()
            .iter()
            .enumerate()
            .filter_map(|(index, name)| Some((index, name.clone()?)))
            .collect();
        Ok(Self { regex, names })
    }

    /// Characters before the start position that a match may inspect
    /// (lookbehinds, `\b`).
    pub fn max_lookbehind(&self) -> usize {
        self.regex.max_lookbehind()
    }

    /// Named groups with their indices.
    pub fn group_names(&self) -> impl Iterator<Item = (usize, &str)> {
        self.names.iter().map(|(index, name)| (*index, name.as_str()))
    }

    /// Search `subject` from byte offset `start`, as Python's
    /// `pattern.search(subject, start, partial=partial)` does. Text before
    /// `start` is context for lookbehinds; `^` and `\A` match only at offset 0.
    /// `allow_empty_at_start: false` rejects an empty match at `start`.
    pub fn search(
        &self,
        subject: &str,
        start: usize,
        partial: bool,
        allow_empty_at_start: bool,
    ) -> std::result::Result<Option<Found>, MatchError> {
        let partial = if partial {
            Partial::Soft
        } else {
            Partial::Never
        };
        let options =
            MatchOptions::new().partial(partial).not_empty_at_start(!allow_empty_at_start);
        let subject = subject.as_bytes();
        let Some(found) = self.regex.find_at_with_options(subject, start, options)? else {
            return Ok(None);
        };
        let mut groups = vec![Some(found.start()..found.end())];
        if !found.is_partial() && !self.names.is_empty() {
            // Searching again from its start finds the same complete match.
            let options = MatchOptions::new()
                .not_empty_at_start(!allow_empty_at_start && found.start() == start);
            let mut locations = self.regex.capture_locations();
            self.regex.captures_read_at_with_options(
                &mut locations,
                subject,
                found.start(),
                options,
            )?;
            groups = (0..locations.len())
                .map(|index| locations.get(index).map(|(start, end)| start..end))
                .collect();
        }
        Ok(Some(Found {
            start: found.start(),
            end: found.end(),
            partial: found.is_partial(),
            groups,
        }))
    }

    /// Successive matches in `text`, as Python's `finditer` finds them.
    pub fn find_iter(&self, text: &str) -> std::result::Result<Vec<Found>, MatchError> {
        self.regex
            .captures_iter_perl(text.as_bytes())
            .map(|captures| {
                let captures = captures?;
                let groups: Vec<_> = (0..captures.len())
                    .map(|index| captures.get(index).map(|group| group.start()..group.end()))
                    .collect();
                let whole = groups[0].clone().expect("the match is set");
                Ok(Found {
                    start: whole.start,
                    end: whole.end,
                    partial: false,
                    groups,
                })
            })
            .collect()
    }
}

/// Rewrite the escapes whose meaning differs between Python and PCRE2: `\Z`
/// (end of string, `\z` in PCRE2) and `\uXXXX` / `\UXXXXXXXX` (`\x{...}`).
fn translate(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.char_indices();
    while let Some((at, c)) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some((_, escaped)) = chars.next() else {
            out.push(c);
            break;
        };
        let digits = match escaped {
            'u' => 4,
            'U' => 8,
            'Z' => {
                out.push_str("\\z");
                continue;
            }
            _ => 0,
        };
        let hex = pattern.get(at + 2..at + 2 + digits);
        match hex.filter(|hex| digits > 0 && hex.chars().all(|c| c.is_ascii_hexdigit())) {
            Some(hex) => {
                out.push_str(&format!("\\x{{{hex}}}"));
                for _ in 0..digits {
                    chars.next();
                }
            }
            None => {
                out.push(c);
                out.push(escaped);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regex(pattern: &str) -> Regex {
        Regex::new("test", pattern).unwrap()
    }

    /// `(start, end, partial)` of `regex.search(text, partial=True)`.
    fn partial(pattern: &str, text: &str) -> Option<(usize, usize, bool)> {
        let found = regex(pattern).search(text, 0, true, true).unwrap()?;
        Some((found.start, found.end, found.partial))
    }

    #[test]
    fn partial_matching_follows_python_regex() {
        // Expected values are `regex.search(pattern, text, partial=True)`.
        let cases = [
            (r"a.*z|a", "abc", Some((0, 1, false))),
            (r"ab|a", "a", Some((0, 1, false))),
            (r"x|abc", "ab", Some((0, 2, true))),
            // A complete match anywhere wins over an earlier partial one.
            (r"abc|b", "ab", Some((1, 2, false))),
            (r"abc|c", "xab", Some((1, 3, true))),
            (r"a(?=bc)", "ab", Some((0, 2, true))),
            (r"<x>|(?=<y>)", "a<x", Some((1, 3, true))),
            (r"(?<=b)c", "abc", Some((2, 3, false))),
            (r"a+", "aaa", Some((0, 3, false))),
            (r"a+?", "aaa", Some((0, 1, false))),
            (r"\s*<t>", "  ", Some((0, 2, true))),
        ];
        for (pattern, text, expected) in cases {
            assert_eq!(partial(pattern, text), expected, "{pattern:?} on {text:?}");
        }
    }

    #[test]
    fn python_escapes_are_translated() {
        assert_eq!(
            translate(r"a\Z|\u00e9\U0001F600|\\u|\uZZ"),
            r"a\z|\x{00e9}\x{0001F600}|\\u|\uZZ"
        );
        assert!(regex(r"a\Z").search("a\n", 0, false, true).unwrap().is_none());
        assert!(regex(r"\u00e9").search("é", 0, false, true).unwrap().is_some());
    }

    #[test]
    fn find_iter_follows_python_finditer() {
        // `[m.span() for m in re.finditer(r"x*", "axb")]`
        let spans: Vec<_> =
            regex(r"x*").find_iter("axb").unwrap().iter().map(|m| m.start..m.end).collect();
        assert_eq!(spans, [0..0, 1..2, 2..2, 3..3]);
        let tags = regex(r"<p>(?P<v>.*?)(?:</p>|(?=<p>))").find_iter("<p>a<p>b</p>").unwrap();
        assert_eq!(
            tags.iter().map(|m| m.start..m.end).collect::<Vec<_>>(),
            [0..4, 4..12]
        );
        // A non-empty match where an empty one was.
        let spans: Vec<_> = regex(r"(?=<p>)|<p>a")
            .find_iter("<p>a<p>")
            .unwrap()
            .iter()
            .map(|m| m.start..m.end)
            .collect();
        assert_eq!(spans, [0..0, 0..4, 4..4]);
    }

    #[test]
    fn names_and_lookbehind() {
        let tag = regex(r"(?<=\n)<p\b (?P<key>\w+)=(?P<value>\w+)>");
        assert_eq!(
            tag.group_names().collect::<Vec<_>>(),
            [(1, "key"), (2, "value")]
        );
        assert_eq!(tag.max_lookbehind(), 1);
        // Text before the start position is context only.
        let found = tag.search("x\n<p a=b> ", 2, false, true).unwrap().unwrap();
        assert_eq!((found.start, found.groups[2].clone()), (2, Some(7..8)));
        assert!(tag.search("x\n<p a=b> ", 3, false, true).unwrap().is_none());
    }
}
