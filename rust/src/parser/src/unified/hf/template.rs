// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Compiled `response_template`.
//!
//! Validation and anchor compilation follow `response_templates.py`; the
//! compiled form additionally maps fields to parser roles and precomputes each
//! state's watched boundaries.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value};

use super::content::{ContentParser, TextArgs};
use super::pcre::{MatchError, Regex};
use super::spec::{AnchorSpec, FieldSpec, TemplateSpec};
use super::transform::{FieldTransform, Transform};
use super::{Result, invalid, unsupported};

/// Named capture groups of a boundary match, as string values: the scope a
/// transform template reads them from.
pub(super) type Captures = Map<String, Value>;

/// A compiled `response_template`, shared by all requests of one model.
#[derive(Debug)]
pub struct ResponseTemplate {
    /// Matches the start of the current assistant turn in the prompt.
    start_anchor: Regex,
    pub(super) regions: Vec<Region>,
    /// The field without an opener: the implicit-open / leftover sink.
    pub(super) implicit: Option<usize>,
    /// Boundaries watched outside explicit regions: every explicit open plus the
    /// implicit region's close.
    pub(super) idle_watch: WatchSet,
    /// Characters of consumed text any boundary may look behind at.
    pub(super) max_lookbehind: usize,
    warned_missing_anchor: AtomicBool,
}

/// One compiled field.
#[derive(Debug)]
pub(super) struct Region {
    pub name: FieldName,
    /// `None`: the implicit region.
    pub open: Option<Boundary>,
    /// `None`: the region runs to the end of the stream.
    pub close: Option<Boundary>,
    /// Boundaries watched inside this region: its close, if any.
    pub close_watch: WatchSet,
    pub optional: bool,
    pub kind: RegionKind,
}

/// A field this parser can report, in the lexical order of its name, which
/// Transformers uses to break ties between boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FieldName {
    Content,
    ReasoningContent,
    Thinking,
    ToolCalls,
}

/// How a region's content is reported.
#[derive(Debug)]
pub(super) enum RegionKind {
    /// Streamed as it arrives.
    Text(TextRegion),
    /// Buffered, and parsed into tool calls at close.
    ToolCalls(ToolCallRegion),
}

#[derive(Debug)]
pub(super) struct TextRegion {
    pub role: TextRole,
    /// Whether the value is stripped (`content_args.strip`).
    pub strip: bool,
    pub repeat: Repeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TextRole {
    Content,
    Reasoning,
}

/// How repeated occurrences of a text field combine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Repeat {
    /// No `repeats`: occurrences stream as one value.
    Once,
    /// `repeats` without `join`: each occurrence is its own value.
    Each,
    /// `repeats` with `join`: occurrences are joined by the separator.
    Join(String),
}

#[derive(Debug)]
pub(super) struct ToolCallRegion {
    pub content: ContentParser,
    pub transform: Option<FieldTransform>,
    /// Opener capture group that provides the function name, allowing the tool
    /// call to start before its arguments are complete.
    pub name_group: Option<String>,
}

/// A region delimiter, compiled as `_compile_anchor` does.
#[derive(Debug)]
pub(super) struct Boundary {
    regex: Regex,
    /// The alternatives of a literal anchor, longest first; `None` for a
    /// pattern.
    pub literals: Option<Vec<String>>,
    /// One literal is a proper prefix of another, so a literal match at the
    /// buffer edge may still grow.
    can_extend: bool,
}

/// A boundary occurrence found by [`Boundary::scan`].
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Hit {
    /// A partial match, or a complete one that more input may still change:
    /// nothing from `start` on can be committed yet.
    Pending { start: usize },
    /// A match that can be committed.
    Match {
        start: usize,
        end: usize,
        captures: Captures,
    },
}

/// One boundary the executor watches for in a given state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Watch {
    pub region: usize,
    pub kind: WatchKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WatchKind {
    Open,
    Close,
}

/// The boundaries watched in one parser mode.
#[derive(Debug, Default)]
pub(super) struct WatchSet {
    pub boundaries: Vec<Watch>,
}

impl ResponseTemplate {
    /// Compile the `response_template` object from `tokenizer_config.json`.
    pub fn from_json(value: &Value) -> Result<Self> {
        let spec = TemplateSpec::from_value(value)?;

        let regions = spec
            .fields
            .iter()
            .enumerate()
            .map(|(index, (name, field))| Region::compile(index, name, field))
            .collect::<Result<Vec<_>>>()?;

        // A field without an open anchor is the implicit-open / leftover sink; at most one is allowed.
        let implicit_fields: Vec<_> =
            regions.iter().enumerate().filter(|(_, region)| region.open.is_none()).collect();
        if implicit_fields.len() > 1 {
            let names: Vec<_> =
                implicit_fields.iter().map(|(_, region)| region.name.as_str()).collect();
            return Err(invalid!(
                "At most one field may omit 'open'/'open_pattern' (that field becomes the \
                 implicit-open / leftover sink). Found: {}",
                names.join(", ")
            ));
        }
        let implicit = implicit_fields.first().map(|(index, _)| *index);

        let start_anchor = match spec.start_anchor()? {
            AnchorSpec::Literals(mut literals) => {
                // Sort longest-first so alternation prefers the longer alternative when both could match.
                literals.sort_by_key(|literal| std::cmp::Reverse(literal.len()));
                Regex::new("response_template", &alternation(&literals))?
            }
            AnchorSpec::Pattern(pattern) => Regex::new("response_template", pattern)?,
        };

        let opens = regions.iter().enumerate().filter_map(|(region, spec)| {
            let watch = Watch {
                region,
                kind: WatchKind::Open,
            };
            Some((watch, spec.open.as_ref()?))
        });
        let implicit_close = implicit.and_then(|region| {
            let watch = Watch {
                region,
                kind: WatchKind::Close,
            };
            Some((watch, regions[region].close.as_ref()?))
        });
        let idle_watch = WatchSet::new(opens.chain(implicit_close));
        let max_lookbehind = regions
            .iter()
            .flat_map(|region| [&region.open, &region.close])
            .flatten()
            .map(Boundary::max_lookbehind)
            .max()
            .unwrap_or(0);

        Ok(Self {
            start_anchor,
            regions,
            implicit,
            idle_watch,
            max_lookbehind,
            warned_missing_anchor: AtomicBool::new(false),
        })
    }

    /// The `xml-inline` tag pattern of tool-call arguments, if any.
    pub fn argument_tag_pattern(&self) -> Option<&str> {
        self.regions.iter().find_map(|region| match &region.kind {
            RegionKind::ToolCalls(calls) => calls.content.tag_pattern(),
            _ => None,
        })
    }

    /// The literal closers of the reasoning field, longest first, or none if it
    /// has no literal close.
    pub fn reasoning_closers(&self) -> &[String] {
        self.regions
            .iter()
            .find_map(|region| match (&region.kind, &region.close) {
                (RegionKind::Text(text), Some(close)) if text.role == TextRole::Reasoning => {
                    close.literals.as_deref()
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Return the start and end offsets of the last start-anchor match in
    /// `prompt`.
    pub(super) fn last_start_anchor(
        &self,
        prompt: &str,
    ) -> std::result::Result<Option<(usize, usize)>, MatchError> {
        let matches = self.start_anchor.find_iter(prompt)?;
        Ok(matches.last().map(|found| (found.start, found.end)))
    }

    /// Log once per template that the prompt carried no start anchor.
    pub(super) fn warn_missing_anchor_once(&self) {
        if !self.warned_missing_anchor.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "response_template start anchor not found in the prompt; parsing starts from the \
                 template's initial state (the chat template may not match the checkpoint's \
                 response_template)"
            );
        }
    }

    /// The boundary of a watch entry.
    pub(super) fn boundary(&self, watch: Watch) -> &Boundary {
        let region = &self.regions[watch.region];
        match watch.kind {
            WatchKind::Open => region.open.as_ref(),
            WatchKind::Close => region.close.as_ref(),
        }
        .expect("watched boundaries exist")
    }
}

impl Region {
    /// Validate the spec of the field at `index` and compile it.
    fn compile(index: usize, name: &str, field: &FieldSpec) -> Result<Self> {
        let scope = format!("Field '{name}'");
        let open = field.open(&scope)?.map(|spec| Boundary::compile(&scope, spec)).transpose()?;
        let close = field.close(&scope)?.map(|spec| Boundary::compile(&scope, spec)).transpose()?;

        let content = ContentParser::new(&scope, field.content, &field.content_args)?;
        let repeat = match (field.repeats, &field.join) {
            (false, None) => Repeat::Once,
            (true, None) => Repeat::Each,
            (true, Some(join)) => Repeat::Join(join.clone()),
            (false, Some(_)) => return Err(invalid!("{scope}: 'join' requires 'repeats': true")),
        };
        let transform = match (&field.transform, field.transform_each) {
            (Some(template), each) => Some(FieldTransform {
                template: Transform::compile(&scope, template)?,
                each,
            }),
            (None, false) => None,
            (None, true) => {
                return Err(invalid!(
                    "{scope}: transform_each is set but no transform was provided"
                ));
            }
        };

        let captured: Vec<&str> =
            [&open, &close].into_iter().flatten().flat_map(Boundary::group_names).collect();
        match &transform {
            None if !captured.is_empty() => {
                // Named captures only reach the output through a transform, so flag any that would be silently dropped.
                return Err(invalid!(
                    "{scope}: open_pattern/close_pattern declares named group(s) {captured:?}, but \
                     the field has no 'transform'. Named captures are only surfaced through a \
                     'transform' template (where they appear alongside 'content'). Either add a \
                     'transform' that uses the captures, or remove the named groups from the pattern."
                ));
            }
            // Without `transform_each`, the scope is the opener's captures plus `content`;
            // any other placeholder fails on every match.
            Some(FieldTransform {
                template,
                each: false,
            }) => {
                let open_groups: Vec<&str> = open.iter().flat_map(Boundary::group_names).collect();
                if let Some(root) = template
                    .placeholder_roots()
                    .find(|root| *root != "content" && !open_groups.contains(root))
                {
                    return Err(invalid!(
                        "{scope}: transform placeholder '{{{root}}}' is neither 'content' nor a \
                         named group of open_pattern"
                    ));
                }
            }
            _ => {}
        }

        let name = FieldName::parse(&scope, name)?;
        let kind = match name.text_role() {
            Some(role) => {
                let strip = match (content, &transform) {
                    (ContentParser::Text(TextArgs { strip }), None) => strip,
                    _ => {
                        return Err(unsupported!(
                            "{scope}: text and reasoning fields must use the 'text' content \
                             parser without a transform"
                        ));
                    }
                };
                RegionKind::Text(TextRegion {
                    role,
                    strip,
                    repeat,
                })
            }
            None => {
                if matches!(repeat, Repeat::Join(_)) {
                    return Err(unsupported!(
                        "{scope}: 'join' requires each match to parse to a string"
                    ));
                }
                let name_group = name_group(transform.as_ref(), open.as_ref());
                RegionKind::ToolCalls(ToolCallRegion {
                    content,
                    transform,
                    name_group,
                })
            }
        };

        let close_watch = WatchSet::new(close.as_ref().map(|close| {
            let watch = Watch {
                region: index,
                kind: WatchKind::Close,
            };
            (watch, close)
        }));

        Ok(Self {
            name,
            open,
            close,
            close_watch,
            optional: field.optional,
            kind,
        })
    }
}

/// The opener capture group that a non-`transform_each` transform uses as the
/// function name, if any.
fn name_group(transform: Option<&FieldTransform>, open: Option<&Boundary>) -> Option<String> {
    let (
        FieldTransform {
            template,
            each: false,
        },
        open,
    ) = (transform?, open?)
    else {
        return None;
    };
    let path = template
        .placeholder_at(&["function", "name"])
        .or_else(|| template.placeholder_at(&["name"]))?;
    match path {
        [root] if open.group_names().any(|group| group == root) => Some(root.clone()),
        _ => None,
    }
}

impl FieldName {
    /// Map a field name to a reportable field.
    fn parse(scope: &str, name: &str) -> Result<Self> {
        Ok(match name {
            "content" => Self::Content,
            "reasoning_content" => Self::ReasoningContent,
            "thinking" => Self::Thinking,
            "tool_calls" => Self::ToolCalls,
            _ => {
                return Err(unsupported!(
                    "{scope}: only 'content', 'reasoning_content'/'thinking', and 'tool_calls' \
                     fields can be reported"
                ));
            }
        })
    }

    /// The role of a text field; `None` for `tool_calls`.
    fn text_role(self) -> Option<TextRole> {
        match self {
            Self::Content => Some(TextRole::Content),
            Self::ReasoningContent | Self::Thinking => Some(TextRole::Reasoning),
            Self::ToolCalls => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::ReasoningContent => "reasoning_content",
            Self::Thinking => "thinking",
            Self::ToolCalls => "tool_calls",
        }
    }
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Boundary {
    /// Compile an anchor into a boundary.
    fn compile(scope: &str, spec: AnchorSpec<'_>) -> Result<Self> {
        Ok(match spec {
            AnchorSpec::Literals(mut literals) => {
                // Sort longest-first so the longer alternative wins when both match.
                literals.sort_by_key(|literal| std::cmp::Reverse(literal.len()));
                let can_extend = literals.iter().any(|long| {
                    literals.iter().any(|short| long != short && long.starts_with(short.as_str()))
                });
                Self {
                    regex: Regex::new(scope, &alternation(&literals))?,
                    literals: Some(literals),
                    can_extend,
                }
            }
            AnchorSpec::Pattern(pattern) => Self {
                regex: Regex::new(scope, pattern)?,
                literals: None,
                can_extend: false,
            },
        })
    }

    /// Named capture groups.
    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        self.regex.group_names().map(|(_, name)| name)
    }

    fn max_lookbehind(&self) -> usize {
        self.regex.max_lookbehind()
    }

    /// One delimiter's step of `_scan`: the first occurrence in `subject` from
    /// byte offset `start`, text before which is context only. Before the end
    /// of the stream, an occurrence more input may change is pending.
    /// `allow_empty_at_start: false` skips an empty match at `start`.
    pub fn scan(
        &self,
        subject: &str,
        start: usize,
        eof: bool,
        allow_empty_at_start: bool,
    ) -> std::result::Result<Option<Hit>, MatchError> {
        let Some(found) = self.regex.search(subject, start, !eof, allow_empty_at_start)? else {
            return Ok(None);
        };
        if !eof && (found.partial || self.can_grow(found.start, found.end, subject.len())) {
            return Ok(Some(Hit::Pending { start: found.start }));
        }
        let captures = self
            .regex
            .group_names()
            .filter_map(|(index, name)| {
                let span = found.groups[index].clone()?;
                Some((name.to_string(), Value::String(subject[span].to_string())))
            })
            .collect();
        Ok(Some(Hit::Match {
            start: found.start,
            end: found.end,
            captures,
        }))
    }

    /// `_can_grow`: whether a complete match may still change with more input.
    /// One ending before the edge has seen its terminating character; at the
    /// edge, an empty match (`$`) is real only at the end of the stream, and a
    /// literal no other literal extends cannot grow.
    fn can_grow(&self, start: usize, end: usize, len: usize) -> bool {
        if end != len {
            return false;
        }
        start == end || self.literals.is_none() || self.can_extend
    }
}

/// A pattern matching any of `literals`, earlier ones first.
fn alternation(literals: &[String]) -> String {
    let escaped: Vec<String> = literals
        .iter()
        .map(|literal| {
            let mut out = String::with_capacity(literal.len());
            for c in literal.chars() {
                // A backslash makes any ASCII non-alphanumeric character literal.
                if c.is_ascii() && !c.is_ascii_alphanumeric() {
                    out.push('\\');
                }
                out.push(c);
            }
            out
        })
        .collect();
    escaped.join("|")
}

impl WatchSet {
    fn new<'a>(entries: impl IntoIterator<Item = (Watch, &'a Boundary)>) -> Self {
        Self {
            boundaries: entries.into_iter().map(|(watch, _)| watch).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVOKE_OPEN: &str = r#"<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">"#;
    const GEMMA4_CALL: &str = r"<\|tool_call>call:(?P<name>\w+)";
    const GPT_OSS_CALL: &str =
        r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>";
    const OPENER: &str =
        "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>";
    /// A call ends where the next one starts.
    const CALL_CLOSE: &str = r"</tool_call>|(?=<tool_call>)";
    /// The opener stops before the first argument and consumes the whitespace
    /// before its tag.
    const GLM_OPEN: &str = r"\s*<tool_call>\s*(?P<name>[^<]*?)\s*(?=<arg_key>|</tool_call>)";

    fn pending(start: usize) -> Option<Hit> {
        Some(Hit::Pending { start })
    }

    fn matched(span: std::ops::Range<usize>, captures: &[(&str, &str)]) -> Option<Hit> {
        Some(Hit::Match {
            start: span.start,
            end: span.end,
            captures: captures
                .iter()
                .map(|(name, value)| (name.to_string(), Value::from(*value)))
                .collect(),
        })
    }

    /// Expected values are the decision of `_scan` for the one delimiter:
    /// `regex.search(text, partial=not eos)` and `_can_grow`.
    #[test]
    fn scan_decides_like_transformers() {
        let cases: &[(&str, &str, bool, Option<Hit>)] = &[
            (
                INVOKE_OPEN,
                "<atem:invoke name=\"get\">rest",
                false,
                matched(0..24, &[("name", "get")]),
            ),
            (
                INVOKE_OPEN,
                "<atem:invoke title=\"天气\" name=\"获取\">rest",
                false,
                matched(0..42, &[("name", "获取")]),
            ),
            (INVOKE_OPEN, "<atem:invoke name=\"get", false, pending(0)),
            (INVOKE_OPEN, "<atem:invoker name=\"x\">r", false, None),
            (INVOKE_OPEN, "<atem:invoke xname=\"x\">r", false, None),
            (
                INVOKE_OPEN,
                "<atem:invoke é name=\"x\">r",
                false,
                matched(0..25, &[("name", "x")]),
            ),
            (INVOKE_OPEN, "<atem:invoke éname=\"x\">r", false, None),
            (INVOKE_OPEN, "<atem:invoke 使用标签", false, pending(0)),
            (INVOKE_OPEN, "<atem:invoke 使用标签", true, None),
            (INVOKE_OPEN, "<atem:invoke 使用> 正文", false, None),
            (r"a\b|ab", "ab!", true, matched(0..2, &[])),
            (GEMMA4_CALL, "<|tool_call>call:get_wea", false, pending(0)),
            (
                GEMMA4_CALL,
                "<|tool_call>call:get_weather{",
                false,
                matched(0..28, &[("name", "get_weather")]),
            ),
            (
                GEMMA4_CALL,
                "<|tool_call>call:天气{",
                false,
                matched(0..23, &[("name", "天气")]),
            ),
            (GEMMA4_CALL, "<|tool_call>call:{", false, None),
            (
                GEMMA4_CALL,
                "<|tool_call>call:get",
                true,
                matched(0..20, &[("name", "get")]),
            ),
            (GPT_OSS_CALL, OPENER, false, pending(0)),
            (
                GPT_OSS_CALL,
                &format!("{OPENER}{{\"a\""),
                false,
                matched(0..75, &[("name", "get_weather")]),
            ),
            (CALL_CLOSE, "<tool_", false, pending(0)),
            (CALL_CLOSE, "<tool_call>", false, matched(0..0, &[])),
            (CALL_CLOSE, "<tool_call>", true, matched(0..0, &[])),
            (CALL_CLOSE, "<tool_x", false, None),
            (CALL_CLOSE, "</tool_call>\n", false, matched(0..12, &[])),
            (CALL_CLOSE, "text<tool_call>", false, matched(4..4, &[])),
            (GLM_OPEN, "<tool_call>get_w", false, pending(0)),
            (GLM_OPEN, "<tool_call>get_w\n<arg_", false, pending(0)),
            (
                GLM_OPEN,
                "<tool_call>get_w\n<arg_key>city",
                false,
                matched(0..17, &[("name", "get_w")]),
            ),
            (
                GLM_OPEN,
                " \n<tool_call>f</tool_call>\n",
                false,
                matched(0..14, &[("name", "f")]),
            ),
            (GLM_OPEN, " \n", false, pending(0)),
            (GLM_OPEN, " x", false, None),
            (
                GLM_OPEN,
                "a <tool_call>f<arg_key>",
                false,
                matched(1..14, &[("name", "f")]),
            ),
        ];
        for (pattern, text, eof, expected) in cases {
            let boundary = Boundary::compile("test", AnchorSpec::Pattern(pattern)).unwrap();
            let hit = boundary.scan(text, 0, *eof, true).unwrap();
            // A pending occurrence at the end holds nothing back.
            let hit = hit.filter(|hit| *hit != Hit::Pending { start: text.len() });
            assert_eq!(&hit, expected, "{pattern:?} on {text:?}, eof: {eof}");
        }
    }

    #[test]
    fn literal_matches_commit_unless_another_literal_extends_them() {
        let close = |literals: &[&str]| {
            let literals = literals.iter().map(|literal| literal.to_string()).collect();
            Boundary::compile("test", AnchorSpec::Literals(literals)).unwrap()
        };
        let think = close(&["</think>"]);
        assert_eq!(
            think.scan("a</think>", 0, false, true).unwrap(),
            matched(1..9, &[])
        );
        assert_eq!(think.scan("a</thi", 0, false, true).unwrap(), pending(1));
        let calls = close(&["[TOOL_CALLS]", "[TOOL_CALLS]v2"]);
        assert_eq!(
            calls.scan("[TOOL_CALLS]", 0, false, true).unwrap(),
            pending(0)
        );
        assert_eq!(
            calls.scan("[TOOL_CALLS]", 0, true, true).unwrap(),
            matched(0..12, &[])
        );
    }
}
