// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Compiled `response_template`.
//!
//! Validation follows `response_templates.py`; the compiled form additionally
//! maps fields to parser roles and precomputes each state's boundary markers.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use regex_automata::meta::Regex;
use serde_json::Value;

use super::content::{ContentParser, TextArgs, compile_regex};
use super::pattern::{Captures, Resolution, StreamingPattern};
use super::spec::{AnchorSpec, FieldSpec, TemplateSpec};
use super::transform::{FieldTransform, Transform};
use super::{Result, invalid, unsupported};

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

/// A region delimiter.
#[derive(Debug)]
pub(super) enum Boundary {
    /// Literal alternatives, longest first.
    Literals(Vec<String>),
    Pattern(Box<StreamingPattern>),
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
    /// Literals, deduplicated, whose occurrences are the only positions any of
    /// the boundaries can match at.
    pub markers: Vec<String>,
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
                let alternation: Vec<_> =
                    literals.iter().map(|literal| regex_syntax::escape(literal)).collect();
                compile_regex("response_template", &alternation.join("|"))?
            }
            AnchorSpec::Pattern(pattern) => compile_regex("response_template", pattern)?,
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

        Ok(Self {
            start_anchor,
            regions,
            implicit,
            idle_watch,
            warned_missing_anchor: AtomicBool::new(false),
        })
    }

    /// Return the start and end offsets of the last start-anchor match in
    /// `prompt`.
    pub(super) fn last_start_anchor(&self, prompt: &str) -> Option<(usize, usize)> {
        self.start_anchor
            .find_iter(prompt)
            .last()
            .map(|found| (found.start(), found.end()))
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
                Self::Literals(literals)
            }
            AnchorSpec::Pattern(pattern) => {
                Self::Pattern(Box::new(StreamingPattern::new(scope, pattern)?))
            }
        })
    }

    /// Literals whose occurrences are the only positions this boundary can match at.
    pub fn markers(&self) -> &[String] {
        match self {
            Self::Literals(literals) => literals,
            Self::Pattern(pattern) => pattern.prefixes(),
        }
    }

    /// Named capture groups.
    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::Literals(_) => None,
            Self::Pattern(pattern) => Some(pattern.group_names()),
        }
        .into_iter()
        .flatten()
    }

    /// Resolve this boundary at the start of `text`.
    pub fn resolve(&self, text: &str, eof: bool) -> Resolution {
        let markers = self.markers();
        // A marker the available text is a proper prefix of: more input decides.
        let partial = !eof
            && markers
                .iter()
                .any(|marker| marker.len() > text.len() && marker.starts_with(text));
        match self {
            Self::Literals(literals) => {
                if partial {
                    return Resolution::Pending;
                }
                match literals.iter().find(|literal| text.starts_with(literal.as_str())) {
                    Some(literal) => Resolution::Matched {
                        len: literal.len(),
                        captures: Captures::new(),
                    },
                    None => Resolution::NoMatch,
                }
            }
            Self::Pattern(pattern) => {
                if markers.iter().any(|marker| text.starts_with(marker.as_str())) {
                    pattern.resolve(text, eof)
                } else if partial {
                    Resolution::Pending
                } else {
                    Resolution::NoMatch
                }
            }
        }
    }
}

impl WatchSet {
    /// Collect the watched boundaries and their deduplicated markers.
    fn new<'a>(entries: impl IntoIterator<Item = (Watch, &'a Boundary)>) -> Self {
        let mut set = Self::default();
        for (watch, boundary) in entries {
            set.boundaries.push(watch);
            for marker in boundary.markers() {
                if !set.markers.contains(marker) {
                    set.markers.push(marker.clone());
                }
            }
        }
        set
    }
}
