// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Compiled `response_template`.
//!
//! Validation follows `response_templates.py`; the compiled form additionally
//! maps fields to parser roles and precomputes each state's boundary candidates.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use regex_automata::meta::Regex;
use serde_json::Value;

use super::content::{ContentParser, TextArgs, compile_regex};
use super::pattern::{Resolution, StreamingPattern};
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
    pub(super) idle_watch: Vec<Watch>,
    pub(super) idle_candidates: Vec<String>,
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
    pub close_watch: Vec<Watch>,
    pub close_candidates: Vec<String>,
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
    pub early_name: Option<String>,
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

impl ResponseTemplate {
    /// Compile the `response_template` object from `tokenizer_config.json`.
    pub fn from_json(value: &Value) -> Result<Self> {
        let spec = TemplateSpec::from_value(value)?;

        let regions = spec
            .fields
            .iter()
            .map(|(name, field)| Region::compile(name, field))
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

        let mut idle_watch: Vec<_> = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.open.is_some())
            .map(|(region, _)| Watch {
                region,
                kind: WatchKind::Open,
            })
            .collect();
        if let Some(region) = implicit
            && regions[region].close.is_some()
        {
            idle_watch.push(Watch {
                region,
                kind: WatchKind::Close,
            });
        }
        let idle_candidates = candidates(&regions, &idle_watch);

        let mut template = Self {
            start_anchor,
            regions,
            implicit,
            idle_watch,
            idle_candidates,
            warned_missing_anchor: AtomicBool::new(false),
        };
        for region in 0..template.regions.len() {
            let close_watch: Vec<_> = template.regions[region]
                .close
                .is_some()
                .then_some(Watch {
                    region,
                    kind: WatchKind::Close,
                })
                .into_iter()
                .collect();
            template.regions[region].close_candidates = candidates(&template.regions, &close_watch);
            template.regions[region].close_watch = close_watch;
        }
        Ok(template)
    }

    /// Return the end offset of the last start-anchor match in `prompt`, and
    /// the start offset of that match.
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
    /// Validate a single field spec and compile it.
    fn compile(name: &str, field: &FieldSpec) -> Result<Self> {
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
                let early_name = early_name(transform.as_ref(), open.as_ref());
                RegionKind::ToolCalls(ToolCallRegion {
                    content,
                    transform,
                    early_name,
                })
            }
        };

        Ok(Self {
            name,
            open,
            close,
            close_watch: Vec::new(),
            close_candidates: Vec::new(),
            optional: field.optional,
            kind,
        })
    }
}

/// The opener capture group that a non-`transform_each` transform uses as the
/// function name, if any.
fn early_name(transform: Option<&FieldTransform>, open: Option<&Boundary>) -> Option<String> {
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
    pub fn candidates(&self) -> &[String] {
        match self {
            Self::Literals(literals) => literals,
            Self::Pattern(pattern) => pattern.prefixes(),
        }
    }

    /// Named capture groups.
    pub fn group_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            Self::Literals(_) => Box::new(std::iter::empty()),
            Self::Pattern(pattern) => Box::new(pattern.group_names()),
        }
    }

    /// Resolve this boundary at the start of `text`.
    pub fn resolve(&self, text: &str, eof: bool) -> Resolution {
        let candidates = self.candidates();
        // A candidate the available text is a proper prefix of: more input decides.
        let partial = !eof
            && candidates
                .iter()
                .any(|candidate| candidate.len() > text.len() && candidate.starts_with(text));
        match self {
            Self::Literals(literals) => {
                if partial {
                    return Resolution::Pending;
                }
                match literals.iter().find(|literal| text.starts_with(literal.as_str())) {
                    Some(literal) => Resolution::Matched {
                        len: literal.len(),
                        captures: Vec::new(),
                    },
                    None => Resolution::NoMatch,
                }
            }
            Self::Pattern(pattern) => {
                if candidates.iter().any(|prefix| text.starts_with(prefix.as_str())) {
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

/// Collect the scan candidates of `watch`, deduplicated.
fn candidates(regions: &[Region], watch: &[Watch]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for entry in watch {
        let region = &regions[entry.region];
        let boundary = match entry.kind {
            WatchKind::Open => region.open.as_ref(),
            WatchKind::Close => region.close.as_ref(),
        };
        for candidate in boundary.into_iter().flat_map(Boundary::candidates) {
            if !out.contains(candidate) {
                out.push(candidate.clone());
            }
        }
    }
    out
}
