// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Compiled `response_template`.
//!
//! Validation follows `response_templates.py`; the compiled form additionally
//! maps fields to parser roles and precomputes each state's boundary candidates.

use std::sync::atomic::{AtomicBool, Ordering};

use regex_automata::meta::Regex;
use serde_json::Value;

use super::content::{ContentParser, compile_regex};
use super::pattern::{Resolution, StreamingPattern};
use super::schema::{AnchorSpec, FieldSpec, TemplateSpec, anchor_spec};
use super::transform::Transform;
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
    pub name: String,
    pub role: Role,
    pub open: Option<Boundary>,
    /// `None`: the region runs to the end of the stream.
    pub close: Option<Boundary>,
    pub close_watch: Vec<Watch>,
    pub close_candidates: Vec<String>,
    pub content: ContentParser,
    pub transform: Option<Transform>,
    pub transform_each: bool,
    pub repeats: bool,
    pub join: Option<String>,
    pub optional: bool,
    /// Opener capture group that provides the function name, allowing the tool
    /// call to start before its arguments are complete.
    pub early_name: Option<String>,
}

/// How a field's content is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Role {
    Text,
    Reasoning,
    ToolCalls,
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
            .field_specs()?
            .into_iter()
            .map(|(name, field)| Region::compile(name, &field))
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

        let start_anchor = match anchor_spec(
            "response_template",
            spec.start_anchor.as_ref(),
            spec.start_anchor_pattern.as_deref(),
            "start_anchor",
            "start_anchor_pattern",
        )? {
            None => {
                return Err(invalid!(
                    "response_template must define 'start_anchor' or 'start_anchor_pattern'."
                ));
            }
            Some(AnchorSpec::Literals(mut literals)) => {
                // Sort longest-first so alternation prefers the longer alternative when both could match.
                literals.sort_by_key(|literal| std::cmp::Reverse(literal.len()));
                let alternation: Vec<_> =
                    literals.iter().map(|literal| regex_syntax::escape(literal)).collect();
                compile_regex("response_template", &alternation.join("|"))?
            }
            Some(AnchorSpec::Pattern(pattern)) => compile_regex("response_template", pattern)?,
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
        let content = ContentParser::new(&scope, &field.content, &field.content_args)?;
        let open = anchor_spec(
            &scope,
            field.open.as_ref(),
            field.open_pattern.as_deref(),
            "open",
            "open_pattern",
        )?
        .map(|spec| Boundary::compile(&scope, spec))
        .transpose()?;
        let close = anchor_spec(
            &scope,
            field.close.as_ref(),
            field.close_pattern.as_deref(),
            "close",
            "close_pattern",
        )?
        .map(|spec| Boundary::compile(&scope, spec))
        .transpose()?;

        if field.join.is_some() && !field.repeats {
            return Err(invalid!("{scope}: 'join' requires 'repeats': true"));
        }
        if field.transform_each && field.transform.is_none() {
            return Err(invalid!(
                "{scope}: transform_each is set but no transform was provided"
            ));
        }
        let transform = field
            .transform
            .as_ref()
            .map(|transform| Transform::compile(&scope, transform))
            .transpose()?;

        let captured: Vec<&str> =
            [&open, &close].into_iter().flatten().flat_map(Boundary::group_names).collect();
        if transform.is_none() && !captured.is_empty() {
            // Named captures only reach the output through a transform, so flag any that would be silently dropped.
            return Err(invalid!(
                "{scope}: open_pattern/close_pattern declares named group(s) {captured:?}, but the \
                 field has no 'transform'. Named captures are only surfaced through a 'transform' \
                 template (where they appear alongside 'content'). Either add a 'transform' that \
                 uses the captures, or remove the named groups from the pattern."
            ));
        }

        let role = match name {
            "content" => Role::Text,
            "reasoning_content" | "thinking" => Role::Reasoning,
            "tool_calls" => Role::ToolCalls,
            _ => {
                return Err(unsupported!(
                    "{scope}: only 'content', 'reasoning_content'/'thinking', and 'tool_calls' \
                     fields can be reported"
                ));
            }
        };
        if role == Role::ToolCalls && field.join.is_some() {
            return Err(unsupported!(
                "{scope}: 'join' requires each match to parse to a string"
            ));
        }
        if role != Role::ToolCalls && (!content.is_text() || transform.is_some()) {
            return Err(unsupported!(
                "{scope}: text and reasoning fields must use the 'text' content parser without a transform"
            ));
        }

        let early_name = match (&transform, &open) {
            (Some(transform), Some(open)) if role == Role::ToolCalls && !field.transform_each => {
                let path = transform
                    .placeholder_at(&["function", "name"])
                    .or_else(|| transform.placeholder_at(&["name"]));
                path.and_then(|path| match path {
                    [root] if open.group_names().any(|group| group == root) => Some(root.clone()),
                    _ => None,
                })
            }
            _ => None,
        };

        Ok(Self {
            name: name.to_string(),
            role,
            open,
            close,
            close_watch: Vec::new(),
            close_candidates: Vec::new(),
            content,
            transform,
            transform_each: field.transform_each,
            repeats: field.repeats,
            join: field.join.clone(),
            optional: field.optional,
            early_name,
        })
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
