// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Reject custom-role turns a chat template renders away.
//!
//! Mirrors `assert_chat_template_rendered_messages` in the Python frontend.
//! Templates that do not filter on role are left alone, as are the standard
//! OpenAI roles (`system`, `user`, `assistant`, `tool`, `developer`,
//! `function`): those may be omitted on purpose. Any other non-empty role is
//! probed. One extra render appends a marker to each suspect; a missing
//! marker is a drop only when clearing that text leaves the prompt unchanged.
//! Tool and media parts are stripped in place so echoed text cannot hide them.

use std::collections::HashSet;

use uuid::Uuid;

use super::{TemplateContent, TemplateContentPart, TemplateMessage};
use crate::error::{Error, Result};

/// Extra template renders spent confirming a drop. Unresolved turns are kept.
const MAX_EXTRA_RENDERS: u32 = 8;

/// Whether `source` dispatches on role, the same cases the Python AST walk
/// treats as a filter: a comparison of `.role` / `['role']`, or
/// `selectattr` / `rejectattr` on `role`.
pub(super) fn template_filters_by_role(source: &str) -> bool {
    let mentions_role =
        source.contains(".role") || source.contains("['role']") || source.contains("[\"role\"]");
    let compares = source.contains("==") || source.contains("!=") || source.contains(" in ");
    if mentions_role && compares {
        return true;
    }
    (source.contains("selectattr") || source.contains("rejectattr"))
        && (source.contains("'role'") || source.contains("\"role\""))
}

/// Raise when a non-empty custom role contributed nothing to `baseline`.
///
/// `render` must apply the same template, tools, and generation options as
/// the render that produced `baseline`. A render that fails is treated as
/// unresolved and the turn is kept.
pub(super) fn reject_dropped_custom_roles<'a>(
    messages: &[TemplateMessage<'a>],
    baseline: &str,
    render: &mut dyn FnMut(&[TemplateMessage<'a>]) -> Option<String>,
) -> Result<()> {
    let mut text_suspects = Vec::new();
    let mut struct_suspects = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if is_standard_role(message.role) {
            continue;
        }
        if has_nonempty_text(&message.content) {
            text_suspects.push(index);
        }
        if has_structural_payload(&message.content) {
            struct_suspects.push(index);
        }
    }
    if text_suspects.is_empty() && struct_suspects.is_empty() {
        return Ok(());
    }

    let mut budget = MAX_EXTRA_RENDERS;
    let text_missing = if text_suspects.is_empty() {
        HashSet::new()
    } else {
        indexes_missing_text(messages, &text_suspects, baseline, render, &mut budget)
    };
    let struct_missing = if struct_suspects.is_empty() {
        HashSet::new()
    } else {
        dropped_indexes(
            messages,
            &struct_suspects,
            baseline,
            render,
            &mut budget,
            strip_structural_payload,
        )
    };

    let mut roles = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if is_standard_role(message.role) {
            continue;
        }
        let structural = struct_suspects.contains(&index);
        // A rendered tool call or image keeps the turn, even when its text
        // is absent. A dropped structural payload rejects the turn on its own.
        let dropped = if structural && struct_missing.contains(&index) {
            true
        } else {
            let contributed = structural && !struct_missing.contains(&index);
            text_missing.contains(&index) && !contributed
        };
        if dropped && !roles.iter().any(|role: &String| role == message.role) {
            roles.push(message.role.to_owned());
        }
    }
    if roles.is_empty() {
        return Ok(());
    }

    let noun = if roles.len() == 1 { "role" } else { "roles" };
    let roles = roles.iter().map(|role| format!("'{role}'")).collect::<Vec<_>>().join(", ");
    Err(Error::DroppedChatRole { noun, roles })
}

fn is_standard_role(role: &str) -> bool {
    matches!(
        role,
        "system" | "user" | "assistant" | "tool" | "developer" | "function"
    )
}

fn has_nonempty_text(content: &TemplateContent) -> bool {
    match content {
        TemplateContent::String(text) => !text.trim().is_empty(),
        TemplateContent::OpenAi(parts) => parts.iter().any(
            |part| matches!(part, TemplateContentPart::Text { text } if !text.trim().is_empty()),
        ),
    }
}

fn has_structural_payload(content: &TemplateContent) -> bool {
    match content {
        TemplateContent::String(_) => false,
        TemplateContent::OpenAi(parts) => {
            parts.iter().any(|part| !matches!(part, TemplateContentPart::Text { .. }))
        }
    }
}

fn append_probe(content: &mut TemplateContent, token: &str) -> bool {
    match content {
        TemplateContent::String(text) if !text.trim().is_empty() => {
            text.push_str(token);
            true
        }
        TemplateContent::OpenAi(parts) => {
            let mut injected = false;
            for part in parts {
                if let TemplateContentPart::Text { text } = part
                    && !text.trim().is_empty()
                {
                    text.push_str(token);
                    injected = true;
                }
            }
            injected
        }
        _ => false,
    }
}

fn blank_text(content: &mut TemplateContent) {
    match content {
        TemplateContent::String(text) => text.clear(),
        TemplateContent::OpenAi(parts) => {
            for part in parts {
                if let TemplateContentPart::Text { text } = part {
                    text.clear();
                }
            }
        }
    }
}

fn strip_structural_payload(content: &mut TemplateContent) {
    if let TemplateContent::OpenAi(parts) = content {
        parts.retain(|part| matches!(part, TemplateContentPart::Text { .. }));
    }
}

fn probe_prefix() -> String {
    let hex = Uuid::new_v4().simple().to_string();
    format!("vllmprobe{}", &hex[..12])
}

/// Suspects whose text is absent from the prompt.
///
/// A missing marker is not enough: the template may keep the original text
/// and clip a suffix. Those indexes are dropped only when clearing the text
/// leaves `baseline` unchanged.
fn indexes_missing_text<'a>(
    messages: &[TemplateMessage<'a>],
    suspects: &[usize],
    baseline: &str,
    render: &mut dyn FnMut(&[TemplateMessage<'a>]) -> Option<String>,
    budget: &mut u32,
) -> HashSet<usize> {
    if *budget == 0 {
        return HashSet::new();
    }
    let prefix = probe_prefix();
    let mut probed = messages.to_vec();
    let mut tokens = Vec::new();
    for index in suspects {
        let token = format!("{prefix}_{index:06}_");
        if append_probe(&mut probed[*index].content, &token) {
            tokens.push((*index, token));
        }
    }
    if tokens.is_empty() {
        return dropped_indexes(messages, suspects, baseline, render, budget, blank_text);
    }

    *budget -= 1;
    let Some(rendered) = render(&probed) else {
        return HashSet::new();
    };
    let folded = rendered.to_lowercase();
    let missing = tokens
        .into_iter()
        .filter(|(_, token)| {
            let needle = token.to_lowercase();
            !folded.contains(&needle)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return HashSet::new();
    }
    dropped_indexes(messages, &missing, baseline, render, budget, blank_text)
}

/// Indexes whose transformed copy leaves `baseline` unchanged.
///
/// A group that still changes the prompt is split while `budget` lasts.
/// Whatever is left when the budget is spent is kept.
fn dropped_indexes<'a>(
    messages: &[TemplateMessage<'a>],
    suspects: &[usize],
    baseline: &str,
    render: &mut dyn FnMut(&[TemplateMessage<'a>]) -> Option<String>,
    budget: &mut u32,
    transform: fn(&mut TemplateContent),
) -> HashSet<usize> {
    if suspects.is_empty() || *budget == 0 {
        return HashSet::new();
    }
    *budget -= 1;
    let mut altered = messages.to_vec();
    for index in suspects {
        transform(&mut altered[*index].content);
    }
    let Some(alternate) = render(&altered) else {
        return HashSet::new();
    };
    if alternate == baseline {
        return suspects.iter().copied().collect();
    }
    if suspects.len() == 1 {
        return HashSet::new();
    }
    let mid = suspects.len() / 2;
    let mut dropped = dropped_indexes(
        messages,
        &suspects[..mid],
        baseline,
        render,
        budget,
        transform,
    );
    dropped.extend(dropped_indexes(
        messages,
        &suspects[mid..],
        baseline,
        render,
        budget,
        transform,
    ));
    dropped
}
