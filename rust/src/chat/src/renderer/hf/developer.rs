// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `developer` messages for chat templates that do not know the role.
//!
//! OpenAI defines `developer` as the new name of `system`, and clients such as
//! Codex send their instructions as a leading developer message and add more
//! mid-conversation. Many templates know only `system`, and some of those
//! render `system` only as the first message. Developer messages stay where
//! the caller put them, merged at most with an adjacent turn, so the
//! conversation keeps its order: media placeholders stay aligned with their
//! data, and the prompt prefix stays stable as later turns are appended.

use std::borrow::Cow;

use super::TemplateContent;
use super::error::TemplateError;
use super::format::{ProbeContext, probe_message, probe_tool};
use super::template::TemplateContext;
use crate::request::{ChatContent, ChatMessage};

/// How developer messages reach a chat template, decided once per template.
///
/// Unless the template names `developer`, developer messages in the leading
/// run of system messages are merged with it into one `system` message:
/// templates that take only one system message read it from
/// `messages[0]`, and merging adjacent messages moves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeveloperRole {
    /// The template names the `developer` role: messages pass unchanged.
    Native,
    /// The template renders a `system` message after other turns: later
    /// developer messages become `system` messages in place.
    System,
    /// The template renders `system` only at the beginning: a later developer
    /// message joins the adjacent `user` turn, the following one if any, so
    /// templates that require user and assistant turns to alternate accept
    /// it.
    LeadingSystem,
}

/// Probe texts; each is checked for in the rendered prompt.
const PROBE_SYSTEM: &str = "PROBE_SYSTEM";
const PROBE_USER: &str = "PROBE_USER";
const PROBE_ASSISTANT: &str = "PROBE_ASSISTANT";
const PROBE_LATER_SYSTEM: &str = "PROBE_LATER_SYSTEM";
const PROBE_USER_2: &str = "PROBE_USER_2";

impl DeveloperRole {
    /// Decide how `template` takes developer messages, rendering a probe
    /// conversation with `render` if it does not name the role.
    ///
    /// Like Python vLLM, a template supports `developer` when its source
    /// names the role as a string literal. A render probe cannot tell this,
    /// since templates with a generic role branch render any role.
    pub(super) fn probe(
        template: &str,
        render: &impl Fn(TemplateContext<'_>) -> Result<String, TemplateError>,
        ctx: &ProbeContext<'_>,
    ) -> Self {
        if template.contains("\"developer\"") || template.contains("'developer'") {
            return Self::Native;
        }

        // A system message after an assistant turn, as clients add context
        // mid-conversation. Some templates accept one only after a user turn.
        let texts = [
            ("system", PROBE_SYSTEM),
            ("user", PROBE_USER),
            ("assistant", PROBE_ASSISTANT),
            ("system", PROBE_LATER_SYSTEM),
            ("user", PROBE_USER_2),
        ];
        let messages = texts
            .map(|(role, text)| probe_message(role, TemplateContent::String(text.to_string())));
        let renders_later_system = |tools| {
            render(TemplateContext {
                messages: &messages,
                add_generation_prompt: true,
                continue_final_message: false,
                tools,
                documents: None,
                template_kwargs: ctx.template_kwargs,
                special_tokens: ctx.special_tokens,
            })
            .is_ok_and(|prompt| texts.iter().all(|(_, text)| prompt.contains(text)))
        };
        // A `tool_use` template may render nothing without tools.
        if renders_later_system(None) || renders_later_system(Some(&[probe_tool()])) {
            Self::System
        } else {
            Self::LeadingSystem
        }
    }

    /// Lower the developer messages in `messages`, borrowing them unchanged
    /// if there is nothing to lower. The caller's own system messages are left
    /// to the template.
    pub(super) fn lower(self, messages: &[ChatMessage]) -> Cow<'_, [ChatMessage]> {
        let is_developer = |message: &ChatMessage| matches!(message, ChatMessage::Developer { .. });
        if self == Self::Native || !messages.iter().any(is_developer) {
            return Cow::Borrowed(messages);
        }

        let leading_run = messages
            .iter()
            .take_while(|message| {
                matches!(
                    message,
                    ChatMessage::System { .. } | ChatMessage::Developer { .. }
                )
            })
            .count();
        let (leading, rest) = messages.split_at(leading_run);
        let mut lowered = Vec::with_capacity(messages.len());
        if leading.iter().any(is_developer) {
            let mut content = ChatContent::Text(String::new());
            for message in leading {
                if let ChatMessage::System { content: next }
                | ChatMessage::Developer { content: next, .. } = message
                {
                    content.append(next.clone());
                }
            }
            lowered.push(ChatMessage::System { content });
        } else {
            lowered.extend_from_slice(leading);
        }

        // Developer content waiting for the next user turn.
        let mut pending: Option<ChatContent> = None;
        for message in rest {
            match message {
                ChatMessage::Developer { content, .. } => match self {
                    Self::LeadingSystem => match &mut pending {
                        Some(pending) => pending.append(content.clone()),
                        None => pending = Some(content.clone()),
                    },
                    _ => lowered.push(ChatMessage::System {
                        content: content.clone(),
                    }),
                },
                ChatMessage::User { content } if pending.is_some() => {
                    let mut merged = pending.take().expect("checked above");
                    merged.append(content.clone());
                    lowered.push(ChatMessage::User { content: merged });
                }
                message => {
                    if let Some(pending) = pending.take() {
                        join_previous_user(&mut lowered, pending);
                    }
                    lowered.push(message.clone());
                }
            }
        }
        if let Some(pending) = pending {
            join_previous_user(&mut lowered, pending);
        }
        Cow::Owned(lowered)
    }
}

/// Append developer content that no user turn follows to the previous user
/// turn, or add it as a user turn if the previous turn is not one.
fn join_previous_user(lowered: &mut Vec<ChatMessage>, developer: ChatContent) {
    match lowered.last_mut() {
        Some(ChatMessage::User { content }) => content.append(developer),
        _ => lowered.push(ChatMessage::User { content: developer }),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use expect_test::expect;
    use vllm_text::backend::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::DeveloperRole;
    use crate::renderer::hf::format::{ChatTemplateContentFormatOption, ProbeContext};
    use crate::renderer::hf::template::CompiledChatTemplate;

    /// Like Llama 2's and Mistral 7B's templates: user and assistant turns
    /// alternate after an optional first system message.
    pub(crate) const ALTERNATING_TURNS: &str = "{%- if messages[0].role == 'system' %}{%- set loop_messages = messages[1:] %}{{ messages[0].role }}={{ messages[0].content }};{%- else %}{%- set loop_messages = messages %}{%- endif %}{%- for m in loop_messages %}{%- if (m.role == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif %}{{ m.role }}={{ m.content }};{%- endfor %}";

    /// Compile `template`, probing it with generic special tokens.
    fn developer_role(template: &str) -> DeveloperRole {
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            ..Default::default()
        };
        let ctx = ProbeContext {
            special_tokens: Some(&special_tokens),
            template_kwargs: None,
        };
        CompiledChatTemplate::new(
            template.to_string(),
            ChatTemplateContentFormatOption::String,
            None,
            &ctx,
        )
        .expect("template should compile")
        .developer_role()
    }

    #[test]
    fn probes_how_templates_take_developer_messages() {
        let cases = [
            // gpt-oss: names the role.
            (
                "names_developer",
                "{%- for m in messages %}{%- if m.role == 'developer' %}<dev>{%- endif %}{{ m.content }}{%- endfor %}",
            ),
            // Qwen3, Qwen3-Coder: any role, anywhere.
            (
                "any_role",
                "{%- for m in messages %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            ),
            // Llama 3: renders only with `bos_token`.
            (
                "needs_bos_token",
                "{%- for m in messages %}{{ bos_token + '<' + m.role + '>' + m.content }}{%- endfor %}",
            ),
            // A `tool_use` template: renders only with tools.
            (
                "needs_tools",
                "{%- if not tools %}{{ raise_exception('tools required') }}{%- endif %}{%- for m in messages %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            ),
            // Qwen3.5: raises on a non-leading system message.
            (
                "leading_system_raises",
                "{%- for m in messages %}{%- if m.role == 'system' and not loop.first %}{{ raise_exception('System message must be at the beginning.') }}{%- endif %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            ),
            // Ministral 3: a system message may follow a user turn, not an
            // assistant turn.
            (
                "system_after_user_only",
                "{%- for m in messages %}{%- if m.role == 'system' and loop.previtem is defined and loop.previtem.role == 'assistant' %}{{ raise_exception('Unexpected role system after role assistant') }}{%- endif %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            ),
            // Llama 2: user and assistant turns alternate after an optional
            // first system message.
            ("alternating_turns", ALTERNATING_TURNS),
            // Renders only the first message as system and skips later ones.
            (
                "leading_system_only",
                "{%- if messages[0].role == 'system' %}<system>{{ messages[0].content }}{%- endif %}{%- for m in messages %}{%- if m.role != 'system' %}<{{ m.role }}>{{ m.content }}{%- endif %}{%- endfor %}",
            ),
        ];
        let summary = cases
            .iter()
            .map(|(name, template)| format!("{name}: {:?}", developer_role(template)))
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            names_developer: Native
            any_role: System
            needs_bos_token: System
            needs_tools: System
            leading_system_raises: LeadingSystem
            system_after_user_only: LeadingSystem
            alternating_turns: LeadingSystem
            leading_system_only: LeadingSystem"#]]
        .assert_eq(&summary);
    }
}
