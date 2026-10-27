// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Content format of the messages handed to a chat template.
//!
//! In `auto` mode, the format is chosen per request from what the template
//! renders correctly, found once per template by rendering small probe
//! conversations through it. Rendering observes macros, aliases, and
//! `content is string` branches directly, which inspecting the template source
//! has to reconstruct.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde_json::Value as JsonValue;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use tracing::{debug, warn};
use vllm_text::backend::hf::HfSpecialTokens;

use super::error::TemplateError;
use super::template::TemplateContext;
use super::{
    TemplateContent, TemplateContentPart, TemplateMessage, TemplateTool, TemplateToolCall,
    TemplateToolDefinition, TemplateToolFunction,
};
use crate::request::{ChatContent, ChatContentPart, ChatMessage};

/// Chat template content format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
}

/// Configurable chat-template content format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay)]
pub enum ChatTemplateContentFormatOption {
    /// Choose the format per request from what the template renders
    /// correctly.
    #[default]
    Auto,
    /// Always flatten content into plain strings before rendering.
    String,
    /// Always pass content through in OpenAI-compatible structured form.
    OpenAi,
}

impl ChatTemplateContentFormatOption {
    pub const AUTO_LITERAL: &str = "auto";
    pub const OPENAI_LITERAL: &str = "openai";
    pub const STRING_LITERAL: &str = "string";
}

impl FromStr for ChatTemplateContentFormatOption {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::STRING_LITERAL) {
            Ok(Self::String)
        } else if value.eq_ignore_ascii_case(Self::OPENAI_LITERAL) {
            Ok(Self::OpenAi)
        } else {
            Err(format!(
                "invalid content format `{value}`; expected one of: {}, {}, {}",
                Self::AUTO_LITERAL,
                Self::STRING_LITERAL,
                Self::OPENAI_LITERAL
            ))
        }
    }
}

impl fmt::Display for ChatTemplateContentFormatOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::String => f.write_str(Self::STRING_LITERAL),
            Self::OpenAi => f.write_str(Self::OPENAI_LITERAL),
        }
    }
}

/// Renderer inputs that every probe renders with: the same for all requests.
#[derive(Default)]
pub(super) struct ProbeContext<'a> {
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    pub(super) template_kwargs: Option<&'a HashMap<String, JsonValue>>,
}

/// Content shapes a chat template renders correctly, found by rendering probe
/// conversations through it.
///
/// A probe passes when the render succeeds, contains every text it was given,
/// and does not print a content-part list into the prompt. The `openai` format
/// turns every role's content into parts, and many templates loop over parts
/// only in user messages, so text parts are probed per role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ContentCapabilities {
    /// A user message with plain-string content.
    string: bool,
    /// Text parts in a message of each role.
    typed_user: bool,
    typed_system: bool,
    typed_assistant: bool,
    typed_tool: bool,
    /// Media parts per modality, rendered as something: the prompt differs
    /// from the same message without the part. A media-only template, such as
    /// an ASR model's, may drop the message's text parts.
    typed_image: bool,
    typed_video: bool,
    typed_audio: bool,
}

/// Probe texts; each is checked for in the rendered prompt.
const PROBE_STRING: &str = "PROBE_STRING";
const PROBE_USER: &str = "PROBE_USER";
const PROBE_USER_2: &str = "PROBE_USER_2";
const PROBE_SYSTEM: &str = "PROBE_SYSTEM";
const PROBE_ASSISTANT: &str = "PROBE_ASSISTANT";
const PROBE_TOOL: &str = "PROBE_TOOL";

impl ContentCapabilities {
    /// Render the probe conversations with `render`, which renders the
    /// template being probed.
    ///
    /// A template that renders nothing without tools, such as a `tool_use`
    /// template, is probed again with one tool.
    pub(super) fn probe(
        render: &impl Fn(TemplateContext<'_>) -> Result<String, TemplateError>,
        ctx: &ProbeContext<'_>,
    ) -> Self {
        let mut capabilities = Self::probe_with_tools(render, ctx, None);
        if !capabilities.renders_any() {
            capabilities = Self::probe_with_tools(render, ctx, Some(&[probe_tool()]));
        }
        if capabilities.renders_any() {
            debug!(?capabilities, "probed chat template content capabilities");
        } else {
            warn!(
                "chat template renders none of the content probes; using string content. \
                 Set `--chat-template-content-format` if this is wrong."
            );
        }
        capabilities
    }

    /// Render the probe conversations with `tools`; a tool round trip is
    /// always rendered with a tool, since a request carrying one has tools.
    fn probe_with_tools(
        render_template: &impl Fn(TemplateContext<'_>) -> Result<String, TemplateError>,
        ctx: &ProbeContext<'_>,
        tools: Option<&[TemplateTool]>,
    ) -> Self {
        let render = |messages: &[TemplateMessage<'_>], tools: Option<&[TemplateTool]>| {
            render_template(TemplateContext {
                messages,
                add_generation_prompt: true,
                continue_final_message: false,
                tools,
                documents: None,
                template_kwargs: ctx.template_kwargs,
                special_tokens: ctx.special_tokens,
            })
            .ok()
        };
        let passes = |messages: &[TemplateMessage<'_>], expected: &[&str]| {
            render(messages, tools).is_some_and(|prompt| renders_texts(&prompt, expected))
        };

        let probe_tool = probe_tool();
        let tool_round_trip = [
            probe_message("user", text_parts(&[PROBE_USER])),
            TemplateMessage {
                tool_calls: Some(vec![TemplateToolCall {
                    // 9 alphanumeric characters: Mistral templates reject other IDs.
                    id: "call00000".to_string(),
                    r#type: "function",
                    function: TemplateToolFunction {
                        name: probe_tool.function.name.clone(),
                        arguments: JsonValue::Object(Default::default()),
                    },
                }]),
                ..probe_message("assistant", text_parts(&[""]))
            },
            TemplateMessage {
                tool_call_id: Some("call00000".to_string()),
                ..probe_message("tool", text_parts(&[PROBE_TOOL]))
            },
        ];
        let typed_tool = render(&tool_round_trip, Some(std::slice::from_ref(&probe_tool)))
            .is_some_and(|prompt| renders_texts(&prompt, &[PROBE_USER, PROBE_TOOL]));

        let text_only = [probe_message(
            "user",
            text_parts(&[PROBE_USER, PROBE_USER_2]),
        )];
        let text_only_prompt = render(&text_only, tools);
        let renders_media = |part: TemplateContentPart| {
            let Some(text_only_prompt) = &text_only_prompt else {
                return false;
            };
            let with_media = [probe_message(
                "user",
                TemplateContent::OpenAi(vec![
                    TemplateContentPart::Text {
                        text: PROBE_USER.to_string(),
                    },
                    part,
                    TemplateContentPart::Text {
                        text: PROBE_USER_2.to_string(),
                    },
                ]),
            )];
            let expected: &[&str] = if renders_texts(text_only_prompt, &[PROBE_USER, PROBE_USER_2])
            {
                &[PROBE_USER, PROBE_USER_2]
            } else {
                &[]
            };
            render(&with_media, tools).is_some_and(|prompt| {
                renders_texts(&prompt, expected) && prompt != *text_only_prompt
            })
        };

        Self {
            string: passes(
                &[probe_message(
                    "user",
                    TemplateContent::String(PROBE_STRING.to_string()),
                )],
                &[PROBE_STRING],
            ),
            typed_user: passes(
                &[probe_message("user", text_parts(&[PROBE_USER]))],
                &[PROBE_USER],
            ),
            typed_system: passes(
                &[
                    probe_message("system", text_parts(&[PROBE_SYSTEM])),
                    probe_message("user", text_parts(&[PROBE_USER])),
                ],
                &[PROBE_SYSTEM, PROBE_USER],
            ),
            typed_assistant: passes(
                &[
                    probe_message("user", text_parts(&[PROBE_USER])),
                    probe_message("assistant", text_parts(&[PROBE_ASSISTANT])),
                    probe_message("user", text_parts(&[PROBE_USER_2])),
                ],
                &[PROBE_USER, PROBE_ASSISTANT, PROBE_USER_2],
            ),
            typed_tool,
            typed_image: renders_media(TemplateContentPart::Image),
            typed_video: renders_media(TemplateContentPart::Video),
            typed_audio: renders_media(TemplateContentPart::Audio),
        }
    }

    fn renders_any(&self) -> bool {
        self.string || self.typed_user || self.typed_image || self.typed_video || self.typed_audio
    }

    /// Choose the content format for one request.
    ///
    /// Parts are used when the template renders every message of the request
    /// as parts; otherwise plain strings, unless the template renders only
    /// parts.
    pub(super) fn content_format_for(&self, messages: &[ChatMessage]) -> ChatTemplateContentFormat {
        if messages.iter().all(|message| self.renders_parts(message)) {
            ChatTemplateContentFormat::OpenAi
        } else if self.string || !self.typed_user {
            ChatTemplateContentFormat::String
        } else {
            ChatTemplateContentFormat::OpenAi
        }
    }

    fn renders_parts(&self, message: &ChatMessage) -> bool {
        match message {
            // Media parts were probed in a user message, where a media-only
            // template may drop the text parts beside them.
            ChatMessage::User { content } => match self.renders_media(content) {
                Some(renders_media) => renders_media,
                None => self.typed_user,
            },
            ChatMessage::System { content } => {
                self.typed_system && self.renders_media(content) != Some(false)
            }
            ChatMessage::Assistant { .. } => self.typed_assistant,
            ChatMessage::ToolResponse { content, .. } => {
                self.typed_tool && self.renders_media(content) != Some(false)
            }
            // Not probed: how the template handles these roles is unknown.
            ChatMessage::Developer { .. } | ChatMessage::Custom { .. } => false,
        }
    }

    /// Whether the template renders every media part of `content`; `None` if
    /// it has none.
    fn renders_media(&self, content: &ChatContent) -> Option<bool> {
        let ChatContent::Parts(parts) = content else {
            return None;
        };
        let mut media = parts
            .iter()
            .filter_map(|part| match part {
                ChatContentPart::Text { .. } => None,
                ChatContentPart::ImageUrl { .. } => Some(self.typed_image),
                ChatContentPart::VideoUrl { .. } => Some(self.typed_video),
                ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                    Some(self.typed_audio)
                }
            })
            .peekable();
        media.peek().is_some().then(|| media.all(|renders| renders))
    }
}

fn probe_message(role: &'static str, content: TemplateContent) -> TemplateMessage<'static> {
    TemplateMessage {
        role,
        content,
        tools: None,
        reasoning: None,
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn text_parts(texts: &[&str]) -> TemplateContent {
    TemplateContent::OpenAi(
        texts
            .iter()
            .map(|text| TemplateContentPart::Text {
                text: text.to_string(),
            })
            .collect(),
    )
}

/// The tool a probe declares when it renders with tools.
fn probe_tool() -> TemplateTool {
    TemplateTool {
        tool_type: "function",
        function: TemplateToolDefinition {
            name: "get_weather".to_string(),
            description: Some("Get the weather for a city.".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
            strict: None,
            defer_loading: None,
        },
    }
}

/// Whether `prompt` contains every expected text and no content-part list,
/// which MiniJinja prints as `[{"type": "text", "text": ...}]`.
fn renders_texts(prompt: &str, expected: &[&str]) -> bool {
    const PART_TYPES: [&str; 4] = [
        r#""type": "text""#,
        r#""type": "image""#,
        r#""type": "video""#,
        r#""type": "audio""#,
    ];
    expected.iter().all(|text| prompt.contains(text))
        && !PART_TYPES.iter().any(|part| prompt.contains(part))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use vllm_text::backend::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::{
        ChatTemplateContentFormat, ChatTemplateContentFormatOption, ContentCapabilities,
        ProbeContext,
    };
    use crate::renderer::hf::template::CompiledChatTemplate;
    use crate::request::{ChatContentPart, ChatMessage};

    /// Probe `template` with generic special tokens, as a model's tokenizer
    /// config provides them.
    fn probe(template: &str) -> ContentCapabilities {
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: Some(NamedSpecialToken::Text("</s>".to_string())),
            ..Default::default()
        };
        let ctx = ProbeContext {
            special_tokens: Some(&special_tokens),
            template_kwargs: None,
        };
        let template = CompiledChatTemplate::new(
            template.to_string(),
            ChatTemplateContentFormatOption::String,
            None,
            &ctx,
        )
        .expect("template should compile");
        ContentCapabilities::probe(&|ctx| template.apply(ctx), &ctx)
    }

    /// Whether strings render, then the roles and media modalities whose parts
    /// render.
    fn summary(capabilities: ContentCapabilities) -> String {
        let ContentCapabilities {
            string,
            typed_user,
            typed_system,
            typed_assistant,
            typed_tool,
            typed_image,
            typed_video,
            typed_audio,
        } = capabilities;
        let names = |flags: &[(bool, &str)]| {
            flags
                .iter()
                .filter(|(flag, _)| *flag)
                .map(|(_, name)| *name)
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "string={:5} parts=[{}] media=[{}]",
            string,
            names(&[
                (typed_user, "user"),
                (typed_system, "system"),
                (typed_assistant, "assistant"),
                (typed_tool, "tool"),
            ]),
            names(&[
                (typed_image, "image"),
                (typed_video, "video"),
                (typed_audio, "audio"),
            ]),
        )
    }

    fn format_name(format: ChatTemplateContentFormat) -> &'static str {
        match format {
            ChatTemplateContentFormat::String => "string",
            ChatTemplateContentFormat::OpenAi => "openai",
        }
    }

    fn image() -> ChatContentPart {
        ChatContentPart::image_url("https://example.com/a.png")
    }

    fn audio() -> ChatContentPart {
        ChatContentPart::audio_url("https://example.com/a.wav")
    }

    fn user_text_parts() -> ChatMessage {
        ChatMessage::user(vec![ChatContentPart::text("a"), ChatContentPart::text("b")])
    }

    #[test]
    fn content_format_per_capabilities_and_request() {
        let none = ContentCapabilities {
            string: false,
            typed_user: false,
            typed_system: false,
            typed_assistant: false,
            typed_tool: false,
            typed_image: false,
            typed_video: false,
            typed_audio: false,
        };
        let capabilities = [
            (
                "all",
                ContentCapabilities {
                    string: true,
                    typed_user: true,
                    typed_system: true,
                    typed_assistant: true,
                    typed_tool: true,
                    typed_image: true,
                    typed_video: true,
                    typed_audio: true,
                },
            ),
            (
                "string",
                ContentCapabilities {
                    string: true,
                    ..none
                },
            ),
            (
                "text_parts",
                ContentCapabilities {
                    string: true,
                    typed_user: true,
                    typed_system: true,
                    typed_assistant: true,
                    typed_tool: true,
                    ..none
                },
            ),
            (
                "user_parts",
                ContentCapabilities {
                    string: true,
                    typed_user: true,
                    typed_image: true,
                    ..none
                },
            ),
            (
                "parts_only",
                ContentCapabilities {
                    typed_user: true,
                    typed_system: true,
                    typed_image: true,
                    ..none
                },
            ),
            (
                "audio_only",
                ContentCapabilities {
                    typed_audio: true,
                    ..none
                },
            ),
            ("none", none),
        ];
        let user = || ChatMessage::user("u");
        let requests = [
            ("user", vec![user()]),
            ("user_text_parts", vec![user_text_parts()]),
            (
                "user_image",
                vec![ChatMessage::user(vec![image(), ChatContentPart::text("a")])],
            ),
            (
                "user_audio",
                vec![ChatMessage::user(vec![audio(), ChatContentPart::text("a")])],
            ),
            (
                "user_image_audio",
                vec![ChatMessage::user(vec![image(), audio()])],
            ),
            ("system_user", vec![ChatMessage::system("s"), user()]),
            (
                "system_image_user",
                vec![ChatMessage::system(vec![image()]), user()],
            ),
            (
                "assistant",
                vec![user(), ChatMessage::assistant_text("a"), user()],
            ),
            (
                "tool",
                vec![user(), ChatMessage::tool_response("t", "call00000")],
            ),
            (
                "tool_image",
                vec![
                    user(),
                    ChatMessage::tool_response(vec![image()], "call00000"),
                ],
            ),
            (
                "developer_user",
                vec![ChatMessage::developer("d", None), user()],
            ),
        ];

        let header = capabilities.iter().map(|(name, _)| format!(" {name:11}"));
        let rows = requests.iter().map(|(request, messages)| {
            let formats = capabilities.iter().map(|(_, capabilities)| {
                format!(
                    " {:11}",
                    format_name(capabilities.content_format_for(messages))
                )
            });
            format!("{request:18}{}", formats.collect::<String>())
        });
        let table = std::iter::once(format!("{:18}{}", "", header.collect::<String>()))
            .chain(rows)
            .map(|line| line.trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
                               all         string      text_parts  user_parts  parts_only  audio_only  none
            user               openai      string      openai      openai      openai      string      string
            user_text_parts    openai      string      openai      openai      openai      string      string
            user_image         openai      string      string      openai      openai      string      string
            user_audio         openai      string      string      string      openai      openai      string
            user_image_audio   openai      string      string      string      openai      string      string
            system_user        openai      string      openai      string      openai      string      string
            system_image_user  openai      string      string      string      openai      string      string
            assistant          openai      string      openai      string      openai      string      string
            tool               openai      string      openai      string      openai      string      string
            tool_image         openai      string      string      string      openai      string      string
            developer_user     string      string      string      string      openai      string      string"#]].assert_eq(&table);
    }

    /// Chat template constructs that decide the content format, each distilled
    /// from a public model's template, with a request that exercises it.
    #[test]
    fn probes_chat_template_constructs() {
        // Every role's content may be parts.
        const PARTS_EVERY_ROLE: &str = "{%- for m in messages %}<{{ m.role }}>{%- if m.content is string %}{{ m.content }}{%- else %}{%- for p in m.content %}{%- if p.type == 'image' %}<image>{%- elif p.type == 'video' %}<video>{%- elif p.type == 'text' %}{{ p.text }}{%- endif %}{%- endfor %}{%- endif %}{%- endfor %}";
        // Only user content may be parts; other roles print it.
        const PARTS_IN_USER: &str = "{%- for m in messages %}<{{ m.role }}>{%- if m.role == 'user' and m.content is not string %}{%- for p in m.content %}{%- if p.type == 'image' %}<image>{%- elif p.type == 'text' %}{{ p.text }}{%- endif %}{%- endfor %}{%- else %}{{ m.content }}{%- endif %}{%- endfor %}";
        // Parts are looped over without a string branch.
        const PARTS_ONLY: &str = "{%- for m in messages %}<{{ m.role }}>{%- for p in m.content %}{%- if p.type == 'image' %}<image>{%- else %}{{ p.text }}{%- endif %}{%- endfor %}{%- endfor %}";
        let cases: Vec<(&str, &str, Vec<ChatMessage>)> = vec![
            // Qwen3: content is printed.
            (
                "strings",
                "{%- for m in messages %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
                vec![user_text_parts()],
            ),
            // Qwen3.5.
            (
                "parts_every_role",
                PARTS_EVERY_ROLE,
                vec![ChatMessage::system("s"), ChatMessage::user(vec![image()])],
            ),
            // GLM-4.5: tool results are printed.
            (
                "tool_content_printed",
                "{%- for m in messages %}{%- if m.role == 'tool' %}<tool_response>{{ m.content }}</tool_response>{%- elif m.content is string %}{{ m.content }}{%- else %}{%- for p in m.content %}{%- if p.type == 'text' %}{{ p.text }}{%- endif %}{%- endfor %}{%- endif %}{%- endfor %}",
                vec![
                    ChatMessage::user("u"),
                    ChatMessage::tool_response("t", "call00000"),
                ],
            ),
            // InternVL2.5: system content is printed.
            (
                "system_content_printed",
                PARTS_IN_USER,
                vec![ChatMessage::system("s"), ChatMessage::user(vec![image()])],
            ),
            // GLM-4.1V: assistant content is printed.
            (
                "assistant_content_printed",
                PARTS_IN_USER,
                vec![
                    ChatMessage::user(vec![image()]),
                    ChatMessage::assistant_text("a"),
                    ChatMessage::user("u"),
                ],
            ),
            // Nemotron-Nano-12B-v2-VL: audio parts are skipped.
            (
                "audio_skipped_image",
                PARTS_EVERY_ROLE,
                vec![ChatMessage::user(vec![image()])],
            ),
            (
                "audio_skipped_audio",
                PARTS_EVERY_ROLE,
                vec![ChatMessage::user(vec![audio()])],
            ),
            // LLaVA-1.5.
            ("parts_only", PARTS_ONLY, vec![ChatMessage::user("u")]),
            (
                "parts_only_unprobed_role",
                PARTS_ONLY,
                vec![ChatMessage::developer("d", None), ChatMessage::user("u")],
            ),
            // Qwen3-ASR: text is dropped, audio becomes audio tokens.
            (
                "audio_only",
                "{%- for m in messages %}{%- if m.content is not string %}{%- for p in m.content %}{%- if p.type == 'audio' %}<|audio|>{%- endif %}{%- endfor %}{%- endif %}{%- endfor %}",
                vec![ChatMessage::user(vec![audio(), ChatContentPart::text("a")])],
            ),
            // Hermes 2 Pro `tool_use`: renders only with tools.
            (
                "tools_required",
                "{%- if not tools %}{{ raise_exception('tools required') }}{%- endif %}{%- for m in messages %}{{ m.content }}{%- endfor %}",
                vec![ChatMessage::user("u")],
            ),
            // Llama 3: `bos_token` is concatenated with the content.
            (
                "bos_token_concatenated",
                "{%- for m in messages %}{%- set content = m.content %}{%- if loop.first %}{%- set content = bos_token + content %}{%- endif %}{{ content }}{%- endfor %}",
                vec![ChatMessage::user("u")],
            ),
            // A part-rendering macro that no message reaches.
            (
                "unused_part_macro",
                "{%- macro render(content) %}{%- for p in content %}{{ p.text }}{%- endfor %}{%- endmacro %}{%- for m in messages %}{{ m.role + ': ' + m.content }}{%- endfor %}",
                vec![user_text_parts()],
            ),
            // MiMo-V2.5: a macro renders the content passed to it.
            (
                "macro_bound_to_content",
                "{%- macro render_content(message_content) %}{%- if message_content is string %}{{ message_content }}{%- else %}{%- for content in message_content %}{%- if content.type == 'image' %}<image>{%- else %}{{ content.text }}{%- endif %}{%- endfor %}{%- endif %}{%- endmacro %}{%- for m in messages %}{{ render_content(m.content) }}{%- endfor %}",
                vec![ChatMessage::user(vec![image()])],
            ),
            // TranslateGemma: only its own input format renders.
            (
                "custom_input_format",
                "{%- for m in messages %}{%- if m.content is string or m.content[0].source_lang_code is not defined %}{{ raise_exception('expected source_lang_code') }}{%- endif %}{{ m.content[0].text }}{%- endfor %}",
                vec![ChatMessage::user("u")],
            ),
        ];

        let snapshot = cases
            .iter()
            .map(|(name, template, messages)| {
                let capabilities = probe(template);
                format!(
                    "{name:26} {:6} {}",
                    format_name(capabilities.content_format_for(messages)),
                    summary(capabilities),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            strings                    string string=true  parts=[] media=[]
            parts_every_role           openai string=true  parts=[user,system,assistant,tool] media=[image,video]
            tool_content_printed       string string=true  parts=[user,system,assistant] media=[]
            system_content_printed     string string=true  parts=[user] media=[image]
            assistant_content_printed  string string=true  parts=[user] media=[image]
            audio_skipped_image        openai string=true  parts=[user,system,assistant,tool] media=[image,video]
            audio_skipped_audio        string string=true  parts=[user,system,assistant,tool] media=[image,video]
            parts_only                 openai string=false parts=[user,system,assistant,tool] media=[image]
            parts_only_unprobed_role   openai string=false parts=[user,system,assistant,tool] media=[image]
            audio_only                 openai string=false parts=[] media=[audio]
            tools_required             string string=true  parts=[] media=[]
            bos_token_concatenated     string string=true  parts=[] media=[]
            unused_part_macro          string string=true  parts=[] media=[]
            macro_bound_to_content     openai string=true  parts=[user,system,assistant,tool] media=[image]
            custom_input_format        string string=false parts=[] media=[]"#]].assert_eq(&snapshot);
    }
}
