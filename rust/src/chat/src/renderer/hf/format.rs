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

use super::template::{CompiledChatTemplate, TemplateContext};
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
pub(super) struct ProbeContext<'a> {
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    pub(super) template_kwargs: &'a HashMap<String, JsonValue>,
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
    /// Render the probe conversations through `template`.
    ///
    /// A template that renders nothing without tools, such as a `tool_use`
    /// template, is probed again with one tool.
    pub(super) fn probe(template: &CompiledChatTemplate, ctx: &ProbeContext<'_>) -> Self {
        let mut capabilities = Self::probe_with_tools(template, ctx, None);
        if !capabilities.renders_any() {
            capabilities = Self::probe_with_tools(template, ctx, Some(&[probe_tool()]));
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
        template: &CompiledChatTemplate,
        ctx: &ProbeContext<'_>,
        tools: Option<&[TemplateTool]>,
    ) -> Self {
        let render = |messages: &[TemplateMessage<'_>], tools: Option<&[TemplateTool]>| {
            template
                .apply(TemplateContext {
                    messages,
                    add_generation_prompt: true,
                    continue_final_message: false,
                    tools,
                    documents: None,
                    template_kwargs: Some(ctx.template_kwargs),
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
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use expect_test::expect;

    use super::{
        ChatTemplateContentFormatOption, CompiledChatTemplate, ContentCapabilities, ProbeContext,
    };

    fn probe(template: &str) -> ContentCapabilities {
        let template =
            CompiledChatTemplate::new(template.to_string(), ChatTemplateContentFormatOption::Auto)
                .expect("template should compile");
        ContentCapabilities::probe(
            &template,
            &ProbeContext {
                special_tokens: None,
                template_kwargs: &HashMap::new(),
            },
        )
    }

    /// One line per template: whether strings render, then the roles and
    /// media modalities whose parts render.
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

    fn template_paths() -> Vec<PathBuf> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/templates");
        let mut paths = [dir.clone(), dir.join("vllm_examples")]
            .iter()
            .flat_map(|dir| fs::read_dir(dir).expect("template directory should exist"))
            .map(|entry| entry.expect("template directory entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jinja"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    #[test]
    fn probes_test_and_vllm_example_templates() {
        let snapshot = template_paths()
            .iter()
            .map(|path| {
                let template = fs::read_to_string(path).expect("template should be readable");
                let name = path
                    .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/templates"))
                    .expect("template path under tests/templates")
                    .display();
                format!("{name:55} {}", summary(probe(&template)))
            })
            .collect::<Vec<_>>()
            .join("\n");

        expect![[r#"
            qwen3.jinja                                             string=true  parts=[] media=[]
            qwen35.jinja                                            string=true  parts=[user,system,assistant,tool] media=[image,video]
            step5_vl.jinja                                          string=true  parts=[user,system,assistant,tool] media=[image]
            vllm_examples/template_alpaca.jinja                     string=true  parts=[] media=[]
            vllm_examples/template_chatglm.jinja                    string=true  parts=[] media=[]
            vllm_examples/template_chatglm2.jinja                   string=true  parts=[] media=[]
            vllm_examples/template_chatml.jinja                     string=true  parts=[] media=[]
            vllm_examples/template_falcon.jinja                     string=true  parts=[] media=[]
            vllm_examples/template_falcon_180b.jinja                string=true  parts=[] media=[]
            vllm_examples/template_inkbot.jinja                     string=true  parts=[] media=[]
            vllm_examples/template_teleflm.jinja                    string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_deepseekr1.jinja       string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_deepseekv3.jinja       string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_deepseekv31.jinja      string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_functiongemma.jinja    string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_gemma3_pythonic.jinja  string=true  parts=[user,system,assistant,tool] media=[image]
            vllm_examples/tool_chat_template_gemma4.jinja           string=true  parts=[user,assistant,tool] media=[image,video,audio]
            vllm_examples/tool_chat_template_glm4.jinja             string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_granite.jinja          string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_granite_20b_fc.jinja   string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_hermes.jinja           string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_hunyuan_a13b.jinja     string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_internlm2_tool.jinja   string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_llama3.1_json.jinja    string=true  parts=[user,system,assistant,tool] media=[]
            vllm_examples/tool_chat_template_llama3.2_json.jinja    string=true  parts=[user,system,assistant,tool] media=[image]
            vllm_examples/tool_chat_template_llama3.2_pythonic.jinja string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_llama4_json.jinja      string=true  parts=[user,system,assistant,tool] media=[image]
            vllm_examples/tool_chat_template_llama4_pythonic.jinja  string=true  parts=[user,system,assistant,tool] media=[image]
            vllm_examples/tool_chat_template_mistral.jinja          string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_mistral3.jinja         string=true  parts=[user,system] media=[image]
            vllm_examples/tool_chat_template_mistral_parallel.jinja string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_phi4_mini.jinja        string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_qwen3coder.jinja       string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_toolace.jinja          string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_xlam_llama.jinja       string=true  parts=[] media=[]
            vllm_examples/tool_chat_template_xlam_qwen.jinja        string=true  parts=[] media=[]"#]].assert_eq(&snapshot);
    }
}
