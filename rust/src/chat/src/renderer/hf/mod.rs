use std::collections::HashMap;

use serde::Serialize;
use serde_json::Value as JsonValue;
use thiserror_ext::AsReport as _;
use tracing::{info, trace, warn};
use vllm_text::Prompt;
use vllm_text::backend::hf::{
    HfSpecialTokens, HfTokenizerConfig, ResolvedModelFiles, load_tokenizer_config,
};

use self::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption as ContentFormatOption,
};
use self::template::{CompiledChatTemplate, TemplateContext};
use self::value::{TemplateValue, to_template_value};
use super::{ChatRenderer, RenderedPrompt, effective_template_kwargs};
use crate::error::Result;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};
use crate::{
    AssistantContentBlock, AssistantMessageExt, ChatTool, Error, LoadModelBackendsOptions,
};

mod error;
mod format;
mod template;
mod tojson;
mod value;

pub use template::{load_chat_template, resolve_chat_template};

pub use self::format::ChatTemplateContentFormatOption;

/// Template-visible placeholder tokens per supported modality.
///
/// A `None` token means the loaded model does not support that modality, and
/// content parts of that modality are rejected during rendering.
#[derive(Debug, Clone, Default)]
pub struct MultimodalRenderInfo {
    pub image_token: Option<String>,
    pub video_token: Option<String>,
}

/// Hugging Face chat-template renderer backed by the local Jinja chat-template
/// state.
pub struct HfChatRenderer {
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: HashMap<String, JsonValue>,
    content_format: ContentFormatOption,
    special_tokens: Option<HfSpecialTokens>,
    multimodal: Option<MultimodalRenderInfo>,
}

impl HfChatRenderer {
    /// Create a renderer from the given template string.
    pub fn new(
        template: Option<String>,
        default_template_kwargs: HashMap<String, JsonValue>,
        content_format: ContentFormatOption,
    ) -> Result<Self> {
        Ok(Self {
            default_template: template
                .map(|template| {
                    CompiledChatTemplate::new(template, content_format)
                        .map_err(|error| Error::ChatTemplate(error.to_report_string()))
                })
                .transpose()?,
            default_template_kwargs,
            content_format,
            special_tokens: None,
            multimodal: None,
        })
    }

    pub fn with_special_tokens(mut self, special_tokens: Option<HfSpecialTokens>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    pub fn with_multimodal(mut self, multimodal: Option<MultimodalRenderInfo>) -> Self {
        self.multimodal = multimodal;
        self
    }

    /// Create a renderer from the given model files and loading options.
    pub fn load(
        files: &ResolvedModelFiles,
        options: LoadModelBackendsOptions,
        multimodal: Option<MultimodalRenderInfo>,
    ) -> Result<Self> {
        let HfTokenizerConfig {
            special_tokens,
            chat_template,
            ..
        } = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let mut template = chat_template;
        let special_tokens = (!special_tokens.is_empty()).then_some(special_tokens);

        if let Some(configured_template) = options.chat_template.as_deref() {
            template = Some(
                resolve_chat_template(configured_template)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))?,
            );
            info!("using configured chat template override");
        } else if let Some(chat_template_path) = files.chat_template_path.as_deref() {
            // If independent chat template file(s) exist and contain non-empty content,
            // they take priority over template entries in the tokenizer config
            let file_template = load_chat_template(chat_template_path)
                .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

            if file_template.as_ref().is_some_and(|t| !t.trim().is_empty()) {
                info!(
                    path = %chat_template_path.display(),
                    "loaded dedicated chat template file, overriding tokenizer_config chat_template"
                );
                template = file_template;
            } else {
                warn!(
                    path = %chat_template_path.display(),
                    "ignoring empty dedicated chat template file and falling back to tokenizer_config chat_template"
                );
            }
        }

        Ok(Self::new(
            template,
            options.default_chat_template_kwargs,
            options.chat_template_content_format,
        )?
        .with_special_tokens(special_tokens)
        .with_multimodal(multimodal))
    }

    /// Apply the chat template to one chat request, rendering the prompt string
    /// to be tokenized and submitted to the model.
    ///
    /// If the request carries a per-request `chat_template` override, a
    /// temporary template is compiled from that string and used instead of
    /// the model's default.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        let override_template = request
            .chat_options
            .chat_template
            .as_ref()
            .map(|template| {
                CompiledChatTemplate::new(template.clone(), self.content_format)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))
            })
            .transpose()?;
        let template = override_template
            .as_ref()
            .or(self.default_template.as_ref())
            .ok_or(Error::MissingChatTemplate)?;

        self.apply_chat_template_inner(template, request)
    }

    fn apply_chat_template_inner(
        &self,
        effective_template: &CompiledChatTemplate,
        request: &ChatRequest,
    ) -> Result<RenderedPrompt> {
        let mut messages = to_template_messages(
            &request.messages,
            effective_template.content_format(),
            self.multimodal.as_ref(),
        )?;

        // Handling of `continue_final_message`:
        // Append a sentinel tag to the final message content, render as usual, then
        // truncate the rendered prompt at the tag so any template suffix after the
        // final message content (e.g. the end-of-turn marker) is dropped.
        let final_message_text = if request.chat_options.continue_final_message() {
            let final_message = messages.last_mut().ok_or(Error::EmptyMessages)?;
            Some(append_continue_final_message_tag(final_message)?)
        } else {
            None
        };

        let tools = request.tool_parsing_enabled().then(|| to_template_tools(&request.tools));
        trace!(
            message_count = messages.len(),
            content_format = ?effective_template.content_format(),
            ?messages,
            ?tools,
            "applying chat template"
        );

        let effective_template_kwargs =
            effective_template_kwargs(&self.default_template_kwargs, request);
        let prompt = effective_template
            .apply(TemplateContext {
                messages: &messages,
                add_generation_prompt: request.chat_options.add_generation_prompt(),
                continue_final_message: request.chat_options.continue_final_message(),
                tools: tools.as_deref(),
                documents: request.documents.as_deref(),
                template_kwargs: Some(&effective_template_kwargs),
                special_tokens: self.special_tokens.as_ref(),
            })
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

        let prompt = match &final_message_text {
            Some(final_message_text) => {
                truncate_prompt_at_continue_final_message_tag(prompt, final_message_text)?
            }
            None => prompt,
        };

        trace!(
            prompt_len = prompt.len(),
            prompt, "rendered chat template prompt"
        );

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
            effective_template_kwargs,
        })
    }
}

impl ChatRenderer for HfChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        self.apply_chat_template(request)
    }
}

/// Chat message in the JSON shape expected by Jinja chat templates.
// TODO: borrow more fields directly from the original `ChatMessage`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateMessage {
    role: &'static str,
    content: TemplateContent,
    // Developer-role messages may provide message-local tools in the same shape
    // as top-level request tools.
    tools: Option<Vec<TemplateTool>>,
    // Reasoning-capable HF templates are inconsistent on the exact field name,
    // so expose both variants for compatibility.
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    // Function-call-capable templates commonly expect assistant tool calls
    // under this OpenAI-compatible field name.
    tool_calls: Option<Vec<TemplateToolCall>>,
    // Tool-role messages refer back to the assistant call they are answering.
    tool_call_id: Option<String>,
}

/// Chat content in the two shapes HF templates commonly expect.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum TemplateContent {
    String(String),
    OpenAi(Vec<TemplateContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TemplateContentPart {
    Text { text: String },
    Image,
    Video,
}

#[derive(Debug, Serialize)]
struct TemplateToolCall {
    id: String,
    r#type: &'static str, // always "function"
    function: TemplateToolFunction,
}

#[derive(Debug, Serialize)]
struct TemplateToolFunction {
    name: String,
    arguments: TemplateValue,
}

#[derive(Debug, Serialize)]
pub(super) struct TemplateTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: TemplateToolDefinition,
}

#[derive(Debug, Serialize)]
struct TemplateToolDefinition {
    name: String,
    description: Option<String>,
    parameters: TemplateValue,
    strict: Option<bool>,
}

/// Convert chat messages into the JSON shape expected by Jinja chat templates.
fn to_template_messages(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateMessage>> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format, multimodal))
        .collect()
}

fn to_template_message(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateMessage> {
    Ok(match message {
        ChatMessage::System { content } => TemplateMessage {
            role: "system",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Developer { content, tools } => TemplateMessage {
            role: "developer",
            content: to_template_content(content, content_format, multimodal)?,
            tools: tools.as_deref().map(to_template_tools),
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::User { content } => TemplateMessage {
            role: "user",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Assistant { content } => {
            let text = content.text();
            let reasoning = content.reasoning();
            let tool_calls = to_template_tool_calls(content)?;
            let content =
                to_template_content(&ChatContent::Text(text), content_format, multimodal)?;
            TemplateMessage {
                role: "assistant",
                content,
                tools: None,
                reasoning: reasoning.clone(),
                reasoning_content: reasoning,
                tool_calls,
                tool_call_id: None,
            }
        }
        ChatMessage::ToolResponse {
            content,
            tool_call_id,
        } => TemplateMessage {
            role: "tool",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        },
    })
}

fn to_template_tool_calls(
    content: &[AssistantContentBlock],
) -> Result<Option<Vec<TemplateToolCall>>> {
    let mut tool_calls = Vec::new();

    for tool_call in content.tool_calls() {
        let arguments = serde_json::from_str(&tool_call.arguments).map_err(|error| {
            Error::ChatTemplate(format!(
                "assistant tool call `{}` has invalid JSON arguments: {}",
                tool_call.id,
                error.as_report()
            ))
        })?;
        let arguments = to_template_value(arguments);

        tool_calls.push(TemplateToolCall {
            id: tool_call.id.clone(),
            r#type: "function",
            function: TemplateToolFunction {
                name: tool_call.name.clone(),
                arguments,
            },
        });
    }

    Ok((!tool_calls.is_empty()).then_some(tool_calls))
}

fn to_template_content(
    content: &ChatContent,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateContent> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(to_template_string_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::OpenAi => {
            TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
        }
    })
}

fn to_template_openai_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateContentPart>> {
    match content {
        ChatContent::Text(text) => Ok(vec![TemplateContentPart::Text { text: text.clone() }]),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ChatContentPart::Text { text } => {
                    Ok(TemplateContentPart::Text { text: text.clone() })
                }
                // All multimodal contents are normalized to `{ "type": <modality> }`.
                ChatContentPart::ImageUrl { .. } => {
                    multimodal
                        .and_then(|multimodal| multimodal.image_token.as_ref())
                        .ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                    Ok(TemplateContentPart::Image)
                }
                ChatContentPart::VideoUrl { .. } => {
                    multimodal
                        .and_then(|multimodal| multimodal.video_token.as_ref())
                        .ok_or(Error::UnsupportedMultimodalContent("video_url"))?;
                    Ok(TemplateContentPart::Video)
                }
            })
            .collect(),
    }
}

fn to_template_string_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<String> {
    match content {
        ChatContent::Text(text) => Ok(text.clone()),
        ChatContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::ImageUrl { .. } => {
                        let image_token = multimodal
                            .and_then(|multimodal| multimodal.image_token.as_ref())
                            .ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                        out.push_str(image_token);
                    }
                    ChatContentPart::VideoUrl { .. } => {
                        let video_token = multimodal
                            .and_then(|multimodal| multimodal.video_token.as_ref())
                            .ok_or(Error::UnsupportedMultimodalContent("video_url"))?;
                        out.push_str(video_token);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Sentinel appended to the final message content when `continue_final_message`
/// is requested, used to locate the truncation point in the rendered prompt.
///
/// Same literal as `transformers`. Occurrences of this string earlier in the
/// prompt are harmless because truncation uses the rightmost match, and the
/// appended sentinel ends up last as long as the template renders messages in
/// order.
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Append [`CONTINUE_FINAL_MESSAGE_TAG`] to the trailing text of the final
/// message, returning the original text for post-render validation.
// TODO: transformers v5 also allows continuing a non-`content` field (e.g.
// `reasoning_content`) by passing a field name; only the boolean form is
// supported here.
fn append_continue_final_message_tag(message: &mut TemplateMessage) -> Result<String> {
    let text = match &mut message.content {
        TemplateContent::String(text) => Some(text),
        // Pick the last text part in the message.
        TemplateContent::OpenAi(parts) => parts.iter_mut().rev().find_map(|part| match part {
            TemplateContentPart::Text { text } => Some(text),
            TemplateContentPart::Image | TemplateContentPart::Video => None,
        }),
    };
    let text = text.ok_or_else(|| {
        Error::ChatTemplate(
            "continue_final_message is set but there is no text to continue \
             in the final message"
                .to_string(),
        )
    })?;

    let original = text.clone();
    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
    Ok(original)
}

/// Truncate the rendered prompt at [`CONTINUE_FINAL_MESSAGE_TAG`] so that it
/// ends exactly with the final message content, dropping any template suffix
/// such as end-of-turn markers.
fn truncate_prompt_at_continue_final_message_tag(
    mut rendered: String,
    final_message_text: &str,
) -> Result<String> {
    let tag_loc = rendered
        .rfind(CONTINUE_FINAL_MESSAGE_TAG.trim_end())
        .filter(|_| rendered.contains(final_message_text.trim()));
    let Some(tag_loc) = tag_loc else {
        return Err(Error::ChatTemplate(format!(
            "continue_final_message is set but the final message does not appear \
             in the prompt after applying the chat template! This can happen if \
             the chat template deletes portions of the final message. Final \
             message to continue: {}",
            final_message_text.trim(),
        )));
    };

    if rendered[tag_loc..].starts_with(CONTINUE_FINAL_MESSAGE_TAG) {
        // The template preserved spacing, so a plain cut at the tag suffices.
        rendered.truncate(tag_loc);
    } else {
        // The template trimmed the trailing spacing of the message content, so
        // apply the same trimming to the retained prefix.
        rendered.truncate(tag_loc);
        rendered.truncate(rendered.trim_end().len());
    }
    Ok(rendered)
}

fn to_template_tools(tools: &[ChatTool]) -> Vec<TemplateTool> {
    tools
        .iter()
        .map(|tool| TemplateTool {
            tool_type: "function",
            function: TemplateToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: to_template_value(tool.parameters.clone()),
                strict: tool.strict,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use expect_test::expect;
    use serde_json::Value;
    use vllm_text::Prompt;
    use vllm_text::backend::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::{ChatTemplateContentFormatOption, HfChatRenderer, MultimodalRenderInfo};
    use crate::request::{
        ChatContentPart, ChatMessage, ChatRequest, ChatRole, ChatTool, ChatToolChoice,
        GenerationPromptMode, ReasoningEffort,
    };
    use crate::{AssistantContentBlock, ChatRenderer, Error, Result};

    const QWEN3_0_6B_TEMPLATE: &str = include_str!("../../../tests/templates/qwen3.jinja");
    const QWEN3_5_0_8B_TEMPLATE: &str = include_str!("../../../tests/templates/qwen35.jinja");

    fn sample_request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            messages,
            request_id: "render-test".to_string(),
            ..ChatRequest::for_test()
        }
    }

    fn render(template: Option<&str>, request: &ChatRequest) -> Result<String> {
        HfChatRenderer::new(
            template.map(str::to_owned),
            HashMap::new(),
            ChatTemplateContentFormatOption::Auto,
        )?
        .render(request)?
        .prompt
        .into_text()
        .map_err(|_| unreachable!("HF renderer should return text prompt"))
    }

    fn render_mm(
        template: &str,
        request: &ChatRequest,
        content_format: ChatTemplateContentFormatOption,
    ) -> Result<crate::RenderedPrompt> {
        HfChatRenderer::new(Some(template.to_string()), HashMap::new(), content_format)?
            .with_multimodal(Some(MultimodalRenderInfo {
                image_token: Some("<image>".to_string()),
                video_token: Some("<video>".to_string()),
            }))
            .render(request)
    }

    fn image_request() -> ChatRequest {
        sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("a"),
            ChatContentPart::image_url("data:image/png;base64,test"),
            ChatContentPart::text("b"),
        ])])
    }

    fn video_request() -> ChatRequest {
        sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("a"),
            ChatContentPart::video_url("https://example.com/demo.mp4"),
            ChatContentPart::text("b"),
        ])])
    }

    #[test]
    fn string_content_format_replaces_image_with_placeholder_text() {
        let rendered = render_mm(
            "{{ messages[0].content }}",
            &image_request(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("a<image>b".to_string()));
    }

    #[test]
    fn string_content_format_replaces_video_with_placeholder_text() {
        let rendered = render_mm(
            "{{ messages[0].content }}",
            &video_request(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("a<video>b".to_string()));
    }

    #[test]
    fn openai_content_format_normalizes_image_url_for_template() {
        let rendered = render_mm(
            "{% for item in messages[0].content %}{% if item.type == 'image' %}<|image_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}",
            &image_request(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("a<|image_pad|>b".to_string()));
    }

    #[test]
    fn openai_content_format_normalizes_video_url_for_template() {
        let rendered = render_mm(
            "{% for item in messages[0].content %}{% if item.type == 'video' %}<|video_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}",
            &video_request(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("a<|video_pad|>b".to_string()));
    }

    #[test]
    fn video_parts_are_rejected_when_model_lacks_video_support() {
        let error = HfChatRenderer::new(
            Some("{{ messages[0].content }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap()
        .with_multimodal(Some(MultimodalRenderInfo {
            image_token: Some("<image>".to_string()),
            video_token: None,
        }))
        .render(&video_request())
        .unwrap_err();

        assert!(matches!(
            error,
            Error::UnsupportedMultimodalContent("video_url")
        ));
    }

    #[test]
    fn chat_template_supports_pycompat_templates() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "<think>hello")]);

        let rendered = render(
            Some(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "think");
    }

    #[test]
    fn chat_template_passes_continue_final_message_to_template() {
        let mut request = sample_request(vec![ChatMessage::text(
            ChatRole::Assistant,
            "The capital of",
        )]);
        let template =
            "{% if continue_final_message %}continue:{% endif %}{{ messages[0].content }}";

        assert_eq!(render(Some(template), &request).unwrap(), "The capital of");

        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        assert_eq!(
            render(Some(template), &request).unwrap(),
            "continue:The capital of"
        );
    }

    #[test]
    fn continue_final_message_truncates_template_suffix() {
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "What is the capital of France?"),
            ChatMessage::text(ChatRole::Assistant, "The capital of"),
        ]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        // The Qwen3 template is unaware of `continue_final_message`; the
        // end-of-turn marker it appends must still be stripped.
        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            What is the capital of France?<|im_end|>
            <|im_start|>assistant
            <think>

            </think>

            The capital of"#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn continue_final_message_trims_like_the_template_does() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::Assistant, "Sure, ")]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        // The template trims the trailing spacing of the message content, so
        // the truncated prompt must be trimmed the same way.
        let rendered = render(
            Some("{{ messages[0].content.strip() }}<|im_end|>"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "Sure,");
    }

    #[test]
    fn continue_final_message_appends_to_last_text_part() {
        // The renderer itself is role-agnostic like transformers (the
        // assistant-final restriction is enforced by request validation
        // upstream), so a multimodal user message exercises the part
        // selection: the sentinel must attach to the last *text* part,
        // skipping the trailing image.
        let mut request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("Sure,"),
            ChatContentPart::image_url("data:image/png;base64,test"),
        ])]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        let rendered = render_mm(
            "{% for item in messages[0].content %}{% if item.type == 'image' %}<image>{% else %}{{ item.text }}{% endif %}{% endfor %}<|im_end|>",
            &request,
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap()
        .prompt;

        // Anything rendered after the continued text (here the image
        // placeholder and the end marker) is truncated away, matching
        // transformers.
        assert_eq!(rendered, Prompt::Text("Sure,".to_string()));
    }

    #[test]
    fn continue_final_message_composes_with_aware_templates() {
        // A template that reads `continue_final_message` and skips its own
        // end-of-turn marker must produce the same prompt as an unaware one:
        // the sentinel truncation degenerates to a cut at the very end.
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "hi"),
            ChatMessage::text(ChatRole::Assistant, "Sure,"),
        ]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        let aware = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{% if not (loop.last and continue_final_message) %}<|im_end|>\n{% endif %}{% endfor %}";
        let unaware = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}";

        let expected = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nSure,";
        assert_eq!(render(Some(aware), &request).unwrap(), expected);
        assert_eq!(render(Some(unaware), &request).unwrap(), expected);
    }

    #[test]
    fn continue_final_message_errors_when_template_drops_final_message() {
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "hi"),
            ChatMessage::text(ChatRole::Assistant, "Sure,"),
        ]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        let error = render(
            Some(
                "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}",
            ),
            &request,
        )
        .unwrap_err();

        assert!(matches!(error, Error::ChatTemplate(_)));
    }

    #[test]
    fn chat_template_flattens_text_parts_for_string_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_exposes_developer_tools() {
        let request = sample_request(vec![ChatMessage::developer(
            "policy",
            Some(vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                strict: Some(true),
            }]),
        )]);

        let rendered = render(
            Some("{{ messages[0].role }}|{{ messages[0].content }}|{{ messages[0].tools[0].function.name }}|{{ messages[0].tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "developer|policy|get_weather|city");
    }

    #[test]
    fn chat_template_keeps_string_text_for_openai_detected_templates() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn chat_template_emits_openai_text_blocks_for_structured_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text("world"),
        ])]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello|world|");
    }

    #[test]
    fn chat_template_per_request_override() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        // Default template renders one way.
        let default_rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();
        assert_eq!(default_rendered, "hello");

        // Per-request override replaces the default template entirely.
        request.chat_options.chat_template = Some("override:{{ messages[0].content }}".to_string());
        let overridden = render(Some("{{ messages[0].content }}"), &request).unwrap();
        assert_eq!(overridden, "override:hello");
    }

    #[test]
    fn chat_template_per_request_override_without_default_template() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.chat_options.chat_template = Some("override:{{ messages[0].content }}".to_string());

        let rendered = render(None, &request).unwrap();

        assert_eq!(rendered, "override:hello");
    }

    #[test]
    fn chat_template_requires_a_template() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let error = render(None, &request).unwrap_err();

        assert!(matches!(error, Error::MissingChatTemplate));
    }

    #[test]
    fn chat_template_injects_special_tokens_into_context() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<bos>".to_string())),
            ..Default::default()
        };

        let rendered = HfChatRenderer::new(
            Some("{{ bos_token }}|{{ bos_token is defined }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap()
        .with_special_tokens(Some(special_tokens))
        .apply_chat_template(&request)
        .unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("<bos>|true".to_string()));
    }

    #[test]
    fn chat_template_exposes_assistant_reasoning_separately() {
        let request = sample_request(vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "inner".to_string(),
            },
            AssistantContentBlock::Text {
                text: "outer".to_string(),
            },
        ])]);

        let rendered = render(
            Some("{{ messages[0].reasoning_content }}|{{ messages[0].content }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "inner|outer");
    }

    #[test]
    fn chat_template_forces_string_content_format_when_configured() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some(
                "{%- if messages[0].content is string -%}{{ messages[0].content }}{%- else -%}{%- for item in messages[0].content %}{{ item.text }}|{%- endfor -%}{%- endif -%}".to_string(),
            ),
            HashMap::new(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap()
        .render(&request)
        .unwrap()
        .prompt;

        assert_eq!(rendered, Prompt::Text("hello world".to_string()));
    }

    #[test]
    fn chat_template_forces_openai_content_format_when_configured() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some("{{ messages[0].content[0].text }}{{ messages[0].content[1].text }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap()
        .render(&request)
        .unwrap()
        .prompt;

        assert_eq!(rendered, Prompt::Text("hello world".to_string()));
    }

    #[test]
    fn chat_template_merges_default_template_kwargs_before_request_kwargs() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), Value::Bool(true));

        let renderer = HfChatRenderer::new(
            Some("{{ enable_thinking }}|{{ default_only }}".to_string()),
            HashMap::from([
                ("enable_thinking".to_string(), Value::Bool(false)),
                ("default_only".to_string(), Value::String("x".to_string())),
            ]),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap().prompt;

        assert_eq!(rendered, Prompt::Text("true|x".to_string()));
    }

    #[test]
    fn chat_template_reasoning_effort_overrides_template_kwargs() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.chat_options.reasoning_effort = Some(ReasoningEffort::Max);
        request.chat_options.template_kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String("low".to_string()),
        );

        let renderer = HfChatRenderer::new(
            Some("{{ reasoning_effort }}".to_string()),
            HashMap::from([(
                "reasoning_effort".to_string(),
                Value::String("medium".to_string()),
            )]),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("max".to_string()));
        assert_eq!(
            rendered.effective_template_kwargs.get("reasoning_effort"),
            Some(&Value::String("max".to_string()))
        );
        assert_eq!(
            rendered.effective_template_kwargs.get("enable_thinking"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn chat_template_reasoning_effort_preserves_request_enable_thinking() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.chat_options.reasoning_effort = Some(ReasoningEffort::None);
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), Value::Bool(true));

        let renderer = HfChatRenderer::new(
            Some("{{ reasoning_effort }}|{{ enable_thinking }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap();

        assert_eq!(rendered.prompt, Prompt::Text("none|true".to_string()));
        assert_eq!(
            rendered.effective_template_kwargs.get("reasoning_effort"),
            Some(&Value::String("none".to_string()))
        );
        assert_eq!(
            rendered.effective_template_kwargs.get("enable_thinking"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn qwen3_template_omits_reasoning_for_historical_assistant_messages() {
        let request = sample_request(vec![
            ChatMessage::text(
                ChatRole::User,
                "Hi. Tell me about the capital of France in short",
            ),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "\nOkay, the user is asking... I think that's all.\n".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "Paris is the capital of France.".to_string(),
                },
            ]),
            ChatMessage::text(ChatRole::User, "Tell me about Paris more."),
        ]);

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            Hi. Tell me about the capital of France in short<|im_end|>
            <|im_start|>assistant
            Paris is the capital of France.<|im_end|>
            <|im_start|>user
            Tell me about Paris more.<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_keeps_reasoning_after_the_last_user_query() {
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "What is 1 + 1?"),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "need simple arithmetic".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "2".to_string(),
                },
            ]),
        ]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            What is 1 + 1?<|im_end|>
            <|im_start|>assistant
            <think>
            need simple arithmetic
            </think>

            2<|im_end|>
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chat_template_exposes_tools_to_templates_when_auto_enabled() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.tools = vec![ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
            strict: None,
        }];
        request.tool_choice = ChatToolChoice::Auto;

        let rendered = render(
            Some("{{ tools[0].function.name }}|{{ tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|city");
    }

    #[test]
    fn chat_template_exposes_assistant_tool_calls_and_tool_messages() {
        let request = sample_request(vec![
            ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(
                crate::AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },
            )]),
            ChatMessage::tool_response("Sunny", "call_1"),
        ]);

        let rendered = render(
            Some(
                "{{ messages[0].tool_calls[0].function.name }}|{{ messages[0].tool_calls[0].function.arguments.city }}|{{ messages[1].tool_call_id }}|{{ messages[1].content }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|Paris|call_1|Sunny");
    }

    #[test]
    fn chat_template_tool_call_argument_items_method_is_not_shadowed_by_field() {
        let request = sample_request(vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::ToolCall(crate::AssistantToolCall {
                id: "call_1".to_string(),
                name: "add".to_string(),
                arguments: r#"{"items":"operands","x":2,"y":1.0}"#.to_string(),
            }),
        ])]);

        let rendered = render(
            Some(
                "{%- set arguments = messages[0].tool_calls[0].function.arguments -%}
{%- for key, value in arguments.items() -%}{{ key }}={{ value }};{%- endfor -%}
|{{ arguments['items'] }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "items=operands;x=2;y=1.0;|operands");
    }

    #[test]
    fn qwen35_template_renders_prefilled_reasoning_start_when_thinking_enabled() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), Value::Bool(true));

        let rendered = render(Some(QWEN3_5_0_8B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
            <think>
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen35_template_renders_closed_empty_reasoning_span_when_thinking_disabled() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), Value::Bool(false));

        let rendered = render(Some(QWEN3_5_0_8B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
            <think>

            </think>

        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen35_template_omits_assistant_reasoning_prefill_without_generation_prompt() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), Value::Bool(true));

        let rendered = render(Some(QWEN3_5_0_8B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            hello<|im_end|>
        "#]]
        .assert_eq(&rendered);
    }
}
