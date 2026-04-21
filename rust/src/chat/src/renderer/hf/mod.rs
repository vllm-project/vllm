use std::collections::HashMap;

use openai_protocol::common::Tool as OpenAiTool;
use serde::Serialize;
use serde_json::Value;
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
use super::{ChatRenderer, RenderedPrompt};
use crate::error::Result;
use crate::request::{ChatContent, ChatMessage, ChatRequest};
use crate::{
    AssistantContentBlock, AssistantMessageExt, ChatTool, Error, LoadModelBackendsOptions,
};

mod error;
mod format;
mod template;
mod tojson;

pub use template::{load_chat_template, resolve_chat_template};

pub use self::format::ChatTemplateContentFormatOption;

/// Hugging Face chat-template renderer backed by the local Jinja chat-template state.
pub struct HfChatRenderer {
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: HashMap<String, Value>,
    content_format: ContentFormatOption,
    special_tokens: Option<HfSpecialTokens>,
}

impl HfChatRenderer {
    /// Create a renderer from the given template string.
    pub fn new(
        template: Option<String>,
        default_template_kwargs: HashMap<String, Value>,
        content_format: ContentFormatOption,
        special_tokens: Option<HfSpecialTokens>,
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
            special_tokens,
        })
    }

    /// Create a renderer from the given model files and loading options.
    pub fn load(files: &ResolvedModelFiles, options: LoadModelBackendsOptions) -> Result<Self> {
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
            // If independent chat template file(s) exist and contain non-empty content, they take
            // priority over template entries in the tokenizer config
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

        Self::new(
            template,
            options.default_chat_template_kwargs,
            options.chat_template_content_format,
            special_tokens,
        )
    }

    /// Apply the chat template to one chat request, rendering the prompt string to be tokenized
    /// and submitted to the model.
    ///
    /// If the request carries a per-request `chat_template` override, a temporary template is
    /// compiled from that string and used instead of the model's default.
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
        let messages =
            to_template_messages(&request.messages, effective_template.content_format())?;
        let tools = request
            .tool_parsing_enabled()
            .then(|| to_template_tools(&request.tools));
        trace!(
            message_count = messages.len(),
            content_format = ?effective_template.content_format(),
            ?messages,
            ?tools,
            "applying chat template"
        );

        let mut merged_template_kwargs = self.default_template_kwargs.clone();
        merged_template_kwargs.extend(request.chat_options.template_kwargs.clone());

        let prompt = effective_template
            .apply(TemplateContext {
                messages: &messages,
                add_generation_prompt: request.chat_options.add_generation_prompt(),
                continue_final_message: request.chat_options.continue_final_message(),
                tools: tools.as_deref(),
                documents: request.documents.as_deref(),
                template_kwargs: Some(&merged_template_kwargs),
                special_tokens: self.special_tokens.as_ref(),
            })
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

        trace!(
            prompt_len = prompt.len(),
            prompt, "rendered chat template prompt"
        );

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
        })
    }
}

impl ChatRenderer for HfChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        self.apply_chat_template(request)
    }
}

/// Chat message in the JSON shape expected by Jinja chat templates.
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
    OpenAi(ChatContent),
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
    arguments: Value,
}

#[derive(Debug, Serialize)]
#[serde(transparent)]
pub(super) struct TemplateTool(OpenAiTool);

/// Convert chat messages into the JSON shape expected by Jinja chat templates.
fn to_template_messages(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<TemplateMessage>> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format))
        .collect()
}

fn to_template_message(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
) -> Result<TemplateMessage> {
    Ok(match message {
        ChatMessage::System { content } => TemplateMessage {
            role: "system",
            content: to_template_content(content, content_format)?,
            tools: None,
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Developer { content, tools } => TemplateMessage {
            role: "developer",
            content: to_template_content(content, content_format)?,
            tools: tools.as_deref().map(to_template_tools),
            reasoning: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::User { content } => TemplateMessage {
            role: "user",
            content: to_template_content(content, content_format)?,
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
            let content = to_template_content(&ChatContent::Text(text), content_format)?;
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
            content: to_template_content(content, content_format)?,
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
        let arguments = serde_json::from_str::<Value>(&tool_call.arguments).map_err(|error| {
            Error::ChatTemplate(format!(
                "assistant tool call `{}` has invalid JSON arguments: {}",
                tool_call.id,
                error.as_report()
            ))
        })?;

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
) -> Result<TemplateContent> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(content.try_flatten_to_text()?)
        }
        ChatTemplateContentFormat::OpenAi => TemplateContent::OpenAi(content.clone()),
    })
}

fn to_template_tools(tools: &[ChatTool]) -> Vec<TemplateTool> {
    tools
        .iter()
        .map(|tool| TemplateTool(tool.to_openai_tool()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use expect_test::expect;
    use serde_json::Value;
    use vllm_text::Prompt;
    use vllm_text::backend::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::{ChatTemplateContentFormatOption, HfChatRenderer};
    use crate::request::{
        ChatContentPart, ChatMessage, ChatRequest, ChatRole, ChatTool, ChatToolChoice,
        GenerationPromptMode,
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
            None,
        )?
        .render(request)?
        .prompt
        .into_text()
        .map_err(|_| unreachable!("HF renderer should return text prompt"))
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

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "new"
        );

        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "continue"
        );
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
            Some(special_tokens),
        )
        .unwrap()
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
            None,
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
            None,
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
            None,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap().prompt;

        assert_eq!(rendered, Prompt::Text("true|x".to_string()));
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
