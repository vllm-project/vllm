// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser for Beam checkpoints.
//!
//! Beam output is parsed by [`HfUnifiedParser`] with the checkpoint's
//! `response_template`; this parser adds Beam's output grammar.
//! A Beam message after `<|start|>assistant` is:
//!
//! ```text
//! <|channel|>analysis<|message|>REASONING<|end|><|start|>assistant
//! ( <|channel|>final<|message|>ANSWER<|return|>
//! | CALL (<|end|><|start|>assistant CALL)* <|call|> )
//! ```
//!
//! where reasoning may end at any closer of the template's reasoning field,
//! and `CALL` is
//! ` to=NAME<|channel|>analysis <|constrain|>xml<|message|>` followed by
//! `<rfl:parameter name="KEY">\nVALUE\n</rfl:parameter>` parameters separated
//! by newlines. Where the template looks ahead for the next parameter, a value
//! may also end on a newline right before it.
//!
//! The grammar enforces the schema of every declared tool's arguments, in
//! declared or alphabetical order, since Beam drifts from schemas it does not
//! enforce. When the model may answer instead, calls may also name any other
//! tool with any arguments, which the parser passes on so a harness can tell
//! the model the tool does not exist. `ANSWER` follows the request's structured
//! output, if any.
//!
//! The chat template rejects a history turn without reasoning, so a message the
//! model opens holds non-empty reasoning: an empty one would come back as null
//! reasoning and fail the next request.
//!
//! Unless `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT` turns it off, the grammar
//! applies to every message, including one the request does not constrain.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use serde_json::Value;
use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::ToolChoice;
use xgrammar_structural_tag::format::{EndBoundary, Format};
use xgrammar_structural_tag::tool::{
    BuilderToolChoice, NormalizedToolChoice, ToolChoiceValue, normalize_tool_choice,
};

use super::{
    HfPosition, HfUnifiedParser, ResponseTemplate, Result, UnifiedParser, UnifiedParserError,
    UnifiedParserOutput,
};
use crate::output_grammar::arguments::{
    self, ArgumentOptions, ArgumentSyntax, JsonType, ParameterKey, Parameters, ValueOption, one_of,
};
use crate::output_grammar::{
    self, BuiltOutputGrammar, OutputGrammarContext, tool_grammar_applies, tool_params,
};
use crate::tool::Tool;

const ANALYSIS_OPEN: &str = "<|channel|>analysis<|message|>";
const FINAL_OPEN: &str = "<|channel|>final<|message|>";
const END: &str = "<|end|>";
/// Closes one message and opens the next assistant message.
const NEXT_MESSAGE: &str = "<|end|><|start|>assistant";
const RETURN: &str = "<|return|>";
const CALL: &str = "<|call|>";
/// Follows ` to=NAME` in a call.
const CALL_HEADER_END: &str = "<|channel|>analysis <|constrain|>xml<|message|>";
const PARAMETER_OPEN: &str = "<rfl:parameter name=\"";
const PARAMETER_NAME_CLOSE: &str = "\">\n";
const PARAMETER_CLOSE: &str = "\n</rfl:parameter>";
/// Ends a value where the template looks ahead for the next parameter.
const NEXT_PARAMETER: &str = "\n<rfl:parameter name=\"";
/// Ends of a lenient value that another parameter follows: the closer and
/// separator before the next opener, or the opener right after the value.
const NEXT_PARAMETER_ENDS: [&str; 3] = [
    "\n</rfl:parameter><rfl:parameter name=\"",
    "\n</rfl:parameter>\n<rfl:parameter name=\"",
    NEXT_PARAMETER,
];
/// Text that ends the call in the parser, even inside a JSON value.
const CALL_TERMINATORS: [&str; 2] = [END, CALL];

/// Whether every Beam message follows the grammar, even when the request
/// constrains neither tool calls nor the answer. Read from
/// `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT` like the Python frontend: on unless
/// set to something other than `true` or `1`.
static ALWAYS_CONSTRAIN_OUTPUT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT").map_or(true, |value| {
        matches!(value.to_lowercase().as_str(), "true" | "1")
    })
});

/// Beam parser: [`HfUnifiedParser`] with Beam's output grammar.
pub struct BeamUnifiedParser {
    inner: HfUnifiedParser,
    /// Whether a value may end where the next parameter opens.
    lenient: bool,
    /// What ends the reasoning and opens the next message.
    reasoning_ends: Vec<String>,
}

impl BeamUnifiedParser {
    /// Create a parser for one request from the checkpoint's `response_template`.
    pub fn new(template: Arc<ResponseTemplate>, tools: &[Tool], tokenizer: DynTokenizer) -> Self {
        Self {
            lenient: template.argument_lookahead() == Some(PARAMETER_OPEN),
            reasoning_ends: match template.reasoning_closers() {
                [] => vec![NEXT_MESSAGE.to_string()],
                closers => closers.to_vec(),
            },
            inner: HfUnifiedParser::new(template, tools, tokenizer),
        }
    }
}

impl UnifiedParser for BeamUnifiedParser {
    fn create(_tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>> {
        Err(UnifiedParserError::NoNamedConstructor {
            parser: "beam",
            built_from: "the checkpoint's response_template",
        })
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        build_output_grammar(
            self.inner.position(),
            ctx,
            *ALWAYS_CONSTRAIN_OUTPUT,
            self.lenient,
            &self.reasoning_ends,
        )
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.inner.parse_into(delta, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

/// The grammar of a Beam message from the first generated token, or `None`
/// when the parser is neither at the start of the message nor inside its
/// reasoning, or when the request constrains nothing and `always_constrain` is
/// off. Values may end where the next parameter opens if `lenient`, and
/// reasoning ends at any of `reasoning_ends`.
fn build_output_grammar(
    position: HfPosition,
    ctx: &OutputGrammarContext<'_>,
    always_constrain: bool,
    lenient: bool,
    reasoning_ends: &[String],
) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
    let forced = !ctx.tools.is_empty()
        && matches!(
            ctx.tool_choice,
            ToolChoice::Value(ToolChoiceValue::Required) | ToolChoice::NamedFunction(_)
        );
    let answer_format = ctx.answer.filter(|_| !forced);
    // The prompt may still list the tools, so only the grammar keeps the model
    // from calling them.
    let tools_forbidden = !ctx.tools.is_empty()
        && matches!(ctx.tool_choice, ToolChoice::Value(ToolChoiceValue::None));
    if !(always_constrain
        || tool_grammar_applies(ctx)
        || tools_forbidden
        || answer_format.is_some())
    {
        return Ok(None);
    }
    let text = Format::any_text();
    let reasoning_end = EndBoundary::from(reasoning_ends.to_vec());
    let reasoning = match position {
        HfPosition::ResponseStart => Format::tag(
            ANALYSIS_OPEN,
            // At least one character, which cannot start Beam's closers.
            Format::sequence(vec![Format::regex("[^<]"), text]),
            reasoning_end,
        ),
        HfPosition::Reasoning => Format::tag("", text, reasoning_end),
        // Inside an answer the prompt opened, only the always-on grammar
        // applies, finishing a free answer. A tool choice that forces a call
        // cannot be met there, and the request's own constraint applies as is.
        HfPosition::Content => {
            return Ok(
                (always_constrain && !forced && answer_format.is_none()).then(|| {
                    BuiltOutputGrammar::from_token_zero(Format::sequence(answer_rest(None)))
                }),
            );
        }
        HfPosition::Other => return Ok(None),
    };
    let NormalizedToolChoice {
        function_tools,
        choice,
        ..
    } = normalize_tool_choice(
        &tool_params(ctx.tools, ctx.tool_strict_level),
        ctx.tool_choice.clone(),
    )?;
    if function_tools.is_empty() {
        return Ok(match choice {
            // `none`, or no tools to call: the model can only answer.
            BuilderToolChoice::Auto => {
                Some(BuiltOutputGrammar::from_token_zero(Format::sequence(vec![
                    reasoning,
                    answer(answer_format),
                ])))
            }
            BuilderToolChoice::Required | BuilderToolChoice::Forced => None,
        });
    }

    let mut calls = function_tools
        .iter()
        .map(|tool| {
            let header =
                Format::const_string(format!(" to={}{CALL_HEADER_END}", tool.function.name));
            let schema = tool.function.parameters.clone().unwrap_or(Value::Bool(true));
            Format::sequence(vec![header, arguments(&schema, lenient)])
        })
        .collect::<Vec<_>>();
    if matches!(choice, BuilderToolChoice::Auto) {
        let names = ctx.tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>();
        let header = Format::sequence(vec![
            Format::const_string(" to="),
            Format::regex(other_names(&names)),
            Format::const_string(CALL_HEADER_END),
        ]);
        calls.push(Format::sequence(vec![
            header,
            arguments(&Value::Bool(true), lenient),
        ]));
    }
    let call = one_of(calls);
    let mut calls = vec![call.clone()];
    if ctx.parallel_tool_calls && !choice.is_forced() {
        calls.push(Format::star(Format::sequence(vec![
            Format::const_string(NEXT_MESSAGE),
            call,
        ])));
    }
    // The last call ends with the `<|call|>` stop token.
    calls.push(Format::optional(Format::const_string(CALL)));
    let calls = Format::sequence(calls);
    let output = match choice {
        BuilderToolChoice::Auto => Format::or(vec![calls, answer(answer_format)]),
        BuilderToolChoice::Required | BuilderToolChoice::Forced => calls,
    };
    Ok(Some(BuiltOutputGrammar::from_token_zero(Format::sequence(
        vec![reasoning, output],
    ))))
}

/// A final answer, ending with the `<|return|>` stop token.
fn answer(format: Option<&Format>) -> Format {
    Format::sequence([vec![Format::const_string(FINAL_OPEN)], answer_rest(format)].concat())
}

/// The rest of an open final answer, holding `format` or else free text.
fn answer_rest(format: Option<&Format>) -> Vec<Format> {
    vec![
        // Without `<|end|>`, no unconstrained message can follow the answer.
        format.map_or_else(|| Format::any_text_excluding(&[END, RETURN]), Format::clone),
        Format::optional(Format::const_string(RETURN)),
    ]
}

/// The arguments under `schema`: its declared parameters in declared or
/// alphabetical order, required ones mandatory, then any additional ones. If
/// `lenient`, each value but the last may end by opening the next parameter.
fn arguments(schema: &Value, lenient: bool) -> Format {
    let options = ArgumentOptions {
        excludes: CALL_TERMINATORS.map(str::to_string).to_vec(),
        ..ArgumentOptions::default()
    };
    let render = |syntax: BeamArguments| arguments::parameters(schema, &syntax, &options);
    if lenient {
        lenient_arguments(render(BeamArguments::Joined), render(BeamArguments::Last))
    } else {
        closed_arguments(render(BeamArguments::Closed))
    }
}

/// Arguments whose values all end with a closer.
fn closed_arguments(
    Parameters {
        declared,
        additional,
    }: Parameters,
) -> Format {
    let additional = additional.map(Format::star);
    let lists = orders(&declared, |parameter| &parameter.key)
        .into_iter()
        .map(|order| {
            let listed = order
                .into_iter()
                .map(|parameter| listed(parameter.format.clone(), parameter.required));
            listed.chain(additional.clone()).collect::<Vec<_>>()
        })
        .filter(|list| !list.is_empty())
        .map(Format::sequence)
        .collect::<Vec<_>>();
    if lists.is_empty() {
        Format::const_string("")
    } else {
        one_of(lists)
    }
}

/// A declared parameter as one another parameter follows, `joined`, and as the
/// last one.
struct LenientParameter {
    key: String,
    joined: Format,
    last: Format,
    required: bool,
}

impl LenientParameter {
    fn joined(&self) -> Format {
        listed(self.joined.clone(), self.required)
    }
}

/// Arguments whose values end by opening the next parameter, but for the last
/// one, from the parameters as [`BeamArguments::Joined`] and as
/// [`BeamArguments::Last`].
fn lenient_arguments(joined: Parameters, last: Parameters) -> Format {
    // Both forms share their terminators, so they keep the same parameters.
    let declared = joined
        .declared
        .into_iter()
        .zip(last.declared)
        .map(|(joined, last)| LenientParameter {
            key: joined.key,
            joined: joined.format,
            last: last.format,
            required: joined.required,
        })
        .collect::<Vec<_>>();
    let additional = joined.additional.zip(last.additional);

    let mut lists = vec![];
    for order in orders(&declared, |parameter| &parameter.key) {
        // The last parameter is a declared one that follows every required
        // one, or an additional one.
        if !order.is_empty() {
            let first = order.iter().rposition(|parameter| parameter.required).unwrap_or(0);
            let mut list =
                order[..first].iter().map(|parameter| parameter.joined()).collect::<Vec<_>>();
            list.push(last_present(&order[first..]));
            lists.push(Format::sequence(list));
        }
        if let Some((joined, last)) = &additional {
            let mut list = order.iter().map(|parameter| parameter.joined()).collect::<Vec<_>>();
            list.extend([Format::star(joined.clone()), last.clone()]);
            lists.push(Format::sequence(list));
        }
    }
    if lists.is_empty() {
        return Format::const_string("");
    }

    // Each parameter starts past the opener, as the value before writes it.
    let arguments = Format::sequence(vec![
        separator(),
        Format::const_string(PARAMETER_OPEN),
        one_of(lists),
    ]);
    if declared.iter().any(|parameter| parameter.required) {
        arguments
    } else {
        Format::optional(arguments)
    }
}

/// `parameters`, at least one, in order and the last present one in its last
/// form. Only the first may be required.
fn last_present(parameters: &[&LenientParameter]) -> Format {
    if let [only] = parameters {
        return only.last.clone();
    }
    // The last one is in the first half, or after all of it. Halving keeps the
    // grammar O(n log n) in size and O(log n) deep.
    let (head, tail) = parameters.split_at(parameters.len() / 2);
    let mut after_head = head.iter().map(|parameter| parameter.joined()).collect::<Vec<_>>();
    after_head.push(last_present(tail));
    Format::or(vec![last_present(head), Format::sequence(after_head)])
}

/// `parameters` in declared order, and in alphabetical order by `key` if that
/// differs, since Beam often writes them sorted.
fn orders<P>(parameters: &[P], key: impl Fn(&P) -> &str) -> Vec<Vec<&P>> {
    let declared = parameters.iter().collect::<Vec<_>>();
    let mut sorted = declared.clone();
    sorted.sort_by_key(|parameter| key(parameter));
    if sorted
        .iter()
        .map(|parameter| key(parameter))
        .eq(declared.iter().map(|parameter| key(parameter)))
    {
        vec![declared]
    } else {
        vec![declared, sorted]
    }
}

/// `parameter` as listed: optional unless `required`.
fn listed(parameter: Format, required: bool) -> Format {
    if required {
        parameter
    } else {
        Format::optional(parameter)
    }
}

fn separator() -> Format {
    Format::optional(Format::const_string("\n"))
}

/// A regex for the tool names the call header can spell (no whitespace or
/// `<`) besides `names`.
fn other_names(names: &[&str]) -> String {
    #[derive(Default)]
    struct Node {
        children: BTreeMap<char, Node>,
        end: bool,
    }

    fn escape(char: char, meta: &str) -> String {
        if meta.contains(char) {
            format!("\\{char}")
        } else {
            char.to_string()
        }
    }

    /// A name character other than `chars`.
    fn other<'a>(chars: impl Iterator<Item = &'a char>) -> String {
        let chars = chars.map(|&char| escape(char, r"\]^-[")).collect::<String>();
        format!(r"[^\s<{chars}]")
    }

    /// The rest of a name past `node` that ends on none of `names`.
    fn rest(node: &Node, first: bool) -> String {
        let any = other([].iter());
        let mut branches = vec![format!("{}{any}*", other(node.children.keys()))];
        branches.extend(node.children.iter().map(|(&char, child)| {
            format!("{}{}", escape(char, r".^$*+?{}[]\|()"), rest(child, false))
        }));
        let optional = if first || node.end { "" } else { "?" };
        format!("(?:{}){optional}", branches.join("|"))
    }

    let mut root = Node::default();
    for name in names {
        let mut node = &mut root;
        for char in name.chars() {
            node = node.children.entry(char).or_default();
        }
        node.end = true;
    }
    rest(&root, true)
}

/// How one parameter is written: `<rfl:parameter name="KEY">\nVALUE` followed
/// by its end.
///
/// The parser reads every value as text and converts it by the tool schema, so
/// string options are raw text and every other option is JSON.
enum BeamArguments {
    /// The whole parameter, ending with `\n</rfl:parameter>`, after an optional
    /// newline.
    Closed,
    /// The parameter past its opener, where the template looks ahead for the
    /// next one: it ends by opening the next parameter, also right after the
    /// value's newline, so raw text holds no such opener.
    Joined,
    /// The last parameter past its opener under such a template, ending with
    /// `\n</rfl:parameter>`.
    Last,
}

impl ArgumentSyntax for BeamArguments {
    fn separator(&self) -> Option<Format> {
        None
    }

    fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
        let terminators: &[&str] = match self {
            Self::Closed => &["</rfl:parameter>"],
            Self::Joined | Self::Last => &["</rfl:parameter>", NEXT_PARAMETER],
        };
        let values = options
            .iter()
            .filter_map(|option| match option.ty {
                JsonType::String => option.raw_string(terminators),
                _ => Some(option.json()),
            })
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| {
            let value = one_of(values);
            match self {
                Self::Closed => Format::sequence(vec![
                    separator(),
                    key.tag(PARAMETER_OPEN, PARAMETER_NAME_CLOSE, value, PARAMETER_CLOSE),
                ]),
                Self::Joined => key.tag(
                    "",
                    PARAMETER_NAME_CLOSE,
                    value,
                    NEXT_PARAMETER_ENDS.to_vec(),
                ),
                Self::Last => key.tag("", PARAMETER_NAME_CLOSE, value, PARAMETER_CLOSE),
            }
        })
    }
}

#[cfg(test)]
mod tests;
