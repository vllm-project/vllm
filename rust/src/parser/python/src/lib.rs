// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Thin PyO3 bindings for Rust unified parsers.
//!
//! Parser state and model-specific grammar stay in Rust. Python supplies tool
//! schemas and an immutable tokenizer metadata snapshot, then adapts ordered
//! parser events to its serving protocol.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_parser::tool::{Tool, ToolCallDelta};
use vllm_parser::unified::{
    InklingUnifiedParser, MiniMaxM3CombinedParser, UnifiedParser, UnifiedParserEvent,
    UnifiedParserOutput,
};
use vllm_tokenizer::{DynTokenizer, Tokenizer, TokenizerError};

macro_rules! unified_parser_registry {
    ($($name:literal => $parser:ty),+ $(,)?) => {
        const UNIFIED_PARSER_NAMES: &[&str] = &[$($name),+];

        /// Construct one registered Rust unified parser.
        fn create_unified_parser(
            name: &str,
            tools: &[Tool],
            tokenizer: DynTokenizer,
        ) -> PyResult<Box<dyn UnifiedParser>> {
            let result = match name {
                $($name => <$parser>::create(tools, tokenizer),)+
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported unified parser `{name}`"
                    )));
                }
            };
            result.map_err(|error| PyValueError::new_err(error.to_report_string()))
        }
    };
}

unified_parser_registry! {
    "inkling" => InklingUnifiedParser,
    "minimax_m3" => MiniMaxM3CombinedParser,
}

/// Immutable tokenizer data required by the registered unified parsers.
struct MetadataTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_ids: HashSet<u32>,
}

impl Tokenizer for MetadataTokenizer {
    fn encode(&self, _text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        Err(TokenizerError(
            "metadata tokenizer does not support encode".to_string(),
        ))
    }

    fn encode_ordinary(&self, _text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
        Err(TokenizerError(
            "metadata tokenizer does not support encode_ordinary".to_string(),
        ))
    }

    fn decode(
        &self,
        _token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_tokenizer::Result<String> {
        Err(TokenizerError(
            "metadata tokenizer does not support decode".to_string(),
        ))
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.special_ids.contains(&token_id)
    }
}

#[pyclass(
    name = "TokenizerMetadata",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyTokenizerMetadata(DynTokenizer);

#[pymethods]
impl PyTokenizerMetadata {
    #[new]
    fn new(token_to_id: HashMap<String, u32>, special_ids: HashSet<u32>) -> Self {
        let id_to_token =
            token_to_id.iter().map(|(token, token_id)| (*token_id, token.clone())).collect();
        Self(Arc::new(MetadataTokenizer {
            token_to_id,
            id_to_token,
            special_ids,
        }))
    }
}

#[pyclass(name = "Tool", module = "vllm._rust_tool_parser", skip_from_py_object)]
#[derive(Clone)]
struct PyTool(Tool);

#[pymethods]
impl PyTool {
    #[new]
    #[pyo3(signature = (name, description, parameters, strict=None))]
    fn new(
        name: String,
        description: Option<String>,
        parameters: &Bound<'_, PyAny>,
        strict: Option<bool>,
    ) -> PyResult<Self> {
        let parameters = depythonize::<Value>(parameters).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert tool parameters from Python to JSON: {error}"
            ))
        })?;
        Ok(Self(Tool {
            name,
            description,
            parameters,
            strict,
        }))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.0.description.as_deref()
    }

    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize(py, &self.0.parameters).map(Bound::unbind).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert tool parameters from JSON to Python: \
                     {error}"
            ))
        })
    }

    #[getter]
    fn strict(&self) -> Option<bool> {
        self.0.strict
    }
}

#[pyclass(
    name = "ToolCallDelta",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolCallDelta(ToolCallDelta);

#[pymethods]
impl PyToolCallDelta {
    #[getter]
    fn tool_index(&self) -> usize {
        self.0.tool_index
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[getter]
    fn arguments(&self) -> &str {
        &self.0.arguments
    }
}

#[pyclass(
    name = "UnifiedParserEvent",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyUnifiedParserEvent {
    kind: &'static str,
    text: Option<String>,
    tool_call: Option<PyToolCallDelta>,
}

impl From<UnifiedParserEvent> for PyUnifiedParserEvent {
    fn from(event: UnifiedParserEvent) -> Self {
        match event {
            UnifiedParserEvent::Text(text) => Self {
                kind: "text",
                text: Some(text),
                tool_call: None,
            },
            UnifiedParserEvent::Reasoning(text) => Self {
                kind: "reasoning",
                text: Some(text),
                tool_call: None,
            },
            UnifiedParserEvent::ToolCall(tool_call) => Self {
                kind: "tool_call",
                text: None,
                tool_call: Some(PyToolCallDelta(tool_call)),
            },
        }
    }
}

#[pymethods]
impl PyUnifiedParserEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    #[getter]
    fn tool_call(&self) -> Option<PyToolCallDelta> {
        self.tool_call.clone()
    }
}

#[pyclass(
    name = "UnifiedParserOutput",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone, Default)]
struct PyUnifiedParserOutput(UnifiedParserOutput);

#[pymethods]
impl PyUnifiedParserOutput {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[getter]
    fn events(&self) -> Vec<PyUnifiedParserEvent> {
        self.0.events.iter().cloned().map(PyUnifiedParserEvent::from).collect()
    }

    fn append(&mut self, other: PyRef<'_, PyUnifiedParserOutput>) {
        self.0.append(other.0.clone());
    }
}

#[pyclass(name = "UnifiedParser", module = "vllm._rust_tool_parser", unsendable)]
struct PyUnifiedParser(Box<dyn UnifiedParser>);

impl PyUnifiedParser {
    fn parse_into_output(
        &mut self,
        chunk: &str,
        output: &mut PyUnifiedParserOutput,
    ) -> PyResult<()> {
        self.0
            .parse_into(chunk, &mut output.0)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }
}

#[pymethods]
impl PyUnifiedParser {
    #[new]
    fn new(
        py: Python<'_>,
        parser_name: &str,
        tools: Vec<Py<PyTool>>,
        tokenizer: Py<PyTokenizerMetadata>,
    ) -> PyResult<Self> {
        let tools = tools.iter().map(|tool| tool.borrow(py).0.clone()).collect::<Vec<_>>();
        let tokenizer = tokenizer.borrow(py).0.clone();
        create_unified_parser(parser_name, &tools, tokenizer).map(Self)
    }

    fn initialize(&mut self, prompt_token_ids: Vec<u32>) -> PyResult<()> {
        self.0
            .initialize(&prompt_token_ids)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }

    fn parse_into(
        &mut self,
        chunk: &str,
        mut output: PyRefMut<'_, PyUnifiedParserOutput>,
    ) -> PyResult<()> {
        self.parse_into_output(chunk, &mut output)
    }

    fn finish(&mut self) -> PyResult<PyUnifiedParserOutput> {
        self.0
            .finish()
            .map(PyUnifiedParserOutput)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.0.preserve_special_tokens()
    }

    fn reasoning_start_str(&self) -> Option<&str> {
        self.0.reasoning_start_str()
    }

    fn reasoning_end_str(&self) -> Option<&str> {
        self.0.reasoning_end_str()
    }

    fn is_reasoning_end(&self, input_ids: Vec<u32>) -> bool {
        self.0.is_reasoning_end(&input_ids)
    }

    fn count_reasoning_tokens(&self, input_ids: Vec<u32>) -> usize {
        self.0.count_reasoning_tokens(&input_ids)
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.0.tool_call_id(tool_index)
    }
}

#[pyfunction]
fn list_unified_parsers() -> Vec<&'static str> {
    UNIFIED_PARSER_NAMES.to_vec()
}

#[pymodule]
fn _rust_tool_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_unified_parsers, m)?)?;
    m.add_class::<PyTokenizerMetadata>()?;
    m.add_class::<PyTool>()?;
    m.add_class::<PyToolCallDelta>()?;
    m.add_class::<PyUnifiedParserEvent>()?;
    m.add_class::<PyUnifiedParserOutput>()?;
    m.add_class::<PyUnifiedParser>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    fn metadata(
        py: Python<'_>,
        vocab: impl IntoIterator<Item = (&'static str, u32)>,
    ) -> PyResult<Py<PyTokenizerMetadata>> {
        let token_to_id = vocab
            .into_iter()
            .map(|(token, token_id)| (token.to_string(), token_id))
            .collect::<HashMap<_, _>>();
        let special_ids = token_to_id.values().copied().collect();
        Py::new(py, PyTokenizerMetadata::new(token_to_id, special_ids))
    }

    fn tool(py: Python<'_>, name: &str) -> PyResult<Py<PyTool>> {
        let parameters = pythonize(
            py,
            &json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }),
        )?;
        Py::new(py, PyTool::new(name.to_string(), None, &parameters, None)?)
    }

    #[test]
    fn registry_lists_python_supported_unified_parsers() {
        assert_eq!(list_unified_parsers(), ["inkling", "minimax_m3"]);
    }

    #[test]
    fn minimax_m3_emits_reasoning_text_and_tool_events() {
        with_python(|py| {
            let tokenizer = metadata(py, [("<mm:think>", 256), ("</mm:think>", 257)])?;
            let tools = vec![tool(py, "get_weather")?];
            let mut parser = PyUnifiedParser::new(py, "minimax_m3", tools, tokenizer)?;
            parser.initialize(vec![256])?;

            let mut output = PyUnifiedParserOutput::new();
            parser.parse_into_output(
                "plan</mm:think>answer\
                 ]<]minimax[>[<tool_call>\
                 ]<]minimax[>[<invoke name=\"get_weather\">\
                 ]<]minimax[>[<city>Paris]<]minimax[>[</city>\
                 ]<]minimax[>[</invoke>\
                 ]<]minimax[>[</tool_call>",
                &mut output,
            )?;
            output.0.append(parser.finish()?.0);

            assert_eq!(
                output.0.events,
                vec![
                    UnifiedParserEvent::Reasoning("plan".to_string()),
                    UnifiedParserEvent::Text("answer".to_string()),
                    UnifiedParserEvent::ToolCall(ToolCallDelta {
                        tool_index: 0,
                        name: Some("get_weather".to_string()),
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    }),
                ]
            );
            PyResult::Ok(())
        })
        .unwrap();
    }

    #[test]
    fn inkling_uses_native_unified_parser() {
        with_python(|py| {
            let tokenizer = metadata(
                py,
                [
                    ("<|message_model|>", 200001),
                    ("<|content_text|>", 200004),
                    ("<|content_thinking|>", 200008),
                ],
            )?;
            let tools = vec![tool(py, "get_weather")?];
            let mut parser = PyUnifiedParser::new(py, "inkling", tools, tokenizer)?;
            parser.initialize(Vec::new())?;

            let mut output = PyUnifiedParserOutput::new();
            parser.parse_into_output(
                "<|content_thinking|>plan<|end_message|>\
                 <|content_text|>answer<|end_message|>\
                 <|content_invoke_tool_json|>\
                 {\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}\
                 <|end_message|>",
                &mut output,
            )?;
            output.0.append(parser.finish()?.0);

            assert_eq!(
                output.0.events,
                vec![
                    UnifiedParserEvent::Reasoning("plan".to_string()),
                    UnifiedParserEvent::Text("answer".to_string()),
                    UnifiedParserEvent::ToolCall(ToolCallDelta {
                        tool_index: 0,
                        name: Some("get_weather".to_string()),
                        arguments: String::new(),
                    }),
                    UnifiedParserEvent::ToolCall(ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    }),
                ]
            );
            PyResult::Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parser_rejects_unknown_name() {
        with_python(|py| {
            let tokenizer = metadata(py, [])?;
            let error = match PyUnifiedParser::new(py, "missing", Vec::new(), tokenizer) {
                Ok(_) => panic!("missing unified parser unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("unsupported unified parser `missing`"));
            PyResult::Ok(())
        })
        .unwrap();
    }
}
