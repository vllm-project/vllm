use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_tool_parser::{
    DeepSeekV3ToolParser, DeepSeekV4ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser,
    Gemma4ToolParser, Glm45MoeToolParser, Glm47MoeToolParser, HermesToolParser, HyV3ToolParser,
    KimiK2ToolParser, Llama3JsonToolParser, MinimaxM2ToolParser, MinimaxM3ToolParser,
    MistralToolParser, Qwen3CoderToolParser, Qwen3XmlToolParser, Tool, ToolCallDelta, ToolParser,
    ToolParserOutput,
};

#[pyclass(name = "Tool", module = "vllm._rust_tool_parser", skip_from_py_object)]
#[derive(Clone)]
struct PyTool {
    inner: Tool,
}

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
        Ok(Self {
            inner: Tool {
                name,
                description,
                parameters,
                strict,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[getter]
    fn parameters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize(py, &self.inner.parameters).map(Bound::unbind).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert tool parameters from JSON to Python: {error}"
            ))
        })
    }

    #[getter]
    fn strict(&self) -> Option<bool> {
        self.inner.strict
    }
}

#[pyclass(
    name = "ToolCallDelta",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolCallDelta {
    inner: ToolCallDelta,
}

#[pymethods]
impl PyToolCallDelta {
    #[new]
    #[pyo3(signature = (tool_index, name, arguments))]
    fn new(tool_index: usize, name: Option<String>, arguments: String) -> Self {
        Self {
            inner: ToolCallDelta {
                tool_index,
                name,
                arguments,
            },
        }
    }

    #[getter]
    fn tool_index(&self) -> usize {
        self.inner.tool_index
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    #[getter]
    fn arguments(&self) -> &str {
        &self.inner.arguments
    }
}

#[pyclass(
    name = "ToolParserOutput",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolParserOutput {
    inner: ToolParserOutput,
}

impl PyToolParserOutput {
    fn from_inner(inner: ToolParserOutput) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyToolParserOutput {
    #[new]
    #[pyo3(signature = (normal_text="", calls=None))]
    fn new(py: Python<'_>, normal_text: &str, calls: Option<Vec<Py<PyToolCallDelta>>>) -> Self {
        let calls = calls
            .unwrap_or_default()
            .iter()
            .map(|call| call.borrow(py).inner.clone())
            .collect();
        Self {
            inner: ToolParserOutput {
                normal_text: normal_text.to_owned(),
                calls,
            },
        }
    }

    #[getter]
    fn normal_text(&self) -> &str {
        &self.inner.normal_text
    }

    #[getter]
    fn calls(&self) -> Vec<PyToolCallDelta> {
        self.inner
            .calls
            .iter()
            .cloned()
            .map(|inner| PyToolCallDelta { inner })
            .collect()
    }

    fn append(&mut self, other: PyRef<'_, PyToolParserOutput>) {
        self.inner.append(other.inner.clone());
    }

    fn coalesce_calls(&self) -> Self {
        Self::from_inner(self.inner.clone().coalesce_calls())
    }
}

#[pyclass(name = "ToolParser", module = "vllm._rust_tool_parser", unsendable)]
struct PyToolParser {
    parser: Box<dyn ToolParser>,
}

impl PyToolParser {
    fn from_name(name: &str, tools: &[Tool]) -> PyResult<Self> {
        let parser = match name {
            "deepseek_v3" => DeepSeekV3ToolParser::create(tools),
            "deepseek_v31" => DeepSeekV31ToolParser::create(tools),
            "deepseek_v32" => DeepSeekV32ToolParser::create(tools),
            "deepseek_v4" => DeepSeekV4ToolParser::create(tools),
            "gemma4" => Gemma4ToolParser::create(tools),
            "glm45" => Glm45MoeToolParser::create(tools),
            "glm47" => Glm47MoeToolParser::create(tools),
            "hermes" => HermesToolParser::create(tools),
            "hy_v3" => HyV3ToolParser::create(tools),
            "kimi_k2" => KimiK2ToolParser::create(tools),
            "llama3_json" | "llama4_json" => Llama3JsonToolParser::create(tools),
            "minimax_m2" => MinimaxM2ToolParser::create(tools),
            "minimax_m3" => MinimaxM3ToolParser::create(tools),
            "mistral" => MistralToolParser::create(tools),
            "qwen3_xml" => Qwen3XmlToolParser::create(tools),
            "qwen3_coder" => Qwen3CoderToolParser::create(tools),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported tool parser `{name}`"
                )));
            }
        }
        .map_err(|error| PyValueError::new_err(error.to_report_string()))?;

        Ok(Self { parser })
    }

    fn parse_into_output(&mut self, chunk: &str, output: &mut PyToolParserOutput) -> PyResult<()> {
        self.parser
            .parse_into(chunk, &mut output.inner)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }
}

#[pymethods]
impl PyToolParser {
    #[new]
    fn new(py: Python<'_>, parser_name: &str, tools: Vec<Py<PyTool>>) -> PyResult<Self> {
        let tools = tools.iter().map(|tool| tool.borrow(py).inner.clone()).collect::<Vec<_>>();
        Self::from_name(parser_name, &tools)
    }

    fn parse_into(
        &mut self,
        chunk: &str,
        mut output: PyRefMut<'_, PyToolParserOutput>,
    ) -> PyResult<()> {
        self.parse_into_output(chunk, &mut output)
    }

    fn finish(&mut self) -> PyResult<PyToolParserOutput> {
        self.parser
            .finish()
            .map(PyToolParserOutput::from_inner)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }

    fn reset(&mut self) -> String {
        self.parser.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }
}

#[pymodule]
fn _rust_tool_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTool>()?;
    m.add_class::<PyToolCallDelta>()?;
    m.add_class::<PyToolParserOutput>()?;
    m.add_class::<PyToolParser>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const NS: &str = "]<]minimax[>[";

    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    fn tool_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "user_id": {"type": "integer"},
                "shipping": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "integer"}
                    }
                }
            }
        })
    }

    fn build_call() -> String {
        format!(
            "{NS}<tool_call>\n\
             {NS}<invoke name=\"create_order\">\
             {NS}<user_id>42{NS}</user_id>\
             {NS}<shipping>\
             {NS}<city>Singapore{NS}</city>\
             {NS}<zip>018956{NS}</zip>\
             {NS}</shipping>\
             {NS}</invoke>\n\
             {NS}</tool_call>"
        )
    }

    fn make_py_tool(py: Python<'_>) -> PyResult<Py<PyTool>> {
        let parameters = pythonize(py, &tool_schema()).map_err(|error| {
            PyValueError::new_err(format!(
                "failed to convert test schema from JSON to Python: {error}"
            ))
        })?;
        Py::new(
            py,
            PyTool::new(
                "create_order".to_owned(),
                Some("Create an order".to_owned()),
                &parameters,
                None,
            )?,
        )
    }

    #[test]
    fn tool_round_trips_typed_fields() {
        with_python(|py| {
            let tool = make_py_tool(py)?;
            let borrowed = tool.borrow(py);
            assert_eq!(borrowed.name(), "create_order");
            assert_eq!(borrowed.description(), Some("Create an order"));
            assert_eq!(borrowed.strict(), None);

            let parameters = borrowed.parameters(py)?;
            let parameters = depythonize::<Value>(parameters.bind(py))?;
            assert_eq!(parameters, tool_schema());
            PyResult::Ok(())
        })
        .unwrap();
    }

    #[test]
    fn output_append_and_coalesce_calls() {
        with_python(|py| {
            let first = Py::new(
                py,
                PyToolCallDelta::new(0, Some("create_order".to_owned()), "{\"a\"".to_owned()),
            )?;
            let second = Py::new(py, PyToolCallDelta::new(0, None, ":1}".to_owned()))?;
            let mut output = PyToolParserOutput::new(py, "text", Some(vec![first]));
            let other = Py::new(py, PyToolParserOutput::new(py, "", Some(vec![second])))?;
            output.append(other.borrow(py));

            let coalesced = output.coalesce_calls();
            assert_eq!(coalesced.normal_text(), "text");
            let calls = coalesced.calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_index(), 0);
            assert_eq!(calls[0].name(), Some("create_order"));
            assert_eq!(calls[0].arguments(), "{\"a\":1}");
            PyResult::Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parser_parse_finish_and_preserve_special_tokens() {
        with_python(|py| {
            let tool = make_py_tool(py)?;
            let mut parser = PyToolParser::new(py, "minimax_m3", vec![tool])?;
            assert!(!parser.preserve_special_tokens());

            let mut output = PyToolParserOutput::new(py, "", None);
            parser.parse_into_output(&build_call(), &mut output)?;
            let finish = Py::new(py, parser.finish()?)?;
            output.append(finish.borrow(py));
            let output = output.coalesce_calls();

            assert_eq!(output.normal_text(), "");
            let calls = output.calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name(), Some("create_order"));
            assert_eq!(
                serde_json::from_str::<Value>(calls[0].arguments()).unwrap(),
                json!({
                    "user_id": 42,
                    "shipping": {
                        "city": "Singapore",
                        "zip": 18956
                    }
                })
            );

            assert_eq!(parser.reset(), "");
            PyResult::Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parser_errors_for_unknown_name() {
        with_python(|py| {
            let tool = make_py_tool(py)?;
            let error = match PyToolParser::new(py, "missing", vec![tool]) {
                Ok(_) => panic!("missing parser name unexpectedly succeeded"),
                Err(error) => error,
            };
            let message = format!("{error}");
            assert!(message.contains("unsupported tool parser `missing`"));
            PyResult::Ok(())
        })
        .unwrap();
    }
}
