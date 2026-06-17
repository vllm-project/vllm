//! Thin PyO3 bindings for `vllm_tool_parser`.
//!
//! This crate exposes the Rust tool parser trait and data shapes to Python
//! while keeping parser state, grammar, and schema-aware argument conversion in
//! Rust. Python callers should use this module as a typed bridge and keep any
//! vLLM protocol adaptation outside the binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_tool_parser::{Tool, ToolCallDelta, ToolParser, ToolParserOutput};

macro_rules! tool_parser_factory {
    ($($parser:ident),+ $(,)?) => {
        fn create_tool_parser(
            name: &str,
            tools: &[Tool],
        ) -> PyResult<Box<dyn ToolParser>> {
            match name {
                $(
                    stringify!($parser) => {
                        <vllm_tool_parser::$parser as ToolParser>::create(tools)
                    }
                )+
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported tool parser `{name}`"
                    )));
                }
            }
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
        }
    };
}

// Export a tool parser to Python by registering it here.
tool_parser_factory! {
    MinimaxM3ToolParser,

    // Below are the parsers just for testing purposes on Python side.
    DeepSeekV4ToolParser,
    KimiK2ToolParser,
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
                "failed to convert tool parameters from JSON to Python: {error}"
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
    #[new]
    #[pyo3(signature = (tool_index, name, arguments))]
    fn new(tool_index: usize, name: Option<String>, arguments: String) -> Self {
        Self(ToolCallDelta {
            tool_index,
            name,
            arguments,
        })
    }

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
    name = "ToolParserOutput",
    module = "vllm._rust_tool_parser",
    skip_from_py_object
)]
#[derive(Clone)]
struct PyToolParserOutput(ToolParserOutput);

#[pymethods]
impl PyToolParserOutput {
    #[new]
    #[pyo3(signature = (normal_text="", calls=None))]
    fn new(py: Python<'_>, normal_text: &str, calls: Option<Vec<Py<PyToolCallDelta>>>) -> Self {
        let calls =
            calls.unwrap_or_default().iter().map(|call| call.borrow(py).0.clone()).collect();
        Self(ToolParserOutput {
            normal_text: normal_text.to_owned(),
            calls,
        })
    }

    #[getter]
    fn normal_text(&self) -> &str {
        &self.0.normal_text
    }

    #[getter]
    fn calls(&self) -> Vec<PyToolCallDelta> {
        self.0.calls.iter().cloned().map(PyToolCallDelta).collect()
    }

    fn append(&mut self, other: PyRef<'_, PyToolParserOutput>) {
        self.0.append(other.0.clone());
    }

    fn coalesce_calls(&self) -> Self {
        Self(self.0.clone().coalesce_calls())
    }
}

#[pyclass(name = "ToolParser", module = "vllm._rust_tool_parser", unsendable)]
struct PyToolParser(Box<dyn ToolParser>);

impl PyToolParser {
    fn parse_into_output(&mut self, chunk: &str, output: &mut PyToolParserOutput) -> PyResult<()> {
        self.0
            .parse_into(chunk, &mut output.0)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }
}

#[pymethods]
impl PyToolParser {
    #[new]
    fn new(py: Python<'_>, parser_name: &str, tools: Vec<Py<PyTool>>) -> PyResult<Self> {
        let tools = tools.iter().map(|tool| tool.borrow(py).0.clone()).collect::<Vec<_>>();
        create_tool_parser(parser_name, &tools).map(Self)
    }

    fn parse_into(
        &mut self,
        chunk: &str,
        mut output: PyRefMut<'_, PyToolParserOutput>,
    ) -> PyResult<()> {
        self.parse_into_output(chunk, &mut output)
    }

    fn finish(&mut self) -> PyResult<PyToolParserOutput> {
        self.0
            .finish()
            .map(PyToolParserOutput)
            .map_err(|error| PyValueError::new_err(error.to_report_string()))
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.0.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.0.tool_call_id(tool_index)
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
        r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="create_order">
<｜DSML｜parameter name="user_id" string="false">42</｜DSML｜parameter>
<｜DSML｜parameter name="shipping" string="false">{"city":"Singapore","zip":18956}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#
            .to_owned()
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
            let mut parser = PyToolParser::new(py, "DeepSeekV4ToolParser", vec![tool])?;
            assert!(parser.preserve_special_tokens());

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
    fn parser_exposes_model_emitted_tool_call_ids() {
        with_python(|py| {
            let tool = make_py_tool(py)?;
            let mut parser = PyToolParser::new(py, "KimiK2ToolParser", vec![tool])?;

            let input = "<|tool_calls_section_begin|>\
                <|tool_call_begin|>functions.create_order:0<|tool_call_argument_begin|>\
                {\"user_id\":42}<|tool_call_end|>\
                <|tool_calls_section_end|>";
            let mut output = PyToolParserOutput::new(py, "", None);
            parser.parse_into_output(input, &mut output)?;

            assert_eq!(parser.tool_call_id(0), Some("functions.create_order:0"));
            assert_eq!(parser.tool_call_id(1), None);
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
