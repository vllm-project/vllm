// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Function tools of a Responses request, including `namespace` tools.
//!
//! Mirrors the Python frontend (`get_function_tools` and
//! `build_responses_tool_call_name_map` in `vllm/tool_parsers/utils.py`,
//! `_resolve_named_tool_choice` in `responses/protocol.py`): the model sees
//! each namespaced function as one flat `<namespace>__<name>` function tool,
//! and a call of that flat name is reported with the function name and its
//! namespace. The flat spelling is defined only by [`ToolIdentity::flat_name`].

use std::borrow::Cow;
use std::collections::HashMap;

use itertools::Itertools as _;
use vllm_chat::ChatTool;

use super::types::{NamespacedTool, ResponsesTool};
use crate::error::{ApiError, bail_invalid_request};

const NAMESPACE_TOOL_SEPARATOR: &str = "__";

/// A function tool as the client names it: a namespace tool's function
/// carries its namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolIdentity<'a> {
    pub namespace: Option<&'a str>,
    pub name: &'a str,
}

impl<'a> ToolIdentity<'a> {
    /// The name the model sees: `<namespace>__<name>`, or the bare name.
    pub(super) fn flat_name(self) -> Cow<'a, str> {
        match self.namespace {
            Some(namespace) => Cow::Owned(format!(
                "{namespace}{NAMESPACE_TOOL_SEPARATOR}{}",
                self.name
            )),
            None => Cow::Borrowed(self.name),
        }
    }
}

/// Request-local map from the flat names of namespaced functions back to
/// their identities, for reporting calls and resolving `tool_choice`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ResponsesToolNames {
    /// Flat name of each namespaced function -> byte length of its namespace.
    /// Like Python, a later function with the same flat name wins.
    namespaced: HashMap<String, usize>,
}

impl ResponsesToolNames {
    /// Convert the request tools to the function tools the model is shown, in
    /// request order, together with the names to map calls back.
    pub(super) fn convert(tools: &[ResponsesTool]) -> (Vec<ChatTool>, Self) {
        let mut chat_tools = Vec::with_capacity(tools.len());
        let mut namespaced = HashMap::new();
        for tool in tools {
            match tool {
                ResponsesTool::Function(function) => {
                    chat_tools.push(ChatTool::from(function.clone()));
                }
                ResponsesTool::Namespace(namespace) => {
                    for NamespacedTool::Function(function) in &namespace.tools {
                        let identity = ToolIdentity {
                            namespace: Some(&namespace.name),
                            name: &function.name,
                        };
                        let flat_name = identity.flat_name().into_owned();
                        namespaced.insert(flat_name.clone(), namespace.name.len());
                        chat_tools.push(ChatTool {
                            name: flat_name,
                            ..ChatTool::from(function.clone())
                        });
                    }
                }
            }
        }
        (chat_tools, Self { namespaced })
    }

    /// The identity of a called tool name. Names of other tools pass through
    /// without a namespace.
    pub(super) fn decode<'a>(&self, called_name: &'a str) -> ToolIdentity<'a> {
        match self.namespaced.get(called_name) {
            Some(&namespace_len) => ToolIdentity {
                namespace: Some(&called_name[..namespace_len]),
                name: &called_name[namespace_len + NAMESPACE_TOOL_SEPARATOR.len()..],
            },
            None => ToolIdentity {
                namespace: None,
                name: called_name,
            },
        }
    }

    /// Resolve a named tool choice to the name the model sees.
    ///
    /// An exact tool name (including a flat `<namespace>__<name>`) wins;
    /// otherwise a function name defined in exactly one namespace selects
    /// that function.
    pub(super) fn resolve_choice(
        &self,
        name: String,
        tools: &[ResponsesTool],
    ) -> Result<String, ApiError> {
        let is_exact = self.namespaced.contains_key(&name)
            || tools.iter().any(
                |tool| matches!(tool, ResponsesTool::Function(function) if function.name == name),
            );
        if is_exact {
            return Ok(name);
        }
        let mut candidates = tools
            .iter()
            .filter_map(|tool| match tool {
                ResponsesTool::Namespace(namespace) => Some(namespace),
                ResponsesTool::Function(_) => None,
            })
            .flat_map(|namespace| {
                namespace
                    .tools
                    .iter()
                    .filter(|NamespacedTool::Function(function)| function.name == name)
                    .map(|_| {
                        ToolIdentity {
                            namespace: Some(&namespace.name),
                            name: &name,
                        }
                        .flat_name()
                        .into_owned()
                    })
            })
            .collect_vec();
        match candidates.len() {
            0 => bail_invalid_request!(
                param = "tool_choice",
                "Tool choice 'function' not found in 'tools' parameter."
            ),
            1 => Ok(candidates.pop().expect("one candidate")),
            _ => bail_invalid_request!(
                param = "tool_choice",
                "Tool choice '{name}' is ambiguous: it is defined in more than one namespace ({}). \
                 Select it by its full '<namespace>__<name>' name.",
                candidates.iter().join(", ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use thiserror_ext::AsReport as _;

    use super::*;

    fn request_tools(tools: serde_json::Value) -> Vec<ResponsesTool> {
        serde_json::from_value(tools).expect("deserialize tools")
    }

    fn namespace(name: &str, functions: &[&str]) -> serde_json::Value {
        json!({
            "type": "namespace",
            "name": name,
            "description": "Grouped tools.",
            "tools": functions
                .iter()
                .map(|function| json!({"type": "function", "name": function}))
                .collect::<Vec<_>>(),
        })
    }

    #[test]
    fn namespaced_functions_are_shown_flat_in_request_order() {
        let tools = request_tools(json!([
            {"type": "function", "name": "lookup", "description": "Top level."},
            {
                "type": "namespace",
                "name": "crm",
                "description": "CRM tools.",
                "tools": [{
                    "type": "function",
                    "name": "get_contact",
                    "description": "Get a contact.",
                    "parameters": {"type": "object"},
                    "strict": true
                }]
            }
        ]));

        let (shown, _) = ResponsesToolNames::convert(&tools);

        let names = shown.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>();
        assert_eq!(names, ["lookup", "crm__get_contact"]);
        assert_eq!(shown[1].description.as_deref(), Some("Get a contact."));
        assert_eq!(shown[1].parameters, json!({"type": "object"}));
        assert_eq!(shown[1].strict, Some(true));
    }

    #[test]
    fn custom_tools_inside_a_namespace_are_rejected() {
        let error = serde_json::from_value::<Vec<ResponsesTool>>(json!([{
            "type": "namespace",
            "name": "crm",
            "description": "CRM tools.",
            "tools": [{"type": "custom", "name": "raw"}]
        }]))
        .expect_err("custom tool inside a namespace");

        let report = error.to_report_string();
        assert!(report.contains("unknown variant `custom`"), "{report}");
    }

    #[test]
    fn called_flat_names_resolve_to_the_namespaced_function() {
        let (_, names) = ResponsesToolNames::convert(&request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("mcp__computer_use", &["get_app_state"]),
        ])));

        assert_eq!(
            names.decode("mcp__computer_use__get_app_state"),
            ToolIdentity {
                namespace: Some("mcp__computer_use"),
                name: "get_app_state",
            }
        );
        let bare = |name| ToolIdentity {
            namespace: None,
            name,
        };
        assert_eq!(names.decode("lookup"), bare("lookup"));
        assert_eq!(names.decode("get_app_state"), bare("get_app_state"));
    }

    #[test]
    fn named_tool_choice_resolves_exact_then_unique_function_names() {
        let tools = request_tools(json!([
            {"type": "function", "name": "lookup"},
            namespace("crm", &["get_contact", "lookup", "search"]),
            namespace("docs", &["search"]),
        ]));
        let (_, names) = ResponsesToolNames::convert(&tools);
        let resolve = |name: &str| names.resolve_choice(name.to_string(), &tools);

        assert_eq!(resolve("lookup").unwrap(), "lookup");
        assert_eq!(resolve("crm__lookup").unwrap(), "crm__lookup");
        assert_eq!(resolve("get_contact").unwrap(), "crm__get_contact");
        assert_eq!(resolve("docs__search").unwrap(), "docs__search");

        let invalid_tool_choice = |name: &str| {
            let error = resolve(name).expect_err("invalid tool choice");
            let ApiError::InvalidRequest { message, param } = error else {
                panic!("unexpected error: {error:?}");
            };
            assert_eq!(param, Some("tool_choice"));
            message
        };
        assert_eq!(
            invalid_tool_choice("search"),
            "Tool choice 'search' is ambiguous: it is defined in more than one namespace \
             (crm__search, docs__search). Select it by its full '<namespace>__<name>' name."
        );
        assert_eq!(
            invalid_tool_choice("missing"),
            "Tool choice 'function' not found in 'tools' parameter."
        );
    }
}
