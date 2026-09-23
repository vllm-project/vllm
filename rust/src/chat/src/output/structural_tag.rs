// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Applies parser-built xgrammar structural-tag constraints.

use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputsParams,
};
use vllm_parser::output_grammar::{BuiltOutputGrammar, GrammarCoverage};
use vllm_text::TextRequest;
use xgrammar_structural_tag::StructuralTag;

use crate::{Error, Result};

/// Apply one parser-built output grammar to the prepared text request.
///
/// A parser-owned grammar replaces any structured output constraint the user
/// supplied, matching the Python frontend. `None` leaves the request untouched.
pub(crate) fn apply_output_grammar(
    request: &mut TextRequest,
    built: Option<BuiltOutputGrammar>,
) -> Result<()> {
    let Some(built) = built else {
        return Ok(());
    };
    let structural_tag = StructuralTag::new(built.format).to_json_string().map_err(|error| {
        Error::OutputGrammar {
            error: Box::new(error),
        }
    })?;

    // Overwrite any existing structured output settings with the structural tag constraint.
    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        backend: StructuredOutputBackend::Xgrammar,
        ..StructuredOutputsParams::structural_tag(structural_tag)
    });
    request.reasoning_ended = match built.coverage {
        GrammarCoverage::FromTokenZero => Some(true),
        GrammarCoverage::FinalOutputOnly => None,
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_engine_core_client::protocol::structured_outputs::{
        StructuredOutputBackend, StructuredOutputsParams,
    };
    use xgrammar_structural_tag::format::Format;

    use super::*;

    #[test]
    fn output_grammar_overwrites_answer_constraint() {
        let mut request = TextRequest::for_test();
        request.sampling_params.structured_outputs =
            Some(StructuredOutputsParams::json(json!({ "type": "object" })));

        let built = BuiltOutputGrammar::from_token_zero(Format::const_string("answer"));
        apply_output_grammar(&mut request, Some(built)).unwrap();

        assert_eq!(request.reasoning_ended, Some(true));
        let params = request.sampling_params.structured_outputs.unwrap();
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        let serialized = params.constraint.as_structural_tag().unwrap();
        let value: serde_json::Value = serde_json::from_str(serialized).unwrap();
        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "const_string");
        assert_eq!(value["format"]["value"], "answer");
    }
}
