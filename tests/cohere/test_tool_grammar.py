# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

from test_const import C2_grammar, C3_grammar, tool_schema_1

from vllm.cohere.guided_decoding.tool_grammar import collect_tool_schema_v2

st = time.time()


def validate_tool_grammar() -> None:
    response_schema_c3 = collect_tool_schema_v2("Cohere2ForCausalLM", tool_schema_1)
    response_schema_c2 = collect_tool_schema_v2("CohereForCausalLM", tool_schema_1)
    assert response_schema_c3 == C3_grammar
    assert response_schema_c2 == C2_grammar

    print("Tool grammar validation passed.")


if __name__ == "__main__":
    validate_tool_grammar()
