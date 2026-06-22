<!-- markdownlint-disable MD024 -->
# Tool-Call Grammar (Guided Decoding)

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) section 4.6 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) — [Guided Generation](../feature_matrix.md#guided-generation)

Pure CPU unit suite over the EBNF tool-call grammar emitted for guided
decoding. No model, no GPU, no engine launch — it builds the grammar string
in-process and asserts on its shape. Guards the whitespace-ambiguity
regression from #849, where the top-level `tools` array rule placed two
nullable `ws*` stars adjacent (`tool ws ("," ws tool)* ws "]"`). For the common
single-tool case that collapses to `tool ws ws "]"`, letting a long whitespace
run be partitioned arbitrarily between the two `ws*`. xgrammar's Earley parser
then accumulates ambiguous states it never collapses, turning
`fill_next_token_bitmask` into an O(n^2) blowup that pins EngineCore on one CPU
core and wedges the serving replica.

<details>
<summary>Test case 1: Tool array rule has no ambiguous adjacent whitespace</summary>

## How it runs

1. In-process CPU unit tests (no server, no GPU) collected by the `build-cpu` /
   `test-cpu` CI suite via `pytest -v -s cohere/cpu`. Each builds the tool
   grammar from a small inline tool schema and asserts on the emitted EBNF
   string.
   - [`tests/cohere/cpu/test_tool_grammar_whitespace.py`](../../../../tests/cohere/cpu/test_tool_grammar_whitespace.py)
2. **Both** EBNF builders are exercised so a regression in either turns CI red:
   `collect_tool_schema_v2` (guided-decoding path, checked for both
   `CohereForCausalLM` and `Cohere2ForCausalLM`) and `collect_tool_schema`
   (the architecture-independent live chat structural-tag path used in prod).
   - [`vllm/cohere/guided_decoding/tool_grammar.py`](../../../../vllm/cohere/guided_decoding/tool_grammar.py) — `collect_tool_schema_v2`
   - [`vllm/reasoning/cohere_command_reasoning_parser.py`](../../../../vllm/reasoning/cohere_command_reasoning_parser.py) — `collect_tool_schema`
3. The same assertions are also available as a standalone dev helper (with a
   `__main__` golden-grammar check) that is not part of the CI suite.
   - [`tests/cohere/test_tool_grammar.py`](../../../../tests/cohere/test_tool_grammar.py)

## Checks

1. The emitted `tools` rule equals the **unambiguous** form
   `tools ::= ws "[" ws tool (ws "," ws tool)* ws "]" ws` for **both builders**
   (exactly one nullable `ws` between adjacent tokens).
   - `test_guided_decoding_builder_has_no_ambiguous_adjacent_whitespace`
   - `test_structural_tag_builder_has_no_ambiguous_adjacent_whitespace`
2. The two **ambiguous adjacent-`ws`** forms (`tool ws ("," ws tool)*` and
   `("," ws tool)*  ws "]"`) are absent from both builders' grammars.
   - `test_guided_decoding_builder_has_no_ambiguous_adjacent_whitespace`
   - `test_structural_tag_builder_has_no_ambiguous_adjacent_whitespace`

## Measurements

N/A. Pure assertion-only unit tests with no CI-uploaded artifacts.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

Pure CPU unit suite — no model inputs, no GPU, no engine launch. Input,
Quantization, Hardware, and vLLM Feature axes are not exercised; entries are
intentionally blank. The suite gates the EBNF whitespace contract of the
tool-call grammar builders.

1. **Input**:
2. **Cohere Feature**: Guided Generation (compatible)
   - [`tests/cohere/cpu/test_tool_grammar_whitespace.py`](../../../../tests/cohere/cpu/test_tool_grammar_whitespace.py)
3. **Model Architecture**: C4 Arch (compatible), C5 Arch (compatible)
   - [`tests/cohere/cpu/test_tool_grammar_whitespace.py`](../../../../tests/cohere/cpu/test_tool_grammar_whitespace.py) — `collect_tool_schema_v2` checked for `CohereForCausalLM` and `Cohere2ForCausalLM`
4. **Quantization**:
5. **Hardware**:
6. **vLLM Feature**:

## Implementation

Primary test: [`tests/cohere/cpu/test_tool_grammar_whitespace.py`](../../../../tests/cohere/cpu/test_tool_grammar_whitespace.py) (CI-collected via `pytest -v -s cohere/cpu`)
Dev helper: [`tests/cohere/test_tool_grammar.py`](../../../../tests/cohere/test_tool_grammar.py)
Runtime path: [`vllm/cohere/guided_decoding/tool_grammar.py`](../../../../vllm/cohere/guided_decoding/tool_grammar.py),
[`vllm/reasoning/cohere_command_reasoning_parser.py`](../../../../vllm/reasoning/cohere_command_reasoning_parser.py)
Code notes: [Guided Decoding: Structural Tags + Tool Grammar](../../code_notes/models-and-inference.md#4-guided-decoding-structural-tags--tool-grammar)

### Setup

1. No env vars, engine flags, or GPU — the tests call the builders directly
   with a small inline tool schema and inspect the returned EBNF string.
2. `collect_tool_schema_v2` is checked for both Cohere text architectures
   (`CohereForCausalLM`, `Cohere2ForCausalLM`); `collect_tool_schema` is
   architecture-independent and checked once.

</details>
