# vllm/parser

Model-output parsing for the serving layer. `Parser` (`abstract_parser.py`)
unifies reasoning extraction and tool-call extraction behind one object; the
serving layer calls `parse` / `parse_delta` and `is_reasoning_end` on it.

## Layout

- `abstract_parser.py` — `Parser`, and `DelegatingParser` for wrapping a legacy
  `ReasoningParser` + `ToolParser` pair.
- `parser_manager.py` — resolves parser names to classes.
- `engine/` — the declarative parser engine. A `ParserEngineConfig`
  (`parser_engine_config.py`) is a set of terminals, transitions and content
  events; `parser_engine.py` and `streaming_parser_engine.py` drive it for the
  non-streaming and streaming paths; `token_id_scanner.py` pre-lexes
  special-token ids and `incremental_lexer.py` lexes text; `adapters.py`
  exposes an engine through the legacy reasoning/tool parser interfaces.
- `<model>.py` — one file per model family. Most subclass `ParserEngine`
  (`deepseek_v4.py`, `qwen3.py`, ...); a few are `DelegatingParser` wrappers
  over legacy parsers (`harmony.py`, `kimi_k3.py`).

## Guidelines

- A new parser either subclasses `ParserEngine` with a declarative config, or
  subclasses `Parser` and implements `parse` / `parse_delta` directly. Use the
  engine when the format is expressible as terminals and transitions, and
  subclass an existing config when it is a variant of one (`seed_oss.py`,
  `ling3.py`). `DelegatingParser` exists for backward compatibility with the
  separate `ReasoningParser` / `ToolParser` classes; do not use it for new
  formats.
- Keep a model-specific fix inside that model's `<model>.py` where possible.
  Change `engine/` or `abstract_parser.py` only when the behavior is wrong for
  every parser, and say so in the PR.
- Text emitted in a state with no `content_events` entry is discarded. Make
  that deliberate when adding a state.
- Special tokens reach the engine as token ids, not text. A text terminal that
  spans a special token never matches in streaming; list it in
  `token_id_terminals`.
- Comment only where the code alone would leave a reader with a question —
  a non-obvious transition, a deliberate swallow, a model quirk being worked
  around. Keep it minimal and concise.

## New model checklist

Before calling a new parser done, check each of these:

- Test every composition the server can build: both parsers, reasoning parser
  alone (without `--enable-auto-tool-choice` no tool parser is composed), and
  tool parser alone. In each, streaming at chunk sizes 1 and >1 must match
  `parse()`, whitespace included.
- Check the markers with the real tokenizer. Text markers that are not special
  tokens stay out of `token_id_terminals`, and `reasoning_start_str` /
  `reasoning_end_str` must tokenize the same alone as inside generated text:
  the thinking-token budget matches them by token ids.
- Choose `wait_for_reasoning` deliberately: it decides when structured output
  starts constraining. If the model may skip reasoning, waiting for an end
  marker it never emits leaves the output unconstrained.
- Compare with the chat template. Newlines the template puts around a marker
  belong to the marker's spellings, or they leak into reasoning and content and
  break multi-turn round-trips.
- Add a builder to `trace_builder._BUILDERS`; the replay suites fail at
  collection without one.
- Run a real streaming agent loop (multi-turn, parallel tool calls) against a
  served model, and check that no marker leaks into `content`.
