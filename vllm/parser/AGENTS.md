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
