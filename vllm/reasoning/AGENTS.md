# vllm/reasoning

Every `--reasoning-parser` name resolves here. The parser has two consumers:
`ParserManager.get_parser` composes it with the tool parser into the unified
`Parser` the serving layer uses, and the structured-output manager
(`vllm/v1/structured_output/__init__.py`) instantiates it directly and calls
`is_reasoning_end` / `is_reasoning_end_streaming` every decode step to decide
when the grammar bitmask starts applying. The `ReasoningParser` interface
itself is legacy: it is kept so existing in-tree parsers and out-of-tree
plugins keep working, not as the base for new ones.

## Layout

- `abs_reasoning_parsers.py` — `ReasoningParser` (`extract_reasoning`,
  `extract_reasoning_streaming`, `is_reasoning_end`,
  `is_reasoning_end_streaming`, `extract_content_ids`, `adjust_request`) and
  `ReasoningParserManager`.
- `__init__.py` — `_REASONING_PARSERS_TO_REGISTER`, the name → (file, class)
  table that lazily registers every built-in parser.
- `basic_parsers.py` — `BaseThinkingReasoningParser`, the base for parsers
  that only need a start/end token pair.
- `<model>_reasoning_parser.py` — one file per family. Either a subclass of a
  `ParserReasoningAdapter` generated from a `vllm/parser/<model>.py` engine
  grammar (`mistral_reasoning_parser.py`, `kimi_k2_reasoning_parser.py`; the
  `_engine_` suffix is not applied consistently), or a self-contained
  `ReasoningParser` subclass.

## Guidelines

- Do not change the `ReasoningParser` interface in `abs_reasoning_parsers.py`.
  Out-of-tree plugins loaded through
  `ReasoningParserManager.import_reasoning_parser` subclass it. New
  capabilities go on `Parser` in `vllm/parser/`; if a legacy parser needs the
  same capability, `DelegatingParser` supplies the default.
- For a new model, implement a unified `Parser` in `vllm/parser/` first, then
  expose it here as a thin adapter subclass plus an entry in
  `_REASONING_PARSERS_TO_REGISTER`.
- The `_REASONING_PARSERS_TO_REGISTER` key is what users pass on the command
  line; treat renames as breaking.
- Keep adapters thin: registration, `adjust_request` tweaks, and class
  attributes. Parsing logic belongs in `vllm/parser/`.
- Keep a model-specific fix inside that model's `<model>_reasoning_parser.py`,
  or its grammar in `vllm/parser/<model>.py`, where possible. Touch
  `abs_reasoning_parsers.py` or `basic_parsers.py` only for behavior every
  parser needs, and say so in the PR.
- `is_reasoning_end_streaming` runs on the engine core for every decode step
  of every structured-output request. Work on `delta_ids`; re-scanning the
  full sequence makes the step cost grow with sequence length.
- The structured-output gate and the frontend's extraction must agree on
  where reasoning ends, whether the turn continues with content or a tool
  call. Test the end-detection methods with the same inputs as extraction.
- Comment only where the code alone would leave a reader with a question —
  a request tweak in `adjust_request`, a marker handled unusually, a model
  quirk being worked around. Keep it minimal and concise.
