# vllm/tool_parsers

Every `--tool-call-parser` name resolves here. `ParserManager.get_parser`
composes the tool parser chosen by name with the reasoning parser into the
unified `Parser` the serving layer uses. The `ToolParser` interface itself is
legacy: it is kept so existing in-tree parsers and out-of-tree plugins keep
working, not as the base for new ones.

## Layout

- `abstract_tool_parser.py` — `ToolParser` (`extract_tool_calls`,
  `extract_tool_calls_streaming`, `adjust_request`) and `ToolParserManager`.
- `__init__.py` — `_TOOL_PARSERS_TO_REGISTER`, the name → (file, class) table
  that lazily registers every built-in parser.
- `structural_tag_registry.py` — structural-tag builders used to constrain
  tool-call output for `tool_choice="required"` / named tools.
- `<model>_tool_parser.py` — one file per family. Either a subclass of a
  `ParserToolAdapter` generated from a `vllm/parser/<model>.py` engine grammar
  (`mistral_tool_parser.py`, `kimi_k2_tool_parser.py`; the `_engine_` suffix
  is not applied consistently), or a self-contained `ToolParser` subclass.

## Guidelines

- Do not change the `ToolParser` interface in `abstract_tool_parser.py`.
  Out-of-tree plugins loaded through `ToolParserManager.import_tool_parser`
  subclass it. New capabilities go on `Parser` in `vllm/parser/`; if a legacy
  parser needs the same capability, `DelegatingParser` supplies the default.
- For a new model, implement a unified `Parser` in `vllm/parser/` first, then
  expose it here as a thin adapter subclass plus an entry in
  `_TOOL_PARSERS_TO_REGISTER`.
- The `_TOOL_PARSERS_TO_REGISTER` key is what users pass on the command line;
  treat renames as breaking.
- Keep adapters thin: registration, `adjust_request` tweaks, and class
  attributes. Parsing logic belongs in `vllm/parser/`.
- Keep a model-specific fix inside that model's `<model>_tool_parser.py`, or
  its grammar in `vllm/parser/<model>.py`, where possible. Touch
  `abstract_tool_parser.py` or `structural_tag_registry.py` only for behavior
  every parser needs, and say so in the PR.
- Set `structural_tag_model` only to a key in `SUPPORTED_STRUCTURAL_TAG_MODELS`
  (an xgrammar builtin, or one registered with `register_vllm_structural_tag`).
  It only takes effect under `VLLM_ENFORCE_STRICT_TOOL_CALLING`, where it also
  disables the generic required/named handling for that parser.
- Comment only where the code alone would leave a reader with a question —
  a request tweak in `adjust_request`, a marker handled unusually, a model
  quirk being worked around. Keep it minimal and concise.
