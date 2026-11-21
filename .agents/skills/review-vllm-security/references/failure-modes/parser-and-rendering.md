# Parser and rendering complexity

- **Source and expansion bounds.** Limit nesting, references, branches,
  alternatives, tool definitions, and generated schema size before compilation.
  Check translated schemas and grammar forms as well as raw input.
- **Invalid and incomplete forms.** Check empty, malformed, recursive, and
  incomplete input on each backend. Confirm preprocessing and validation run
  before compiler or native calls for every supported schema format.
- **Repeated work.** Look for repeated copying and rescanning of growing
  state during rendering, token replacement, incremental parsing, and
  sanitization. Check streaming deltas, unmatched delimiters, nested quotes,
  and immediate rejection paths.
- **Pattern matching cost.** Review regex and grammar complexity on caller
  data and model output, including patterns used in error handling. Prefer
  bounded or linear algorithms where feasible. Check compile and match
  controls on every backend that can consume the input.
- **Timeout behavior.** Bound worker occupancy and work that continues after
  a deadline. Check repeated timeouts, cancelled callers, executor capacity,
  and event-loop responsiveness.
- **Streaming semantics.** Compare buffered parsing with chunks split across
  quotes, escapes, or delimiters. Keep parser state request-local. Preserve
  the distinction between quoted data and tool syntax through dispatch.
- **Executable output boundary.** Keep streaming fragments in parser state
  until a complete call and its required arguments are validated. Emit a call
  for dispatch once, and handle invalid or incomplete terminal output without
  executing it.
- **Template execution.** Apply trust policy, permitted template context, and
  execution limits to model-provided templates and alternate renderers.
  Review template argument preparation as well as the render itself.

## Published advisory examples

Bound worker occupancy and apply
compilation controls across supported backends.

- [Template rendering resources](https://github.com/vllm-project/vllm/security/advisories/GHSA-542m-9234-8gvf)
- [Structured-output regex compilation](https://github.com/vllm-project/vllm/security/advisories/GHSA-48jh-3gj7-fg8v)
