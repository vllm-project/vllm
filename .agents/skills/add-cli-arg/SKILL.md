---
name: add-cli-arg
description: "Recipe for adding or changing a `vllm serve` / `vllm` CLI argument. Use when adding a flag to `EngineArgs` in `vllm/engine/arg_utils.py` or to the frontend args classes (`FrontendArgs`, `BaseFrontendArgs`, `BatchFrontendArgs`), exposing a config field on the command line, or changing an existing arg's type, default, or help text."
---

# Adding a CLI Argument

Every CLI arg is a field on a dataclass. The dataclass owns the type, default,
help text, and validation. `get_kwargs(cls)` in `arg_utils.py` derives the
argparse kwargs (type, choices, `BooleanOptionalAction`, JSON parsing, help)
from the field's annotation and attribute docstring, so a correct field
produces a correct flag with no extra code.

Pick the **owner** first; it decides the recipe:

| The arg configures | Owner | Recipe |
| ------------------ | ----- | ------ |
| The engine (model, cache, parallelism, scheduling, kernels, ...) | A config class in `vllm/config/` | [Engine args](#engine-args) |
| The API server only (HTTP, chat templates, tool parsers, LoRA modules, ...) | `BaseFrontendArgs` (shared) or `FrontendArgs` (server only) in `vllm/entrypoints/common/cli_args.py` | [Frontend args](#frontend-args) |
| `vllm run-batch` only | `BatchFrontendArgs` (search for `class BatchFrontendArgs`) | [Frontend args](#frontend-args) |

## Decide whether a flag is needed

Prefer automatic selection or a better default. If users must control it,
write down why existing controls are not enough (the `pr-checklist` skill asks
for this).

## Write the field

Same for every owner:

```python
foo_bar: int | None = Field(default=None, ge=0)
"""What it does, how to enable it, and the default behaviour.
This docstring is the `--help` text and the rendered CLI reference."""
```

- The owning classes are pydantic dataclasses, so validation lives on the
  field and applies to the Python API as well as the CLI. Express bounds and
  lengths as `pydantic.Field` constraints (`ge`, `gt`, `le`, `min_length`);
  use a validator or `__post_init__` only for rules that span fields.
- Mutable defaults use `Field(default_factory=...)`.
- When the generated kwargs don't fit (wrong parser, missing choices), fix the
  annotation, or extend `_compute_kwargs` in `arg_utils.py` so every class
  gets the fix.

## Engine args

Fields of a config exposed through a JSON arg (search `add_cli_args` for
`--compilation-config`, `--attention-config`, `--kernel-config` and the other
`--*-config` args) are already settable as `--x-config.field value`. Adding
the field to that config class is the whole change. Give such a field its own
flag only when most users will set it, as `--attention-backend` does for
`AttentionConfig.backend`; the steps below then apply as for any other field.

Otherwise `EngineArgs` **mirrors** the config field, in three places in
`arg_utils.py`:

1. **`EngineArgs` field**: the config's annotation verbatim, and the default
   read from the config class, never a literal, so the two cannot drift.

   ```python
   foo_bar: int | None = XConfig.foo_bar
   foo_list: list[str] = get_field(XConfig, "foo_list")  # default_factory
   ```

2. **`add_cli_args`**: in the existing argument group for that config, pass
   the `get_kwargs` output through untouched.

   ```python
   x_group.add_argument("--foo-bar", **x_kwargs["foo_bar"])
   ```

   The flag is the field name with dashes. When the field name is generic,
   prefix both the flag and the `EngineArgs` field with the config's area
   (`MambaConfig.backend` is `--mamba-backend` / `mamba_backend`).

3. **Config construction**: add `foo_bar=self.foo_bar` to the `XConfig(...)`
   call in `create_engine_config` or the matching `create_x_config` method.

## Frontend args

The field is the whole change: `BaseFrontendArgs.add_cli_args` registers every
field of the class as `--field-name`. Read the value from the parsed `args`
where the server is built.

- Kwargs that `get_kwargs` cannot express go in the class's
  `_customize_cli_kwargs` override, which calls `super()` first.
- The Rust frontend serves the same flags. A flag it does not implement must
  fail at startup or be a documented no-op; see `rust/AGENTS.md`.

## Check it

```bash
vllm serve --help=foo-bar     # flag, default, and help text render
.venv/bin/python -m pytest tests/engine/test_arg_utils.py -v
```

Add a test to `tests/engine/test_arg_utils.py` only for behaviour beyond the
field itself (custom parsing, aliases, cross-field validation). A plain field
is covered by the `get_kwargs` tests and needs none.

## Changing an existing arg

Change the owning field (type, default, docstring) and let the CLI follow.
Renames and removals keep the old flag working with a deprecation warning
that names the removal version; search `arg_utils.py` for
`DEPRECATION_MESSAGE` for the existing pattern.
