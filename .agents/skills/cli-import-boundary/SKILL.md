---
name: cli-import-boundary
description: Keep the vllm CLI import-light. Use when adding or modifying CLI commands, EngineArgs or config fields, environment overrides or monkey patches, or anything imported before command dispatch, so vllm --help and vllm serve --help stay fast and never import torch.
---

# vLLM CLI Import Boundary

`vllm --help`, `vllm serve --help`, `--help=all`, and `--help=<Section>` render
without importing torch, a concrete platform, or the runtime engine graph.
Breaking this silently costs every CLI invocation seconds of import time. The
contract is enforced by tests that fail before users notice:

```bash
pytest tests/entrypoints/launchers/test_cli_imports.py tests/test_env_override_lifecycle.py
```

The first renders help in a subprocess with torch and the platform probes
blocked; the second proves `vllm.env_override` imports without torch and that
its post-import patches register, retry after failures, and fire on target
import.

## How dispatch stays light

- `vllm/entrypoints/cli/main.py` picks the command from argv before importing
  anything heavy: `_selected_command()` chooses, `_build_parser()` builds only
  what that selection needs, and `_load_command()` imports just the selected
  command's module.
- `vllm/entrypoints/cli/_utils.py` holds the `CLI_COMMANDS` metadata registry,
  `cli_env_setup()` (the CLI-only multiprocessing spawn default), and
  `is_serve_help()`.
- `vllm/env_override.py` is the single owner of environment setup, in two
  phases: `apply_pre_torch_environment()` runs at `import vllm`, and every
  torch-dependent patch is registered with `_register_post_import_patch` to
  fire when its exact target module (for example `torch._inductor.config`) is
  imported. Patches are never applied at call sites.
- In `vllm/engine/arg_utils.py`, `IS_SERVE_HELP` gates plugin loading and
  platform parser mutation, so rendering serve help never initializes runtime
  state.

## Rules when changing this area

1. New CLI command: register it in `CLI_COMMANDS`; import its implementation
   only at dispatch, never at module scope under `vllm/entrypoints/cli/`.
2. New `EngineArgs` or config field with a torch type: keep the real type under
   `TYPE_CHECKING` with a runtime `object` alias, and union the field with its
   `Literal` alias; `arg_utils` resolves the `Literal` before the runtime shim,
   so argparse option generation is unchanged (see `ModelConfig.dtype`).
3. New environment variable, warning filter, or monkey patch: put it in
   `vllm/env_override.py` as a pre-torch setting or a post-import callback.
4. No module-scope `import torch`, concrete `vllm.platforms` resolution, or
   engine imports anywhere reachable from parser construction.
5. Run the two test files above before pushing. If one fails, move the import
   to its use site or behind a post-import callback; do not relax the test.
