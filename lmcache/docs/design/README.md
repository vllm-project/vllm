# LMCache Design Docs

This directory holds design documents for LMCache modules.

## Layout Convention

**`docs/design/` mirrors the `lmcache/` package tree.**

The design doc(s) for a given module live under the same relative path as the
module's source. To find design docs for code at `lmcache/<path>/`, look under
`docs/design/<path>/`.

Examples:

| Source module | Design doc location |
|---|---|
| `lmcache/cli/` | `docs/design/cli/` |
| `lmcache/cli/commands/ping.py` | `docs/design/cli/commands/ping.md` |
| `lmcache/cli/commands/bench/engine_bench/` | `docs/design/cli/commands/bench/engine_bench/` |
| `lmcache/v1/distributed/l2_adapters/` | `docs/design/v1/distributed/l2_adapters/` |

Not every module has a design doc — only modules whose design warrants prose
beyond what the code and docstrings already communicate. Missing directories
here simply mean "no standalone design doc exists yet."

## Guidelines for Adding Docs

- Place the doc at the path that matches the module it describes. A doc that
  spans multiple sibling submodules belongs at their common parent directory.
- Keep the file name descriptive of the topic, not the module (e.g.
  `query-command.md` under `docs/design/cli/commands/`, not `query.md`).
- When linking between design docs, use relative paths so links survive renames
  within this tree.
