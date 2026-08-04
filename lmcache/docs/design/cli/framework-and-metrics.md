# LMCache CLI Framework & Metrics System Design

**Status:** Proposal  |  **Date:** 2026-03-14

## Scope

This document covers the **CLI framework** (pluggable command discovery) and the
**hierarchical metrics logging system**. It is the implementation plan for Phase 1
of the [CLI design](commands.md), minus the actual server/ping/describe commands
(those come later). A `lmcache mock` command is included as a working example.

---

## 1. Explicit Command Registration

### Goal

Adding a new subcommand (e.g., `lmcache describe`) requires:

1. Creating a new file in `lmcache/cli/commands/` with a `BaseCommand` subclass.
2. Adding one import + one entry to `ALL_COMMANDS` in `commands/__init__.py`.

### Mechanism

```python
# lmcache/cli/commands/my_cmd.py
from lmcache.cli.commands.base import BaseCommand

class MyCommand(BaseCommand):
    def name(self) -> str:
        return "my-cmd"

    def help(self) -> str:
        return "Short help text."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--flag", ...)

    def execute(self, args) -> None:
        ...  # command logic
```

```python
# lmcache/cli/commands/__init__.py  (add import + list entry)
from lmcache.cli.commands.my_cmd import MyCommand

ALL_COMMANDS: list[BaseCommand] = [
    ...,
    MyCommand(),
]
```

`BaseCommand` enforces that all four abstract methods (`name`, `help`,
`add_arguments`, `execute`) are implemented — instantiation fails otherwise.
The concrete `register()` method (inherited, not typically overridden) wires
everything up automatically.

### How command discovery works

1. `lmcache <cmd> ...` invokes `main()` in `main.py`.
2. `main.py` imports `ALL_COMMANDS` from `commands/__init__.py`.
3. At import time, `__init__.py` imports each command class and instantiates
   it into the `ALL_COMMANDS` list.  Instantiation validates that all abstract
   methods are implemented (`TypeError` on failure).
4. `main.py` iterates `ALL_COMMANDS` and calls `cmd.register(subparsers)`.
5. `BaseCommand.register()` creates an argparse subparser (using `name()` and
   `help()`), calls `add_arguments()` to wire up flags, and sets
   `parser.set_defaults(func=self.execute)`.
6. After parsing, `main.py` dispatches via `args.func(args)`, which calls the
   matched command's `execute()`.

### How to add a new subcommand

**Step 1.** Create `lmcache/cli/commands/describe.py`:

```python
from lmcache.cli.commands.base import BaseCommand

class DescribeCommand(BaseCommand):
    def name(self) -> str:
        return "describe"

    def help(self) -> str:
        return "Describe a running KV cache server."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--url", required=True)

    def execute(self, args) -> None:
        ...  # implementation
```

**Step 2.** Register it in `lmcache/cli/commands/__init__.py`:

```python
from lmcache.cli.commands.describe import DescribeCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    DescribeCommand(),   # <-- add here
]
```

That's it — `lmcache describe --url http://localhost:8000` is now available.

### File layout

```
lmcache/cli/
├── __init__.py          # empty
├── main.py              # main() entry point
├── metrics/             # Metrics system (Section 2)
│   ├── __init__.py      # re-exports
│   ├── metrics.py       # Metrics collector
│   ├── section.py       # Section data class
│   ├── handler.py       # StreamHandler, FileHandler
│   └── formatter.py     # TerminalFormatter, JsonFormatter
├── commands/
│   ├── __init__.py      # ALL_COMMANDS registry
│   ├── base.py          # BaseCommand ABC
│   └── mock.py          # lmcache mock  (example command)
└── corpora/             # built-in prompt corpora (future)
```

### Entry point (pyproject.toml)

```toml
[project.scripts]
lmcache = "lmcache.cli.main:main"
```

---

## 2. Hierarchical Metrics System

### Goal

A lightweight, dependency-free metrics system that:

1. Accepts metrics organized into **sections** (categories).
2. Uses a **handler + formatter** architecture (like Python `logging`) to
   separate *where* metrics are written from *how* they are rendered.
3. Supports stdout, file, and future destinations (e.g. Kafka) without
   requiring command authors to manage handlers themselves.

### Architecture

The metrics system has three layers:

- **`Metrics`** — the collector. Holds sections and entries. Calls `emit()`
  to trigger all registered handlers.
- **`MetricsHandler`** — the destination (where to write). Each handler holds
  a formatter. Built-in: `StreamHandler` (writes to a stream like stdout),
  `FileHandler` (writes to a file path).
- **`MetricsFormatter`** — the rendering (how to format). Built-in:
  `TerminalFormatter` (ASCII table), `JsonFormatter` (JSON string).

```
Metrics ──emit()──▶ Handler (destination) ──▶ Formatter (rendering)
                     StreamHandler(stdout)      TerminalFormatter
                     FileHandler("out.json")    JsonFormatter
```

### File layout

```
lmcache/cli/metrics/
├── __init__.py       # re-exports
├── metrics.py        # Metrics collector
├── section.py        # Section data class
├── handler.py        # MetricsHandler, StreamHandler, FileHandler
└── formatter.py      # MetricsFormatter, TerminalFormatter, JsonFormatter
```

### API

Each metric has a **machine key** (used in JSON output) and a **human-readable
label** (used in terminal output). Sections work the same way.

```python
from lmcache.cli.metrics import Metrics, StreamHandler, TerminalFormatter

metrics = Metrics(title="Bench KV Cache Result (30s)")

# Title can be changed after construction
metrics.title("Bench KV Cache Result (60s)")

# Create named sections (machine key + display label)
metrics.add_section("ops", "Operations (ops/s)")
metrics.add_section("hit_rate", "Hit Rate")
metrics.add_section("correctness", "Correctness")

# Add metrics to sections via dict-like access
metrics["ops"].add("store", "Store", 41.3)
metrics["ops"].add("retrieve", "Retrieve", 127.3)
metrics["hit_rate"].add("l1", "L1", "92.3%")
metrics["correctness"].add("checksums", "Checksums", "5060/5060 OK")

# Trigger all handlers
metrics.emit()
```

**Command authors don't register handlers manually.** `BaseCommand.create_metrics()`
sets up default handlers automatically:

```python
# Inside a command's execute() method:
metrics = self.create_metrics("Bench Result", args, width=48)
# ^ automatically adds:
#   - StreamHandler → stdout (formatter chosen by --format, default: terminal)
#   - FileHandler   → if --output is set (same format as --format)
```

### Handlers and Formatters

**Handlers** (destination):

| Handler | Default Formatter | Description |
|---|---|---|
| `StreamHandler(formatter, stream)` | `TerminalFormatter` | Writes to a text stream (default: stdout) |
| `FileHandler(path, formatter)` | `JsonFormatter` | Writes to a file |

**Formatters** (rendering):

| Formatter | Description |
|---|---|
| `TerminalFormatter(width)` | ASCII table with `=`/`-` dividers |
| `JsonFormatter(indent)` | JSON string |

Custom handlers and formatters can be added by subclassing `MetricsHandler`
and `MetricsFormatter`.

### Terminal output format

```
========= Bench KV Cache Result (30s) =========
--------------Operations (ops/s)----------------
Store:                                   41.3
Retrieve:                                127.3
-----------------Hit Rate-----------------------
L1:                                      92.3%
--------------Correctness-----------------------
Checksums:                               5060/5060 OK
================================================
```

Design choices:
- **Fixed total width** of 48 characters (configurable via `width` param on
  `TerminalFormatter`).
- Title row is centered within `=` borders.
- Section headers are centered within `-` borders.
- Key-value lines are left-aligned label, right-aligned value.
- Values are formatted automatically: floats get 2 decimal places, strings are
  printed as-is, `None` is printed as `N/A`.
- Output goes directly to stdout (conventional CLI behavior, not via `logging`).

### JSON output format

JSON uses machine keys, not display labels:

```json
{
  "title": "Bench KV Cache Result (30s)",
  "metrics": {
    "ops": {
      "store": 41.3,
      "retrieve": 127.3
    },
    "hit_rate": {
      "l1": "92.3%"
    },
    "correctness": {
      "checksums": "5060/5060 OK"
    }
  }
}
```

### Flat metrics (no section)

For top-level metrics that don't belong to a section, use `metrics.add()`
directly:

```python
metrics = self.create_metrics("Ping KV Cache", args)
metrics.add("status", "Status", "OK")
metrics.add("rtt_ms", "Round trip time (ms)", 0.42)
metrics.emit()
```

Produces:

```
======= Ping KV Cache =======
Status:                  OK
Round trip time (ms):    0.42
==============================
```

These go into a default unnamed section — no header line is rendered, and in
JSON the entries appear at the top level of `"metrics"`.

### Implementation notes

- `Metrics` holds an ordered list of `Section` objects. Each `Section` stores
  a machine key, a display label, and a list of `(key, label, value)` entries.
- `metrics["name"]` returns the `Section` with that machine key. `KeyError`
  if `add_section()` was not called first.
- `metrics.add(key, label, value)` appends to a default unnamed section
  (created implicitly on first use).
- `emit()` iterates all registered handlers and calls `handler.emit()`.
- `to_dict()` returns `{"title": ..., "metrics": ...}` for programmatic access.
- No external dependencies beyond the Python standard library.

---

## 3. `lmcache mock` — Example Command

A mock command that demonstrates the full framework: argument parsing, metrics
logging, and both terminal and JSON output. It doesn't connect to any server.

```bash
$ lmcache mock --name test-run --num-items 5

============= Mock Result ==============
----------- Input Parameters -----------
Name:                           test-run
Num items:                             5
------------- Mock Metrics -------------
Items processed:                      42
Total time (ms):                   12.34
Throughput (items/s):            3403.73
-------------- Validation --------------
Status:                               OK
========================================

# With --output, both stdout and file are produced (two handlers)
$ lmcache mock --name test-run --num-items 5 --output result.json
(same terminal output)
# result.json → {"title": "Mock Result", "metrics": {"input": {"name": "test-run", ...}, ...}}
```

This command lives in `lmcache/cli/commands/mock.py` and serves as a reference
implementation for future commands. Note how it uses `self.create_metrics()`
instead of manually registering handlers.

---

## 4. Shared CLI Conventions

### `--format` flag

Controls the stdout rendering format. Default: `terminal` (ASCII table). Available:
`terminal`, `json`. Added automatically by `BaseCommand.register()`.

```bash
lmcache bench ... --format json       # JSON on stdout (for scripts)
lmcache bench ... --format terminal   # ASCII table (default)
```

### `--output` flag

Saves metrics to a file. The file format follows ``--format`` (default:
``terminal``). Also added automatically by `BaseCommand.register()`.
Can be combined with `--format`:

```bash
lmcache bench ... --output result.txt                  # terminal format to both stdout and file
lmcache bench ... --format json --output result.json   # JSON to both stdout and file
```

### `--url` flag

The `--url` flag points to the **LMCache HTTP server** (e.g.
`http://localhost:8000`).  Each subcommand configures its own `--url`
flag as needed.

### Error handling

Commands print errors to stderr and return exit code 1. The dispatcher catches
exceptions from `args.func(args)` and prints a clean error message.
