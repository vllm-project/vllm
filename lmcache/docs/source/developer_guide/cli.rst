Extending the CLI
=================

LMCache's CLI is built on a plugin-style architecture that supports
**N-level nested subcommands** with zero-registration auto-discovery.
This guide explains how to add commands at each level.

Architecture Overview
---------------------

The CLI framework is composed of two core classes in
``lmcache/cli/commands/base.py``:

- **BaseCommand** — Abstract base class for leaf commands (commands that
  perform actual work).
- **CompositeCommand** — A ``BaseCommand`` subclass for commands that
  only group sub-subcommands. It auto-discovers child commands by
  scanning the package where it is defined.

The discovery mechanism uses ``pkgutil.iter_modules`` to scan direct
submodules of a package, then collects all concrete ``BaseCommand``
subclasses found in those modules. This means:

1. Each command is a separate ``.py`` file (or a sub-package with an
   ``__init__.py``).
2. No manual registration is needed — just create the file and it is
   picked up automatically.
3. Utility/helper modules should be prefixed with ``_`` (e.g.
   ``_helpers.py``) so they are excluded from the scan.

Auto-Discovery Mechanism
------------------------

The CLI uses a unified ``discover_subclasses()`` utility (defined in
``lmcache/v1/utils/subclass_discovery.py``) to locate command classes at
every level.  Understanding the two discovery entry points is key to
extending the CLI.

Top-Level Command Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the CLI starts, ``lmcache/cli/commands/__init__.py`` calls
``discover_subclasses()`` on its own package (``lmcache.cli.commands``).
This scans all **direct submodules** — both ``.py`` files and
sub-packages (via their ``__init__.py``) — and collects every concrete
``BaseCommand`` subclass found.

.. code-block:: python

   # Simplified from lmcache/cli/commands/__init__.py
   from lmcache.v1.utils.subclass_discovery import discover_subclasses

   ALL_COMMANDS = [
       cls()
       for cls in discover_subclasses(
           __name__,                          # "lmcache.cli.commands"
           BaseCommand,
           module_filter=lambda name: name != "base",  # skip base.py itself
       )
   ]

The resulting ``ALL_COMMANDS`` list is then registered with the root
``argparse`` parser in ``main.py``.  This means any new ``.py`` file (or
sub-package) placed under ``lmcache/cli/commands/`` is automatically
available as a top-level command — no manual imports or registration
needed.

**Filtering rules for top-level discovery:**

- The module named ``base`` is explicitly excluded (it defines the
  abstract base classes, not a command).
- Modules whose name starts with ``_`` are excluded by the underlying
  ``pkgutil.iter_modules`` convention (they are treated as private).
- Only **concrete** subclasses are collected — abstract classes are
  skipped.
- Only classes **defined in** the scanned module are collected (re-exports
  are ignored via ``require_defined_in_module=True``).

Sub-Command Discovery (CompositeCommand)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a ``CompositeCommand`` registers itself with the parser, its
``register()`` method calls ``discover_subclasses()`` on the package
where the ``CompositeCommand`` subclass is defined (i.e. its own
``__init__.py``'s package).

.. code-block:: python

   # Simplified from CompositeCommand.register() in base.py
   package = self.__class__.__module__   # e.g. "lmcache.cli.commands.bench"

   for cls in discover_subclasses(
       package,
       BaseCommand,
       module_filter=lambda name: not name.startswith("_"),
       require_defined_in_module=True,
   ):
       if cls is self.__class__:
           continue          # skip the CompositeCommand itself
       inst = cls()
       inst.register(inner)  # register as a nested subcommand

**Filtering rules for sub-command discovery:**

- Modules starting with ``_`` are excluded (use this prefix for helper
  files like ``_utils.py``).
- The ``CompositeCommand`` class itself is skipped to avoid infinite
  recursion.
- Only concrete ``BaseCommand`` subclasses **defined in** the scanned
  module are collected.
- The scan is **non-recursive** (depth 1 only) — each
  ``CompositeCommand`` only sees its own package's direct children.

**How nesting works:** If a child is itself a ``CompositeCommand``
(defined in a sub-package's ``__init__.py``), its own ``register()``
method will in turn scan *its* package for further children.  This
creates the recursive nesting without any special configuration.

.. code-block:: text

   CLI startup
   └── __init__.py discovers ALL top-level commands
       ├── ping.py          → PingCommand (leaf)
       ├── bench/__init__.py → BenchCommand (composite)
       │   └── BenchCommand.register() discovers:
       │       ├── server_bench/__init__.py → ServerBenchCommand (leaf)
       │       ├── l2_adapter_bench/__init__.py → L2AdapterBenchCommand (leaf)
       │       └── engine_bench/__init__.py → EngineBenchCommand (leaf)
       └── tool/__init__.py → ToolCommand (composite)
           └── ToolCommand.register() discovers:
               └── cache_simulator/__init__.py → CacheSimulatorCommand (composite)
                   └── CacheSimulatorCommand.register() discovers:
                       ├── simulate_command.py     → SimulateCommand (leaf)
                       ├── sweep_command.py        → SweepCommand (leaf)
                       └── gen_dataset_command.py  → GenDatasetCommand (leaf)

Directory Layout
----------------

.. code-block:: text

   lmcache/cli/commands/
   ├── __init__.py            # Top-level discovery (scans this package)
   ├── base.py                # BaseCommand & CompositeCommand
   ├── ping.py                # Level-1 leaf command
   ├── server.py              # Level-1 leaf command
   ├── quota/                 # Level-2 composite command
   │   ├── __init__.py        # QuotaCommand(CompositeCommand)
   │   ├── _helpers.py        # Utility (excluded from scan by _ prefix)
   │   ├── get_command.py     # Level-2 leaf: ``lmcache quota get``
   │   ├── set_command.py     # Level-2 leaf: ``lmcache quota set``
   │   └── ...
   └── tool/                  # Level-2 composite command
       ├── __init__.py         # ToolCommand(CompositeCommand)
       └── cache_simulator/    # Level-3 composite command
           ├── __init__.py     # CacheSimulatorCommand(CompositeCommand)
           ├── simulate_command.py   # Level-3 leaf: ``lmcache tool cache-simulator simulate``
           └── sweep_command.py      # Level-3 leaf: ``lmcache tool cache-simulator sweep``

Key Rules
^^^^^^^^^

- A ``CompositeCommand`` subclass **must** be defined in the
  ``__init__.py`` of its package.
- Modules starting with ``_`` are **excluded** from auto-discovery
  (use this for helpers, utilities, internal logic).
- Each leaf command file should contain exactly one concrete
  ``BaseCommand`` subclass.
- The scan is **non-recursive** — each ``CompositeCommand`` only scans
  its own package's direct submodules.

Level 1: Adding a Top-Level Command
------------------------------------

A top-level command appears directly under ``lmcache <command>``.

**Step 1**: Create a new file under ``lmcache/cli/commands/``:

.. code-block:: python

   # lmcache/cli/commands/hello.py
   """``lmcache hello`` — a simple greeting command."""

   import argparse

   from lmcache.cli.commands.base import BaseCommand


   class HelloCommand(BaseCommand):
       """Print a greeting message."""

       def name(self) -> str:
           return "hello"

       def help(self) -> str:
           return "Print a greeting message."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--name", default="World", help="Who to greet.")

       def execute(self, args: argparse.Namespace) -> None:
           metrics = self.create_metrics("Hello", args)
           metrics.add("greeting", "Greeting", f"Hello, {args.name}!")
           metrics.emit()

**Step 2**: Done! The command is automatically discovered. Test it:

.. code-block:: bash

   lmcache hello --name LMCache

.. note::

   This works because the top-level ``lmcache/cli/commands/__init__.py``
   calls ``discover_subclasses`` on its own package at import time. It
   uses ``pkgutil.iter_modules`` to find all direct submodules (files and
   sub-packages), imports each one, and collects every concrete
   ``BaseCommand`` subclass. The resulting list is stored in
   ``ALL_COMMANDS`` and registered with the argument parser in
   ``main.py``. So adding a new ``.py`` file with a ``BaseCommand``
   subclass is all that is needed — no edits to any other file.

Level 2: Adding a Subcommand Group
-----------------------------------

A subcommand group appears as ``lmcache <group> <subcommand>``.

**Step 1**: Create a package directory:

.. code-block:: bash

   mkdir lmcache/cli/commands/mygroup/

**Step 2**: Define the ``CompositeCommand`` in ``__init__.py``:

.. code-block:: python

   # lmcache/cli/commands/mygroup/__init__.py
   """``lmcache mygroup`` command group.

   Sub-subcommands are auto-discovered from modules in this package.
   """

   from lmcache.cli.commands.base import CompositeCommand


   class MyGroupCommand(CompositeCommand):
       """Command group for my custom operations."""

       def name(self) -> str:
           return "mygroup"

       def help(self) -> str:
           return "My custom command group."

**Step 3**: Add leaf subcommands as separate files:

.. code-block:: python

   # lmcache/cli/commands/mygroup/foo_command.py
   """``lmcache mygroup foo`` — do something."""

   import argparse

   from lmcache.cli.commands.base import BaseCommand


   class FooCommand(BaseCommand):
       """Execute the foo action."""

       def name(self) -> str:
           return "foo"

       def help(self) -> str:
           return "Execute the foo action."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--value", type=int, required=True)

       def execute(self, args: argparse.Namespace) -> None:
           metrics = self.create_metrics("Foo Result", args)
           metrics.add("result", "Result", args.value * 2)
           metrics.emit()

**Step 4**: (Optional) Add helper modules with ``_`` prefix:

.. code-block:: python

   # lmcache/cli/commands/mygroup/_utils.py
   """Internal utilities for mygroup commands (not auto-discovered)."""

   def compute_something(x: int) -> int:
       return x * 42

**Result**:

.. code-block:: bash

   lmcache mygroup foo --value 5

Level 2: Adding a Subcommand to an Existing Group
---------------------------------------------------

If a ``CompositeCommand`` group already exists (e.g. ``bench``, ``quota``,
``trace``), you can extend it by simply adding **one new file** — no other
changes are required.

For example, to add a new ``lmcache bench l2`` subcommand under the
existing ``bench`` group:

**Step 1**: Create a single file (or sub-package) in the existing group's
package directory:

.. code-block:: python

   # lmcache/cli/commands/bench/l2_adapter_bench/__init__.py
   """``lmcache bench l2`` subpackage."""

   import argparse

   from lmcache.cli.commands.base import BaseCommand


   class L2AdapterBenchCommand(BaseCommand):
       """Benchmark an L2 adapter (store / lookup / load)."""

       def name(self) -> str:
           return "l2"

       def help(self) -> str:
           return "Benchmark an L2 adapter (store / lookup / load)."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           from lmcache.cli.commands.bench.l2_adapter_bench.command import (
               add_l2_arguments,
           )
           add_l2_arguments(parser)

       def execute(self, args: argparse.Namespace) -> None:
           from lmcache.cli.commands.bench.l2_adapter_bench.command import (
               run_l2_adapter_bench,
           )
           run_l2_adapter_bench(self, args)

**Step 2**: Done! The parent ``CompositeCommand`` (``BenchCommand``)
auto-discovers the new subcommand at startup. No registration code, no
imports to add, no ``__init__.py`` edits in the parent.

.. note::

   This works because ``CompositeCommand.register()`` scans all direct
   submodules of its package each time the CLI starts. A new file (or
   sub-package) is automatically picked up as long as:

   - It does **not** start with ``_``.
   - It contains a concrete ``BaseCommand`` subclass.

Level N: Arbitrary Nesting
--------------------------

The framework supports **unlimited nesting depth**. Each level follows
the same pattern: a ``CompositeCommand`` in a package's ``__init__.py``
auto-discovers its children.

**Example**: Adding a 3rd level under ``lmcache mygroup``:

.. code-block:: bash

   mkdir lmcache/cli/commands/mygroup/nested/

.. code-block:: python

   # lmcache/cli/commands/mygroup/nested/__init__.py
   """``lmcache mygroup nested`` — a nested command group."""

   from lmcache.cli.commands.base import CompositeCommand


   class NestedCommand(CompositeCommand):
       """Nested subcommand group."""

       def name(self) -> str:
           return "nested"

       def help(self) -> str:
           return "A nested command group under mygroup."

.. code-block:: python

   # lmcache/cli/commands/mygroup/nested/bar_command.py
   """``lmcache mygroup nested bar`` — a deeply nested command."""

   import argparse

   from lmcache.cli.commands.base import BaseCommand


   class BarCommand(BaseCommand):
       """Execute the bar action at level 3."""

       def name(self) -> str:
           return "bar"

       def help(self) -> str:
           return "Execute the bar action."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--msg", default="deep")

       def execute(self, args: argparse.Namespace) -> None:
           metrics = self.create_metrics("Bar Result", args)
           metrics.add("message", "Message", args.msg)
           metrics.emit()

**Result**:

.. code-block:: bash

   lmcache mygroup nested bar --msg "hello from level 3"

You can continue nesting indefinitely by repeating this pattern.

Real-World Example
------------------

The existing ``lmcache tool cache-simulator simulate`` command
demonstrates 3-level nesting:

.. code-block:: text

   lmcache tool cache-simulator simulate
   │       │    │               └── Level-3 leaf (SimulateCommand in simulate_command.py)
   │       │    └── Level-2 composite (CacheSimulatorCommand in cache_simulator/__init__.py)
   │       └── Level-1 composite (ToolCommand in tool/__init__.py)
   └── CLI entry point

Using the Metrics System
------------------------

The metrics system uses a **handler + formatter** architecture:

- **Metrics** — the collector. Holds sections and entries.
- **Handler** — the destination (stdout, file, etc.).
- **Formatter** — the rendering (ASCII table, JSON, etc.).

``BaseCommand.create_metrics()`` sets up default handlers automatically, so
command authors just build metrics and call ``emit()``:

.. code-block:: python

   def execute(self, args: argparse.Namespace) -> None:
       # create_metrics() auto-registers:
       #   - StreamHandler → stdout (formatter chosen by --format, default: terminal)
       #   - FileHandler   → if --output is set (same format as --format)
       metrics = self.create_metrics("Bench KV Cache Result", args)

       # Create named sections
       metrics.add_section("ops", "Operations (ops/s)")
       metrics["ops"].add("store", "Store", 41.3)
       metrics["ops"].add("retrieve", "Retrieve", 127.3)

       # Top-level metrics (no section header)
       metrics.add("status", "Status", "OK")

       # Trigger all handlers
       metrics.emit()

The ``--format`` and ``--output`` flags are added automatically by
``BaseCommand.register()`` — subcommands do not need to add them manually.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Level
     - Pattern
     - How to add
   * - 1 (top)
     - Single ``.py`` file
     - Create ``lmcache/cli/commands/<name>.py`` with a ``BaseCommand``
       subclass.
   * - 2+
     - Package directory
     - Create ``lmcache/cli/commands/<group>/__init__.py`` with a
       ``CompositeCommand`` subclass, then add leaf commands as sibling
       ``.py`` files.
   * - N (any)
     - Nested package
     - Same as level 2, but inside an existing composite command's
       package. Each ``CompositeCommand`` scans only its own direct
       submodules.

.. tip::

   - Prefix helper/utility modules with ``_`` to exclude them from
     auto-discovery.
   - Each ``CompositeCommand`` must be defined in its package's
     ``__init__.py``.
   - The ``name()`` method determines the CLI token (e.g. ``"foo"``
     becomes ``lmcache ... foo``).
