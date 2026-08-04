lmcache tool
============

The ``lmcache tool`` command groups offline analysis utilities bundled with
LMCache.

.. code-block:: bash

   lmcache tool <tool-name> <action> [options]

Three tools are available: ``cache-simulator`` replays lookup logs,
``transfer-channel-benchmark`` measures peer-to-peer read throughput, and
``flamegraph`` profiles any running process.

.. note::

   ``cache-simulator`` depends on the optional ``plot`` extras
   (``sortedcontainers`` / ``matplotlib``). If they are not installed, the
   sub-command is silently omitted from the CLI. Install the extras to enable
   it.


cache-simulator
---------------

Replay LMCache lookup-hash JSONL logs through an LRU cache to measure the
KV-cache token hit rate. It has three actions:

.. code-block:: bash

   lmcache tool cache-simulator {simulate,sweep,gen-dataset} [options]

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Action
     - Description
   * - ``simulate``
     - Replay logs at a fixed cache capacity; print a text report and save a
       7-panel statistics PNG.
   * - ``sweep``
     - Sweep across a range of cache capacities and save a hit-rate vs.
       capacity PNG.
   * - ``gen-dataset``
     - Generate a ``vllm bench serve`` custom dataset (JSONL) from
       lookup-hash JSONL logs, preserving prefix-sharing structure.

Each action has its own flags. Run the built-in help for the full list:

.. code-block:: bash

   lmcache tool cache-simulator simulate --help
   lmcache tool cache-simulator sweep --help
   lmcache tool cache-simulator gen-dataset --help


transfer-channel-benchmark
--------------------------

Measure read throughput (GB/s) of the LMCache transfer channel
(``lmcache/v1/distributed/transfer_channel/``) for batched peer-to-peer
reads. It allocates the transferred objects through the same
``L1MemoryManager`` production uses, so it exercises the real memory path
rather than raw tensors.

The benchmark runs as **two processes**, a ``server`` that registers a
source buffer and serves its object catalog, and a ``client`` that reads a
subset of those objects and reports throughput:

.. code-block:: bash

   # Terminal 1: the source
   lmcache tool transfer-channel-benchmark --role server \
       --url 127.0.0.1:7600 --buffer-size 8GB --object-size 10MB

   # Terminal 2: the reader
   lmcache tool transfer-channel-benchmark --role client \
       --url 127.0.0.1:7600 --object-size 10MB --num-objects 100 --iters 5

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Description
   * - ``--role {server,client}``
     - Required. ``server`` registers the source buffer; ``client`` reads
       from it and reports throughput.
   * - ``--transfer-channel-type``
     - Transfer channel implementation to benchmark (default: ``nixl``).
   * - ``--url``
     - Server: bind address. Client: the server's advertise URL to read
       from (default: ``127.0.0.1:7600``).
   * - ``--buffer-size``
     - Server: total registered L1 source buffer, e.g. ``8GB``.
   * - ``--object-size`` / ``--page-size``
     - Size of each transferred object and the alignment; must match on
       both sides.
   * - ``--num-objects`` / ``--iters``
     - Objects read per iteration, and number of measured iterations.
   * - ``--verify``
     - Check the bytes read against the source after transfer.

The ``--url``, ``--object-size``, and ``--page-size`` values must match
between the two processes. Run either side with ``--help`` for the full
list, including the client's own ``--listen-url`` and the catalog
``--control-url``.


flamegraph
----------

Attach a profiler to an already-running process, record for a fixed
duration, and render a flame graph to find its bottlenecks.

It gives one unified interface over widely adopted profilers (``py-spy``,
``perf``, and ``bcc``), each producing a flame graph. A single ``--mode``
spans the whole spectrum, from Python and GIL contention to
CPU time, kernel frames, and blocked time: you pick a mode by what you want
to see rather than which tool to run, and get consistent output, defaults,
and permission checks across all six. It wraps these tools rather than
replacing them (it names what to install, it does not install it), so a user
or developer can find a pain point or a resource-heavy code path without
stitching the tools together by hand.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Flag
     - Default
     - Description
   * - ``--pid``
     - *(required)*
     - Process to profile.
   * - ``--mode``
     - ``gil``
     - One of ``on-cpu``, ``off-cpu``, ``wakeup``, ``offwake``, ``wall``,
       ``gil``, or several comma-separated to record each in turn.
   * - ``--duration``
     - ``30``
     - Seconds to record. ``0`` records until interrupted with Ctrl-C.
   * - ``--output``
     - *(auto)*
     - SVG path. Default ``/tmp/lmcache_bench_flames/pid<PID>.<mode>.svg``.
   * - ``--flamegraph-scripts-dir``
     - *(auto)*
     - FlameGraph scripts directory; unused by ``wall`` / ``gil``.


Usage
~~~~~

.. code-block:: bash

   # Which threads of an MP cache server hold the GIL, sampled for 20s
   lmcache tool flamegraph --pid $(pgrep -f 'lmcache server') --mode gil --duration 20

   # Capture GIL, CPU, and blocked-time views in one sweep (one SVG per mode)
   lmcache tool flamegraph --pid $(pgrep -f 'lmcache server') --mode gil,on-cpu,off-cpu

Reach for this command to profile a **real, unmodified process** (a
production or vLLM-driven server under actual traffic, an idle server, or any
arbitrary PID) without standing up a benchmark harness. The ``bench``
commands (``lmcache bench l2`` / ``lmcache bench server``) instead profile a
process under synthetic load they generate.

Modes
~~~~~

.. _lmcache-flamegraph-modes:

LMCache is mostly Python, so the first question is usually where Python time
goes and how contended the GIL is. **To profile Python execution or GIL
contention**, use ``gil`` / ``wall`` (``py-spy``): one root frame per thread,
attaching to an **unmodified** CPython target, but with no kernel frames.

* **gil**: only threads holding the GIL, so interpreter-lock contention is
  directly visible (the default for a live server).
* **wall**: wall-clock time per thread, blocked threads included.

But LMCache also uses C++, Rust, and CUDA to accelerate hot paths and release
the GIL, and py-spy sees none of that. **To profile CPU/IO time, kernel
frames, context switches, or a non-Python process** (or to get a
cross-language picture of where the whole process spends time and blocks),
use the whole-process modes (``perf`` / bcc): every thread, kernel frames,
scheduler activity, any process, but merged into one chart.

* **on-cpu** (``perf``): where CPU *cycles* go.
* **off-cpu** (``offcputime-bpfcc``): time spent *blocked* (I/O, locks).
* **wakeup** (``wakeuptime-bpfcc``): the stacks that *do* the waking, ending
  others' sleep.
* **offwake** (``offwaketime-bpfcc``): ``off-cpu`` and ``wakeup`` combined,
  each blocked stack joined to the stack of the thread that woke it (that
  waker half drawn inverted on top). Use it when ``off-cpu`` shows a big
  blocked tower and you need the cause.

**How they work.** All run as external processes; what differs is *where the
data is collected*, in the kernel (perf, bcc) or by reading the target's
memory (py-spy):

* **perf** samples the CPU at 99 Hz, recording the running thread's stack, so
  it sees only on-CPU work and walks native stacks via frame pointers.
* **bcc** is *not* sampled: an eBPF program fires on scheduler events, so it
  can measure *blocked* time perf cannot, at a cost that grows with the
  target's context-switch rate. Loading eBPF needs privilege.
* **py-spy** works in user space, reading the target's memory
  (``process_vm_readv``) with ``--nonblocking`` (no pause); it sees only
  Python frames and needs ``ptrace`` permission.

**Equivalent raw commands** for ``--pid P``:

.. code-block:: text

   gil      py-spy record --gil --rate 200 --threads --idle --nonblocking --pid P
   wall     py-spy record       --rate 200 --threads --idle --nonblocking --pid P
   on-cpu   perf record -F 99 -g -p P  ->  perf script | stackcollapse-perf.pl | flamegraph.pl
   off-cpu  sudo offcputime-bpfcc  -df -p P  |  flamegraph.pl --colors io
   wakeup   sudo wakeuptime-bpfcc   -f -p P  |  flamegraph.pl --colors wakeup
   offwake  sudo offwaketime-bpfcc -df -p P  |  flamegraph.pl --colors chain

**Native stacks need frame pointers.** The perf/bcc modes unwind native
stacks through frame pointers (perf via ``-g``, bcc via ``bpf_get_stackid``),
so a target built with ``-fomit-frame-pointer`` shows broken C stacks in
either; rebuild with ``-fno-omit-frame-pointer``. py-spy is unaffected (it
reads the interpreter) but sees only Python.


Installation
~~~~~~~~~~~~

Each mode wraps an external tool that must be installed on the host running
the command (the same host as the target); a missing one fails fast naming
what to install.

* `py-spy <https://github.com/benfred/py-spy>`__ (``wall`` / ``gil``): reads
  the interpreter and renders its own SVG. Install with ``pip install
  py-spy``.
* Linux ``perf`` (``on-cpu``): ``sudo apt install linux-tools-generic``
  (Debian / Ubuntu) or ``sudo dnf install perf`` (Fedora / RHEL); the package
  must match the running kernel. Rendered with the FlameGraph scripts below.
* `bcc <https://github.com/iovisor/bcc>`__ (``off-cpu`` / ``wakeup`` /
  ``offwake``): ``sudo apt install bpfcc-tools`` (Debian / Ubuntu) or
  ``sudo dnf install bcc-tools`` (Fedora / RHEL). Its ``*-bpfcc`` tools also
  need ``sudo`` at run time.
* `FlameGraph <https://github.com/brendangregg/FlameGraph>`__, by Brendan
  Gregg: folds and renders the perf/bcc SVGs. Cloned to ``~/FlameGraph`` on
  first use (needs ``git``); or point ``--flamegraph-scripts-dir`` at an
  existing checkout.

Permissions
~~~~~~~~~~~

**Naming Python frames in the perf/bcc modes** requires the target to have
been *started* with ``PYTHONPERFSUPPORT=1`` (its perf trampoline map); you
cannot enable it on a running process.

.. warning::

   * **If it was not set:** perf/bcc still record, but Python frames collapse
     to ``[unknown]`` (C/native stack only). Use ``gil`` / ``wall``, or
     restart the target with the variable set.
   * **If it is set:** it adds a standing per-call overhead for the process's
     lifetime, so reserve it for a dedicated profiling session, not
     production.

**System settings to adjust (bare-metal / VM).** Each tool needs a kernel
permission; set the one for your mode, or run as root, which satisfies all
three:

* ``wall`` / ``gil`` (py-spy) read the target's memory via ``ptrace``, gated
  by Yama. Ubuntu / Debian default ``ptrace_scope`` to ``1`` (trace only
  descendants), which blocks attaching to a server:

  .. code-block:: bash

     sudo sysctl -w kernel.yama.ptrace_scope=0

  ``CAP_SYS_PTRACE`` (or root) bypasses it; py-spy also prints this hint when
  it is blocked.
* ``on-cpu`` (perf) samples via ``perf_event_open``. Level ``3`` (a Debian
  addition) blocks it outright, and any level above ``2`` makes ``perf
  record`` silently write an empty file, so lower it:

  .. code-block:: bash

     sudo sysctl -w kernel.perf_event_paranoid=2

* ``off-cpu`` / ``wakeup`` / ``offwake`` (bcc) load an eBPF program, so their
  ``*-bpfcc`` tools need ``sudo`` (root).

.. note::

   **In a container, use the py-spy modes.** perf and bcc gate on host-wide
   kernel state a container should not change, so run them on the host (the
   settings above apply there).

   py-spy works inside a container. Run the profiler in the **same container**
   as the target (or share its PID namespace with ``--pid=container:<target>``)
   and start the container with ``--cap-add SYS_PTRACE``, the container
   equivalent of the ``ptrace_scope`` setting above:

   .. code-block:: bash

      docker run --cap-add SYS_PTRACE ... <image>
      pip install py-spy
      lmcache tool flamegraph --pid <target-pid> --mode gil

   If the platform will not grant ``CAP_SYS_PTRACE`` (RunPod, some
   Kubernetes), you can still profile a process you start yourself: tracing
   your own child needs no capability, so launch the target **under** py-spy
   instead of attaching to it:

   .. code-block:: bash

      py-spy record --rate 200 --format flamegraph --threads --idle --gil \
          --subprocesses --duration 30 -o /workspace/flame.svg \
          -- <the target launch command>


Cost
~~~~

The mechanisms give each mode a different cost *shape*:

* **perf**: *bounded*, a fixed 99 Hz sample, so overhead is constant; the
  cost is disk, as ``perf.data`` grows with duration × rate × threads ×
  depth (deleted once rendered).
* **bcc**: *unbounded*, firing on every scheduler event ties cost to the
  target's event rate, so a busy service fires the probe constantly. Output
  is tiny.
* **py-spy**: lightest *on the target*, its sampling runs off the target's
  CPUs, whereas perf and bcc steal the target's cycles.

On a production server, prefer ``wall`` / ``gil`` and keep ``--duration``
short; if you need a whole-process mode, ``on-cpu``'s bounded cost is safer
than the bcc modes on a busy target.
