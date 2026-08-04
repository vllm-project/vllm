.. SPDX-License-Identifier: Apache-2.0

Adding a New Device Backend
===========================

This guide explains how to add a **new accelerator** to LMCache in
**Multiprocess (MP) mode**.

**For basic users:** read :ref:`Part 1 <part-1-basic>` only — adding a
``DeviceSpec`` is all you need to get your device working with the
built-in Python fallback ops.

**For advanced users:** continue to :ref:`Part 2 <part-2-performance>`
for native ops and advanced transfer modes.

.. _part-1-basic:

Part 1 — Basic Function Enabling
--------------------------------

For the majority of devices, a single ``DeviceSpec`` class is
sufficient.  LMCache ships with a complete Python fallback ops
(``lmcache/python_ops_fallback.py``) that works on any device
supporting standard PyTorch tensor operations — **no custom kernels
required**.

Prerequisites
~~~~~~~~~~~~~

Your PyTorch backend should support:

**Device Discovery & Status:**

- ``torch.<device>.is_available()`` → ``bool``
- ``torch.<device>.device_count()`` → ``int``

**Device Context & Synchronization:**

- ``torch.<device>.set_device(device)`` → ``None``
- ``torch.<device>.current_device()`` → ``int``
- ``torch.<device>.synchronize()`` → ``None``

**Data Movement:**

- ``tensor.to(device)`` / ``tensor.cpu()`` (host↔device transfers)

Add a ``FooDeviceSpec``
~~~~~~~~~~~~~~~~~~~~~~~

Create ``lmcache/v1/platform/foo/__init__.py``:

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """foo-specific platform primitives."""

    from lmcache.v1.platform.base.device_spec import DeviceSpec


    class FooDeviceSpec(DeviceSpec):
        """foo device specification for LMCache registry discovery."""

        @property
        def device_type(self) -> str:
            return "foo"

        @property
        def torch_module_name(self) -> str:
            return "foo"

        @property
        def ops_module(self) -> str | None:
            return None

        def is_available(self) -> bool:
            """Check backend availability without importing lmcache.__init__."""
            try:
                import torch

                return hasattr(torch, "foo") and torch.foo.is_available()
            except Exception:
                return False

Key properties:

.. list-table::
   :header-rows: 1

   * - Property
     - Required
     - Purpose
   * - ``device_type``
     - yes
     - Device type string (e.g. ``"cuda"``, ``"musa"``)
   * - ``torch_module_name``
     - yes
     - Attribute on the ``torch`` package (e.g. ``"cuda"`` →
       ``torch.cuda``)
   * - ``ops_module``
     - no
     - Ops module path; leave the base-class default (``None``) for
       pure fallback, or override in :ref:`Part 2 <part-2-performance>`
   * - ``is_available()``
     - yes
     - Returns ``True`` when the device is usable

.. note::

   The ``hasattr(torch, "foo")`` guard shown above is only needed for
   out-of-tree PyTorch extensions (e.g. ``torch.musa``, ``torch.xpu``
   when installed as a plug-in).  For accelerators shipped inside
   PyTorch itself (like ``torch.cuda``) a plain
   ``torch.foo.is_available()`` is enough.

That's it.  Defining this class is enough for auto-discovery — no
global list or manual registration call is required.  All ops
automatically route through ``lmcache.python_ops_fallback``, which is
not performant but is functionally applicable to any device that
supports the standard PyTorch tensor surface.  Later, each device can
provide its own ops (Python, C, CUDA, Rust, or any language exposing
Python bindings) to override the fallback function-by-function.

Verification
~~~~~~~~~~~~

Start LMCache server:

.. code-block:: bash

    lmcache server --l1-size-gb 10 --eviction-policy LRU --port 5555

Run vLLM with MP connector.  If multiple accelerators are visible on
the host, set ``DEVICE_TYPE`` to force LMCache to pick the new backend
instead of auto-detecting:

.. code-block:: bash

    export DEVICE_TYPE=foo            # optional; only when auto-detection picks the wrong device

    vllm serve <your-model> \
        --kv-transfer-config '{
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": "5555",
                "lmcache.mp.mp_transfer_mode": "engine_driven"
            }
        }' \
        --no-enable-prefix-caching \
        --port 8000

.. note::

   The default transfer mode is ``auto`` (CUDA → LMCache-driven, other
   devices → engine-driven).  The example above explicitly sets
   ``engine_driven`` so that a new non-CUDA device works without
   additional capability checks.  For the ``lmcache_driven`` mode
   (IPC zero-copy), see :ref:`Advanced transfer mode <part-2-performance>`.

Check the LMCache logs — with ``ops_module = None`` you should see::

    torch_dev=..., torch_device_type=foo
    Custom ops not supported for device: foo, using fallback ops.

This confirms your ``DeviceSpec`` was discovered and the Python
fallback is active.  (Once you set ``ops_module`` in
:ref:`Part 2 <part-2-performance>`, the second line becomes
``Using backend: foo_device.ops`` instead.)

Debugging checklist:

- [ ] ``torch.foo.is_available()`` returns ``True``.
- [ ] Set ``DEVICE_TYPE=foo`` to force selection if not picked up
  automatically.
- [ ] Log shows either ``Custom ops not supported for device: foo,
  using fallback ops.`` (pure fallback) or
  ``Using backend: foo_device.ops`` (custom ops loaded).
- [ ] Engine-driven transfer works end-to-end (check the LMCache logs
  to confirm whether the SHM or Pickle sub-path is chosen — both
  should succeed).
- [ ] Store/retrieve correctness is verified.
- [ ] TP>1 / multi-worker behavior is verified.

.. _part-2-performance:

Part 2 — Performance Optimization
---------------------------------

Once basic functionality is verified, add device-specific
optimizations.

Device-specific ops
~~~~~~~~~~~~~~~~~~~

The generic ops interface is defined in
``lmcache/python_ops_fallback.py``.  Each vendor may replace any
subset of these functions with a device-specific implementation; the
choice of language (C, CUDA, SYCL, Python, Rust, …) and packaging is
entirely up to the vendor.  LMCache only cares that the resulting
symbols are importable from Python.

**How it works.** At import time, ``get_backend(device_type)`` imports
the module named by your ``DeviceSpec.ops_module`` and merges its
symbols over ``python_ops_fallback``:

.. code-block:: text

    callers  →  lmcache.c_ops  →  foo_device.ops          (functions you defined)
                               →  lmcache.python_ops_fallback  (everything else)

Integration contract
^^^^^^^^^^^^^^^^^^^^

Regardless of how you build your ops module, the following contract
must hold:

- **Same symbol names.** Every function you override must be exposed
  under the exact name used in ``python_ops_fallback`` (e.g.
  ``multi_layer_block_kv_transfer``).
- **Same call signature.** Positional/keyword arguments, argument
  order and semantics must match the fallback; callers invoke the
  merged module without knowing which backend answered.
- **Importable Python module.** ``ops_module`` is a fully-qualified
  Python module path (``importlib.import_module`` must succeed).  How
  the module gets there — a pure-Python file, a pybind11 extension,
  a ctypes wrapper, a Rust ``PyO3`` module, etc. — is your choice.
- **Partial override is allowed.** You do not have to reimplement
  every function.  Anything you leave out keeps using the Python
  fallback, so incremental optimization is supported.
- **Point ``DeviceSpec.ops_module`` at the module** once it is
  reachable on ``sys.path``:

  .. code-block:: python

      class FooDeviceSpec(DeviceSpec):
          @property
          def ops_module(self) -> str | None:
              return "foo_device.ops"   # any importable module path

Implementation notes
^^^^^^^^^^^^^^^^^^^^

- ``multi_layer_block_kv_transfer`` and ``lmcache_memcpy_async`` are
  the hot entry points for both engine-driven and LMCache-driven
  transfer; other functions in ``python_ops_fallback`` can be
  overridden as needed.
- If you fall back to the generic path from inside a device-specific
  wrapper (e.g. when inputs are unsupported), call the corresponding
  ``lmcache.python_ops_fallback`` function directly to preserve
  semantics.
- Keep your ops module free of side effects at import time — it is
  imported eagerly during ``get_backend()``.

For concrete reference implementations, see
``lmcache/v1/platform/cuda/`` and ``lmcache/v1/platform/musa/``.
Both are examples of what a vendor *may* do, not templates every new
backend has to follow.

Advanced transfer mode
~~~~~~~~~~~~~~~~~~~~~~

By default, the transfer mode is **AUTO**: the router dispatches
strictly by ``device_type`` — ``device_type == "cuda"`` goes to
``LMCacheDrivenTransferContext`` (IPC zero-copy), everything else to
``EngineDrivenTransferContext``.  A non-CUDA device that supports IPC
handle transfer can still opt into LMCache-driven explicitly (below).

.. note::

   ROCm is not a separate backend: under PyTorch a ROCm GPU reports
   ``device_type == "cuda"`` and reuses ``CudaIPCWrapper`` (CUDA IPC)
   for the LMCache-driven path, so it works in AUTO mode with no extra
   setup.  There is currently no dedicated ``platform/rocm`` package.

When the caller (or ``LMCACHE_MP_TRANSFER_MODE``) explicitly requests
``lmcache_driven``, ``_build_lmcache_driven_context`` performs two hard
checks — both must succeed, otherwise the factory raises
``ValueError`` (no silent fallback):

1. Your ``DeviceSpec`` subclass must bind a ``DeviceIPCWrapper``
   subclass (exposing a ``wrap`` classmethod) via
   :attr:`~lmcache.v1.platform.base.device_spec.DeviceSpec.ipc_wrapper_cls`.
   :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` reads that
   binding off the registered spec — no separate registry / auto-scan.
2. ``DeviceSpec.is_handle_transfer_available()`` must return ``True``
   (the base-class default; override to ``False`` only if your device
   lacks IPC handle transfer).

Separately, the LMCache-driven server module also requires a
``BaseCacheContext`` subclass under
``lmcache/v1/platform/foo/cache_context.py`` **and** a matching
``DeviceSpec.create_cache_context`` override that lazy-imports and
instantiates it.  The platform-agnostic factory
``lmcache.v1.platform.cache_context.create_cache_context`` dispatches
by ``device_type`` through the ``DeviceSpec`` registry and invokes
that hook; the default ``DeviceSpec.create_cache_context`` raises
``NotImplementedError`` so a missing override surfaces loudly instead
of silently falling back.  The cache context itself manages the KV
cache layout and pointers used for IPC transfer.

Host-side pinning via ``pin_memory_backend`` is *optional* and only
affects staging-buffer performance; it is not required to enable
LMCache-driven mode.

Event IPC capability
^^^^^^^^^^^^^^^^^^^^

The LMCache-driven multiprocess handle path also requires a platform
event-IPC backend.  The capability is declared by
``DeviceSpec.event_ipc_backend`` and is intentionally separate from
``DeviceOps`` and ``DeviceIPCWrapper``:

* The base ``DeviceSpec`` returns ``None``.  A concrete device must
  explicitly opt in.
* CUDA-style event APIs can use
  ``DefaultEventIPCBackend(event_module=..., device_type=...)``.
* Devices with a different event ABI should implement an
  ``EventIPCBackend`` in their own ``lmcache/v1/platform/foo/`` package.
* Concrete device specs should cache the backend because request futures
  may query this property repeatedly.

The backend contract covers event creation, handle export/import, event
recording, stream wait, query, and synchronization.
``check_event_support(device)`` must raise ``RuntimeError`` when those
operations are unavailable for the requested device.  The default backend
checks for a CUDA-style Event type that accepts ``interprocess=True`` and
provides ``from_ipc_handle``.  A custom backend should instead validate the
equivalent prerequisites for its own event ABI.

For a CUDA-style device, bind and cache the default backend as follows:

.. code-block:: python

    from typing import TYPE_CHECKING

    from lmcache.v1.platform.base.device_spec import DeviceSpec

    if TYPE_CHECKING:
        from lmcache.v1.platform.base.event_ipc import EventIPCBackend

    class FooDeviceSpec(DeviceSpec):
        _event_backend_cache: "EventIPCBackend | None" = None

        @property
        def event_ipc_backend(self) -> "EventIPCBackend":
            backend = self._event_backend_cache
            if backend is None:
                import torch

                from lmcache.v1.platform.base.event_ipc import (
                    DefaultEventIPCBackend,
                )

                backend = DefaultEventIPCBackend(
                    event_module=torch.foo,
                    device_type=self.device_type,
                )
                self._event_backend_cache = backend
            return backend

Event IPC operations must preserve producer/consumer stream ordering without
adding a device-wide synchronization.  ``query_event`` must remain
non-blocking so request futures can poll completion safely.

.. note::

   If the device does not support Event IPC, leave the base
   ``event_ipc_backend`` implementation unchanged so it returns ``None``.
   Engine-driven mode does not require this capability.  LMCache-driven
   mode raises an explicit error rather than falling back to CUDA or to the
   accelerator active in the process.

The capability is checked during worker/server registration and before
constructing a device-aware completion future.  This keeps unsupported
platforms from entering an asynchronous transfer path that cannot order
KV-cache memory safely.  The STORE and RETRIEVE message wire format is
unchanged: the existing event handle bytes are still carried in the
request and response payloads.

Override these methods in your ``DeviceSpec``:

.. code-block:: python

    class FooDeviceSpec(DeviceSpec):
        @property
        def ipc_wrapper_cls(self):
            """Bind the DeviceIPCWrapper subclass for this device.

            Lazy import so the accelerator-specific module is only
            pulled in when the LMCache-driven path is actually used.
            """
            from lmcache.v1.platform.foo.ipc_wrapper import FooIPCWrapper

            return FooIPCWrapper

        def is_handle_transfer_available(self) -> bool:
            """Return True if your device supports IPC handle transfer."""
            return True  # base-class default; override to False if unsupported

        @property
        def pin_memory_backend(self):
            """Return a PinMemoryBackend subclass, or None.

            Optional; only affects host staging performance.
            """
            return None  # default

        def create_cache_context(self, *args, **kwargs):
            """Lazy-import and instantiate the BaseCacheContext for this device.

            Required for LMCache-driven mode; the base-class default
            raises ``NotImplementedError``.
            """
            from lmcache.v1.platform.foo.cache_context import FooCacheContext

            return FooCacheContext(*args, **kwargs)

Opt into LMCache-driven mode by setting ``lmcache.mp.mp_transfer_mode``
to ``lmcache_driven`` in the vLLM ``kv_connector_extra_config`` shown
in :ref:`Part 1 <part-1-basic>`, or by exporting
``LMCACHE_MP_TRANSFER_MODE=lmcache_driven``.  If either hard check
fails, the factory raises ``ValueError`` and refuses to construct the
context — switch back to ``engine_driven`` or ``auto``.

References
----------

.. list-table::
   :header-rows: 1

   * - Topic
     - Path
   * - Device spec base
     - ``lmcache/v1/platform/base/device_spec.py``
   * - Event IPC base
     - ``lmcache/v1/platform/base/event_ipc.py``
   * - Backend loading
     - ``lmcache/v1/platform/__init__.py``
   * - Python fallback
     - ``lmcache/python_ops_fallback.py``
   * - Cache context base
     - ``lmcache/v1/platform/base/cache_context.py``
   * - Cache context factory
     - ``lmcache/v1/platform/cache_context.py``
   * - Reference ``DeviceSpec`` (engine-driven baseline)
     - ``lmcache/v1/platform/cuda/__init__.py``
   * - Reference ``DeviceSpec`` (LMCache-driven capable)
     - ``lmcache/v1/platform/musa/__init__.py``
   * - Engine-driven call site
     - ``lmcache/v1/multiprocess/transfer_context/worker_transfer.py``
       (``EngineDrivenTransferContext``, ``create_transfer_context``)
