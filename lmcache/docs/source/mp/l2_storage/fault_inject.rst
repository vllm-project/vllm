Fault Inject
============

A **test/diagnostic-only** adapter that wraps a real inner adapter and
deterministically drops keys on the load read, simulating partial L2 retrieve
failures.  It exercises CacheBlend's **segmented-prefix** recovery path — a
gapped found-set (hit … miss … hit) that a healthy cache never produces —
without needing an actual faulty backend.

.. warning::

   Never enable in production.  It silently makes reads fail, so any KV it
   serves is intentionally incomplete.

The wrapped backend is given as a full adapter spec under ``inner`` (any
registered ``type``), so it keeps its own configuration, eviction, persist, and
serde settings.  Faulting applies to the load read only: a dropped key is
reported *present at lookup* but its *load fails* (the faithful "L2 retrieve
error"); the prefetch controller releases the load-failed locks via the trim
mask.  Everything else (store, lookup, unlock, delete, usage) passes straight
through to the inner adapter.

**Fields:**

- ``inner`` (required): full adapter spec for the wrapped backend, e.g.
  ``{"type": "fs_native", "base_path": "/dev/shm/l2"}``.
- ``gap_tail_ratios``: positions to always drop, given as distance-from-tail
  fractions of the load length in ``[0, 1]`` (``0.0`` = last chunk, ``0.5`` =
  middle, ``1.0`` = first).  Workload-agnostic and self-scaling across context
  lengths.  Default ``[]``.
- ``gap_indices``: exact head-relative task-positions to always drop.
  Default ``[]``.
- ``rate``: per-key drop probability in ``[0, 1]`` via a stable seeded hash of
  the key.  Default ``0.0`` (pass-through).
- ``seed``: seed for the ``rate`` hash (deterministic given the seed).
  Default ``0``.

Defaults are pass-through; set at least one of ``gap_tail_ratios``,
``gap_indices``, or ``rate`` to drop anything.

Drop a stable mid-prefix gap (the CacheBlend segmented-prefix repro), wrapping
an ``fs_native`` backend:

.. code-block:: bash

    --l2-adapter '{"type": "fault_inject", "inner": {"type": "fs_native", "base_path": "/dev/shm/cb_l2"}, "gap_tail_ratios": [0.5]}'

To actually exercise CacheBlend's **segmented-prefix recovery**, pair the gap
with the server flag ``--enable-segmented-prefix`` (CacheBlend only —
``--engine-type blend``). The dropped mid chunk produces a gapped found-set;
with segmented-prefix enabled the prefix leg **retains the post-gap chunks**
(loads prefix + tail) and recomputes **only the dropped gap**, instead of
truncating the prefix at the gap:

.. code-block:: bash

    lmcache server … \
        --l2-adapter '{"type": "fault_inject", "inner": {"type": "fs_native", "base_path": "/dev/shm/cb_l2"}, "gap_tail_ratios": [0.5]}' \
        --enable-segmented-prefix

Randomly drop ~10% of loads (deterministic given the seed):

.. code-block:: bash

    --l2-adapter '{"type": "fault_inject", "inner": {"type": "fs", "base_path": "/data/l2"}, "rate": 0.1, "seed": 7}'

Drop an exact load position (precise single-gap repro):

.. code-block:: bash

    --l2-adapter '{"type": "fault_inject", "inner": {"type": "fs_native", "base_path": "/dev/shm/cb_l2"}, "gap_indices": [12]}'

At startup the adapter logs ``FaultInjectL2Adapter ACTIVE (rate=… seed=…
gap_indices=… gap_tail_ratios=[…]) wrapping <inner> -- test/diagnostic use
only.``, and ``report_status()`` reports the active fault config under a
``fault_inject`` sub-dict.
