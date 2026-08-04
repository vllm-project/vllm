# MP Observability — Agent Instructions

When working in `lmcache/v1/mp_observability/`:

Design docs live in `docs/design/v1/mp_observability/` — update them whenever
the contracts below change.

1. **New `EventType`** — after adding an entry to `event.py`, update the
   metadata contract table in `docs/design/v1/mp_observability/EVENTS.md`
   with the new type's metadata keys and types.

2. **New metrics subscriber** — after adding counters/histograms, update the
   metrics table in `docs/design/v1/mp_observability/METRICS.md` with the
   metric name, type, and description.

3. **New subscriber class** — follow the step-by-step guide in `README.md`
   (co-located with this file; "How to Add a New Event and Subscriber") and
   the design rules in `docs/design/v1/mp_observability/event-bus.md`.

4. **CLI args** — if you add or change observability CLI flags in `config.py`,
   update `docs/source/mp/observability.rst` to match.
