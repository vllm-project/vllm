# Release Notes

# 9.22.0 (2026-08-03)

This is a minor release introducing two flagship (experimental) features — **client-side caching** and **automatic pipelining** — alongside support for Redis 8.10, new commands, and a large batch of stability and parser-robustness fixes. It consolidates everything shipped in 9.22.0-beta.1, so the notes below cover the full 9.21.0 → 9.22.0 upgrade.

⚠️ Two changes to be aware of when upgrading from 9.21.0:

- **Default configuration values changed** ([#3918](https://github.com/redis/go-redis/pull/3918)): read/write timeouts, retry backoff, cluster state reload interval, and TCP keep-alive defaults are now aligned with the cross-SDK configuration proposal (see the highlight below). Explicitly configured values are unaffected.
- **`WaitAOF` return type corrected** ([#3888](https://github.com/redis/go-redis/pull/3888)): `WaitAOF` now returns `*IntSliceCmd`, matching the two-integer reply of `WAITAOF` (previously `*IntCmd`, which failed to parse the reply at runtime). Code referencing the old return type needs a one-line update.

## 🚀 Highlights

### Client-Side Caching (Experimental)

The standalone `Client` gains server-assisted client-side caching built on RESP3 `CLIENT TRACKING`. Enable it by setting `ClientSideCacheConfig` in `Options` (or supply your own cache via `ClientSideCache` — e.g. to share one cache across clients). Cacheable read results are served from a local in-process cache and invalidated automatically when the server reports a change, cutting round trips for read-heavy workloads.

The invalidation architecture is selected by `ClientSideCacheStrategy`; the default (and currently only) strategy is `CSCStrategySharedTracking`: one shared cache, every pool connection runs plain `CLIENT TRACKING ON`, and a background drainer applies buffered invalidations — portable (no BCAST) and consistent with the other Redis client libraries. Requirements and guardrails: RESP3 (`Protocol: 3`), standalone client, DB 0 only; commands that would change the connection identity (`SELECT`, `AUTH`, ...) are rejected while caching is enabled, and CSC is disabled when a credentials provider is set (fixed `Username`/`Password` work and are namespaced). See the README's [client-side caching section](README.md#client-side-caching) and the runnable [example](example/client-side-caching).

**Experimental:** the API may change in a minor release.

([#3941](https://github.com/redis/go-redis/pull/3941)) by [@ofekshenawa](https://github.com/ofekshenawa)

### Automatic Pipelining (Experimental)

`AutoPipeliner` is a background batcher that coalesces commands from many concurrent goroutines into Redis pipelines, multiplying throughput without any manual pipeline management. It comes in two faces, available on `Client` and `ClusterClient` (and configurable via `Options.AutoPipelineOptions` / `UniversalOptions.AutoPipelineOptions`):

- **`AutoPipeline()`** — the blocking face: a drop-in `Cmdable` where each call blocks until executed, exactly like a plain client, while concurrent callers' commands batch together under the hood (measured locally over loopback: ~1M+ SET/sec vs ~100k unpipelined; indicative, not a guarantee). Per-goroutine command order is preserved.
- **`AsyncAutoPipeline()`** — the deferred face: command calls return immediately and every typed result accessor (`Val`/`Result`/`Err`/...) blocks until the command has executed. Submit a window of commands, then read the results, to keep pipelines deep (~2–3M SET/sec locally; indicative).

`AutoPipelineOptions` controls batching: `MaxBatchSize` (soft target, default 200; the blocking face's preset uses 300), `MaxBatchBytes` (approximate payload cap so huge values flush as several bounded writes), `MaxFlushDelay` with optional `AdaptiveDelay` (delay scales down as the queue fills), and `MaxConcurrentBatches` (default 1 = a single ordered batch stream; raising it requires `Unordered: true`, so ordering is never lost by accident — `Validate()` rejects the combination otherwise). A usage tour and throughput comparison live in [`example/autopipeline`](example/autopipeline).

**Experimental:** the API may change in a future release — pin your go-redis version if you adopt it.

([#3942](https://github.com/redis/go-redis/pull/3942)) by [@ndyakov](https://github.com/ndyakov), with help from [@cxljs](https://github.com/cxljs)

### Redis 8.10 Support

This release adds support for **Redis 8.10**. The README's supported-versions list now includes Redis 8.10, and CI runs the full suite against the `redislabs/client-libs-test:8.10.0` image by default ([#3920](https://github.com/redis/go-redis/pull/3920), [#3940](https://github.com/redis/go-redis/pull/3940)).

Coverage for the new commands and options that ship with Redis 8.10:

- **`HIMPORT`** ([#3919](https://github.com/redis/go-redis/pull/3919)) — bulk hash import via server-side fieldsets, exposed as `HImportPrepare`, `HImportSet`, `HImportDiscard`, and `HImportDiscardAll`. Fieldsets are session state scoped to a single physical connection, which does not mix well with connection pooling — so the client keeps a versioned fieldset registry and lazily replays the `PREPARE` on whichever pooled connection executes a `SET` that needs it, at most once per connection, with no extra round trip (the `PREPARE` is injected into the same write as the `SET`).
- **`LMOVEM` / `BLMOVEM`** ([#3913](https://github.com/redis/go-redis/pull/3913)) — move multiple elements between lists in one call.
- **`SUNIONCARD` / `SDIFFCARD`** ([#3897](https://github.com/redis/go-redis/pull/3897)) — cardinality of set union/difference without materializing the result.
- **`XREAD` / `XREADGROUP` `MAXCOUNT` and `MAXSIZE`** ([#3898](https://github.com/redis/go-redis/pull/3898)) — bound how much data a stream read returns.
- **`TS.READ`** ([#3896](https://github.com/redis/go-redis/pull/3896)), **`TS.QUERYLABELS`** ([#3926](https://github.com/redis/go-redis/pull/3926)), **`TS.NRANGE` / `TS.NREVRANGE`** ([#3870](https://github.com/redis/go-redis/pull/3870)) with multiple aggregators per key ([#3937](https://github.com/redis/go-redis/pull/3937)), and **`EXCLUDEEMPTY`** on `TS.MRANGE` / `TS.MREVRANGE` ([#3912](https://github.com/redis/go-redis/pull/3912)) — new time-series query surface.
- **`FT.ALIASLIST`** ([#3925](https://github.com/redis/go-redis/pull/3925)), **`COLLECT` reducer for `FT.AGGREGATE`** ([#3886](https://github.com/redis/go-redis/pull/3886)), **`RERANK` on HNSW vector fields in `FT.CREATE`** ([#3927](https://github.com/redis/go-redis/pull/3927)), and **`FT.HYBRID` timeout warnings** ([#3911](https://github.com/redis/go-redis/pull/3911)) — search coverage.

### Cross-SDK Aligned Defaults

Default configuration values now follow the cross-SDK configuration proposal shared by all Redis client libraries ([#3918](https://github.com/redis/go-redis/pull/3918)):

| Setting | Old default | New default |
|---|---|---|
| `ReadTimeout` / `WriteTimeout` | 3s | 5s |
| Retry backoff (min/max) | 8ms / 512ms | 10ms / 1s |
| Cluster state reload interval | 10s | 60s |
| TCP keep-alive | 5min period | 30s idle / 5s interval / 3 probes (`net.KeepAliveConfig`) |

Applications that set these values explicitly are unaffected; applications relying on the old defaults inherit the new ones.

### Data-Race and Parser Hardening Sweep

A systematic audit fixed data races across the client — hooks (`AddHook`, [#3868](https://github.com/redis/go-redis/pull/3868)), `Ring.SetAddrs` ([#3862](https://github.com/redis/go-redis/pull/3862)), cluster node slices ([#3861](https://github.com/redis/go-redis/pull/3861)), pub/sub reconnect ([#3906](https://github.com/redis/go-redis/pull/3906)), maintenance notifications ([#3894](https://github.com/redis/go-redis/pull/3894), [#3872](https://github.com/redis/go-redis/pull/3872)), pool handoff ([#3876](https://github.com/redis/go-redis/pull/3876)), and `redisotel` ([#3881](https://github.com/redis/go-redis/pull/3881)) — and hardened the RESP parsers against malformed or unexpected replies: over-reads on nil replies ([#3874](https://github.com/redis/go-redis/pull/3874)), integer overflow when skipping map/attribute bodies ([#3877](https://github.com/redis/go-redis/pull/3877)), unhashable RESP3 map keys ([#3873](https://github.com/redis/go-redis/pull/3873)), odd-length flat replies ([#3900](https://github.com/redis/go-redis/pull/3900)), mismatched declared array lengths ([#3907](https://github.com/redis/go-redis/pull/3907)), unexpected extra reply frames ([#3884](https://github.com/redis/go-redis/pull/3884)), and nil elements in numeric/bool slice replies ([#3922](https://github.com/redis/go-redis/pull/3922)).

### PubSub `Receive` Hang Fix

`PeekPushNotificationName` blocked until 36 bytes were buffered, so a short subscribe confirmation (channel name of six or fewer characters) on an otherwise idle connection hung `PubSub.Receive` forever — a regression introduced in 9.20.1 by [#3842](https://github.com/redis/go-redis/pull/3842). The peek now parses whatever is already buffered and only waits for one more byte when the frame prefix is valid but incomplete. Fixes [#3935](https://github.com/redis/go-redis/issues/3935).

([#3936](https://github.com/redis/go-redis/pull/3936)) by [@ndyakov](https://github.com/ndyakov)

### Correct Cluster Transaction Retries

The cluster transaction pipeline treated a `MULTI`...`EXEC` block as independently retryable commands, which could scatter a transaction across nodes or send malformed transactions on retry. Redirects (`MOVED`/`ASK`/`TRYAGAIN`) and aborts are now handled at the whole-transaction level, matching Redis transaction semantics: the transaction is re-routed and retried as a unit, never partially ([#3909](https://github.com/redis/go-redis/pull/3909)) by [@cxljs](https://github.com/cxljs).

### Credential Redaction in Command Tracing

`rediscmd.AppendCmd` — used by `redisotel` and `rediscensus` to render commands into span attributes — now redacts credential arguments as `<redacted>`: `AUTH`, `HELLO ... AUTH`, `CONFIG SET` of `requirepass` / `masterauth` / TLS key passphrases, `ACL SETUSER` password rules, and `MIGRATE ... AUTH`/`AUTH2`. The client sends `HELLO ... AUTH` on every handshake and `AUTH` on every streaming-credentials rotation through the regular hook chain, so tracing hooks previously captured credentials even when the application never issued an auth command itself ([#3939](https://github.com/redis/go-redis/pull/3939)) by [@saddamr3e](https://github.com/saddamr3e).

## ✨ New Features

- **Client-side caching**: server-assisted caching for the standalone client via `ClientSideCacheConfig` / `ClientSideCache`, with the `CSCStrategySharedTracking` invalidation strategy ([#3941](https://github.com/redis/go-redis/pull/3941)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **Automatic pipelining**: `AutoPipeline()` (blocking) and `AsyncAutoPipeline()` (deferred results) on `Client` and `ClusterClient`, configured via `AutoPipelineOptions` ([#3942](https://github.com/redis/go-redis/pull/3942)) by [@ndyakov](https://github.com/ndyakov), with help from [@cxljs](https://github.com/cxljs)
- **`HIMPORT` command family**: `HImportPrepare` / `HImportSet` / `HImportDiscard` / `HImportDiscardAll` with lazy per-connection fieldset prepare replay ([#3919](https://github.com/redis/go-redis/pull/3919)) by [@ndyakov](https://github.com/ndyakov)
- **`LMOVEM` / `BLMOVEM`**: move multiple list elements in one call, with `COUNT` (up to N) or `EXACTLY` (all-or-nothing) semantics via `LMoveMArgs` ([#3913](https://github.com/redis/go-redis/pull/3913)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`SUnionCard` / `SDiffCard`**: cardinality of set union/difference ([#3897](https://github.com/redis/go-redis/pull/3897)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`XRead` / `XReadGroup` `MAXCOUNT` / `MAXSIZE`**: bound stream read responses by entry count or payload size ([#3898](https://github.com/redis/go-redis/pull/3898)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`TS.READ`**: read samples from a series starting at a given timestamp, with `TSReadEarliest` (`-`), `TSReadLatest` (`+`), and `TSReadNew` (`$`) sentinels ([#3896](https://github.com/redis/go-redis/pull/3896)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`TS.QUERYLABELS`**: query label names/values across time series ([#3926](https://github.com/redis/go-redis/pull/3926)) by [@ndyakov](https://github.com/ndyakov)
- **`TS.NRANGE` / `TS.NREVRANGE`**: range queries across multiple series ([#3870](https://github.com/redis/go-redis/pull/3870)) by [@ofekshenawa](https://github.com/ofekshenawa), with multiple aggregators per key ([#3937](https://github.com/redis/go-redis/pull/3937)) by [@ndyakov](https://github.com/ndyakov)
- **`TS.MRANGE` / `TS.MREVRANGE` `EXCLUDEEMPTY`**: skip series with no samples in the result ([#3912](https://github.com/redis/go-redis/pull/3912)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.ALIASLIST`**: list all index aliases ([#3925](https://github.com/redis/go-redis/pull/3925)) by [@ndyakov](https://github.com/ndyakov)
- **`FT.AGGREGATE` `COLLECT` reducer**: collect grouped values into an array ([#3886](https://github.com/redis/go-redis/pull/3886)) by [@ndyakov](https://github.com/ndyakov)
- **`FT.CREATE` `RERANK`**: `RERANK` parameter on HNSW vector field definitions ([#3927](https://github.com/redis/go-redis/pull/3927)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.HYBRID` timeout warnings**: timeout warnings are now populated in hybrid search results ([#3911](https://github.com/redis/go-redis/pull/3911)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.HYBRID` KNN `SHARD_K_RATIO`** (Redis 8.8+): per-shard K ratio for KNN clauses ([#3841](https://github.com/redis/go-redis/pull/3841)) by [@ndyakov](https://github.com/ndyakov)

## 🐛 Bug Fixes

- **PubSub `Receive` hang**: peek push-notification names without demanding 36 buffered bytes, fixing a hang on short subscribe confirmations (fixes [#3935](https://github.com/redis/go-redis/issues/3935), regression from 9.20.1) ([#3936](https://github.com/redis/go-redis/pull/3936)) by [@ndyakov](https://github.com/ndyakov)
- **Cluster transactions**: re-route the whole tx pipeline on redirect/abort instead of per-command ([#3909](https://github.com/redis/go-redis/pull/3909)) by [@cxljs](https://github.com/cxljs)
- **Credential leak in traces**: `rediscmd.AppendCmd` redacts credential arguments (`AUTH`, `HELLO ... AUTH`, `CONFIG SET` secret params, `ACL SETUSER` password rules, `MIGRATE AUTH`/`AUTH2`), so `redisotel` / `rediscensus` span attributes no longer contain passwords ([#3939](https://github.com/redis/go-redis/pull/3939)) by [@saddamr3e](https://github.com/saddamr3e)
- **`WaitAOF` return type**: returns `*IntSliceCmd` matching the two-integer `WAITAOF` reply ([#3888](https://github.com/redis/go-redis/pull/3888)) by [@CipherN9](https://github.com/CipherN9)
- **`Ring.Publish` routing**: publish to the shard that owns the topic instead of a round-robined one ([#3893](https://github.com/redis/go-redis/pull/3893)) by [@dkindel](https://github.com/dkindel)
- **Pool `OnRemove` hooks**: fire `OnRemove` on `putConn` eviction paths so removal hooks see every evicted connection ([#3932](https://github.com/redis/go-redis/pull/3932)) by [@cxljs](https://github.com/cxljs)
- **`UniversalClient` `InfoMap`**: added `InfoMap` to the `Cmdable` interface ([#3904](https://github.com/redis/go-redis/pull/3904)) by [@nazarli-shabnam](https://github.com/nazarli-shabnam)
- **`SlowLogGet` context**: pass the caller's context instead of a background one ([#3915](https://github.com/redis/go-redis/pull/3915)) by [@sonnemusk](https://github.com/sonnemusk)
- **`ModuleLoadex` nil config**: return an error instead of panicking on nil config ([#3916](https://github.com/redis/go-redis/pull/3916)) by [@sonnemusk](https://github.com/sonnemusk)
- **`ParseURL` IPv6 hosts**: keep single brackets for IPv6 hosts without a port ([#3882](https://github.com/redis/go-redis/pull/3882)) by [@sueun-dev](https://github.com/sueun-dev)
- **`ParseURL` durations**: treat unit durations `<= 0` as disabled ([#3866](https://github.com/redis/go-redis/pull/3866)) by [@sueun-dev](https://github.com/sueun-dev)
- **Nil `*uint8` encoding**: encode nil `*uint8` as `"0"` like other numeric pointers ([#3869](https://github.com/redis/go-redis/pull/3869)) by [@sueun-dev](https://github.com/sueun-dev)
- **`JSONSliceCmd` read errors**: return the read error from `readReply` instead of swallowing it ([#3903](https://github.com/redis/go-redis/pull/3903)) by [@saddamr3e](https://github.com/saddamr3e)
- **RESP parser hardening**: reconcile declared entry-array lengths ([#3907](https://github.com/redis/go-redis/pull/3907)), handle nil elements in int/uint/bool slice parsers ([#3922](https://github.com/redis/go-redis/pull/3922)), drain unexpected reply frames ([#3884](https://github.com/redis/go-redis/pull/3884)), reject odd-length flat replies in Z/KeyValue parsers ([#3900](https://github.com/redis/go-redis/pull/3900)), avoid int overflow when skipping map/attr bodies ([#3877](https://github.com/redis/go-redis/pull/3877)), don't over-read nil replies in `Reader.Discard` ([#3874](https://github.com/redis/go-redis/pull/3874)) by [@saddamr3e](https://github.com/saddamr3e); reject unhashable keys in RESP3 map parsing ([#3873](https://github.com/redis/go-redis/pull/3873)) by [@iabdullah215](https://github.com/iabdullah215)
- **Data races**: hook state during `AddHook` ([#3868](https://github.com/redis/go-redis/pull/3868)), `onNewNode` during `Ring.SetAddrs` ([#3862](https://github.com/redis/go-redis/pull/3862)), shared masters/slaves slices in cluster ([#3861](https://github.com/redis/go-redis/pull/3861)), shared `opt.Addr` during pub/sub reconnect ([#3906](https://github.com/redis/go-redis/pull/3906)), `clusterStateReloadCallback` in maintnotifications ([#3894](https://github.com/redis/go-redis/pull/3894)), conn reader in `isHealthyConn` during handoff ([#3876](https://github.com/redis/go-redis/pull/3876)) by [@saddamr3e](https://github.com/saddamr3e); handoff race window in maintnotifications ([#3872](https://github.com/redis/go-redis/pull/3872)) by [@ndyakov](https://github.com/ndyakov)
- **`redisotel`**: use `ObservableCounter` for cumulative pool stats ([#3914](https://github.com/redis/go-redis/pull/3914)) by [@Solaris-star](https://github.com/Solaris-star); avoid a data race on shared attributes during `MinIdleConns` warmup ([#3881](https://github.com/redis/go-redis/pull/3881)) by [@ndyakov](https://github.com/ndyakov)

## 🧰 Maintenance

- **Cross-SDK default alignment**: new defaults for timeouts, retry backoff, cluster state reload, and TCP keep-alive ([#3918](https://github.com/redis/go-redis/pull/3918)) by [@ndyakov](https://github.com/ndyakov)
- **CI on Redis 8.10**: 8.10 made the default test version ([#3920](https://github.com/redis/go-redis/pull/3920)) with version gating by major.minor ([#3908](https://github.com/redis/go-redis/pull/3908)) by [@ofekshenawa](https://github.com/ofekshenawa); the test stack now runs the GA `redislabs/client-libs-test:8.10.0` image and 8.8 was dropped from the CI matrix ([#3940](https://github.com/redis/go-redis/pull/3940))
- **Type-safe atomics**: use typed `sync/atomic` value types ([#3860](https://github.com/redis/go-redis/pull/3860)) and remove the dead `assertUnstableCommand` RESP3 path ([#3928](https://github.com/redis/go-redis/pull/3928)) by [@cxljs](https://github.com/cxljs)
- **Docs**: clarify that `ExpireTime` / `PExpireTime` return Unix timestamps ([#3917](https://github.com/redis/go-redis/pull/3917)) by [@sonnemusk](https://github.com/sonnemusk); remove a duplicate example step ([#3875](https://github.com/redis/go-redis/pull/3875)) by [@andy-stark-redis](https://github.com/andy-stark-redis)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@andy-stark-redis](https://github.com/andy-stark-redis), [@CipherN9](https://github.com/CipherN9), [@cxljs](https://github.com/cxljs), [@dkindel](https://github.com/dkindel), [@iabdullah215](https://github.com/iabdullah215), [@nazarli-shabnam](https://github.com/nazarli-shabnam), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@saddamr3e](https://github.com/saddamr3e), [@Solaris-star](https://github.com/Solaris-star), [@sonnemusk](https://github.com/sonnemusk), [@sueun-dev](https://github.com/sueun-dev)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.21.0...v9.22.0

# 9.22.0-beta.1 (2026-07-29)

This is a **beta** release adding support for Redis 8.10, new commands, and a large batch of stability and parser-robustness fixes. The 9.22.0 GA release will follow once client-side caching and auto-pipelining are merged.

⚠️ Two changes to be aware of when upgrading from 9.21.0:

- **Default configuration values changed** ([#3918](https://github.com/redis/go-redis/pull/3918)): read/write timeouts, retry backoff, cluster state reload interval, and TCP keep-alive defaults are now aligned with the cross-SDK configuration proposal (see the highlight below). Explicitly configured values are unaffected.
- **`WaitAOF` return type corrected** ([#3888](https://github.com/redis/go-redis/pull/3888)): `WaitAOF` now returns `*IntSliceCmd`, matching the two-integer reply of `WAITAOF` (previously `*IntCmd`, which failed to parse the reply at runtime). Code referencing the old return type needs a one-line update.

## 🚀 Highlights

### Redis 8.10 Support

This release adds support for **Redis 8.10**. The README's supported-versions list now includes Redis 8.10, and CI runs the full suite against the `redislabs/client-libs-test:8.10.0` image by default ([#3920](https://github.com/redis/go-redis/pull/3920), [#3940](https://github.com/redis/go-redis/pull/3940)).

Coverage for the new commands and options that ship with Redis 8.10:

- **`HIMPORT`** ([#3919](https://github.com/redis/go-redis/pull/3919)) — bulk hash import via server-side fieldsets, exposed as `HImportPrepare`, `HImportSet`, `HImportDiscard`, and `HImportDiscardAll`. Fieldsets are session state scoped to a single physical connection, which does not mix well with connection pooling — so the client keeps a versioned fieldset registry and lazily replays the `PREPARE` on whichever pooled connection executes a `SET` that needs it, at most once per connection, with no extra round trip (the `PREPARE` is injected into the same write as the `SET`).
- **`LMOVEM` / `BLMOVEM`** ([#3913](https://github.com/redis/go-redis/pull/3913)) — move multiple elements between lists in one call.
- **`SUNIONCARD` / `SDIFFCARD`** ([#3897](https://github.com/redis/go-redis/pull/3897)) — cardinality of set union/difference without materializing the result.
- **`XREAD` / `XREADGROUP` `MAXCOUNT` and `MAXSIZE`** ([#3898](https://github.com/redis/go-redis/pull/3898)) — bound how much data a stream read returns.
- **`TS.READ`** ([#3896](https://github.com/redis/go-redis/pull/3896)), **`TS.QUERYLABELS`** ([#3926](https://github.com/redis/go-redis/pull/3926)), **`TS.NRANGE` / `TS.NREVRANGE`** ([#3870](https://github.com/redis/go-redis/pull/3870)) with multiple aggregators per key ([#3937](https://github.com/redis/go-redis/pull/3937)), and **`EXCLUDEEMPTY`** on `TS.MRANGE` / `TS.MREVRANGE` ([#3912](https://github.com/redis/go-redis/pull/3912)) — new time-series query surface.
- **`FT.ALIASLIST`** ([#3925](https://github.com/redis/go-redis/pull/3925)), **`COLLECT` reducer for `FT.AGGREGATE`** ([#3886](https://github.com/redis/go-redis/pull/3886)), **`RERANK` on HNSW vector fields in `FT.CREATE`** ([#3927](https://github.com/redis/go-redis/pull/3927)), and **`FT.HYBRID` timeout warnings** ([#3911](https://github.com/redis/go-redis/pull/3911)) — search coverage.

### Cross-SDK Aligned Defaults

Default configuration values now follow the cross-SDK configuration proposal shared by all Redis client libraries ([#3918](https://github.com/redis/go-redis/pull/3918)):

| Setting | Old default | New default |
|---|---|---|
| `ReadTimeout` / `WriteTimeout` | 3s | 5s |
| Retry backoff (min/max) | 8ms / 512ms | 10ms / 1s |
| Cluster state reload interval | 10s | 60s |
| TCP keep-alive | 5min period | 30s idle / 5s interval / 3 probes (`net.KeepAliveConfig`) |

Applications that set these values explicitly are unaffected; applications relying on the old defaults inherit the new ones.

### Data-Race and Parser Hardening Sweep

A systematic audit fixed data races across the client — hooks (`AddHook`, [#3868](https://github.com/redis/go-redis/pull/3868)), `Ring.SetAddrs` ([#3862](https://github.com/redis/go-redis/pull/3862)), cluster node slices ([#3861](https://github.com/redis/go-redis/pull/3861)), pub/sub reconnect ([#3906](https://github.com/redis/go-redis/pull/3906)), maintenance notifications ([#3894](https://github.com/redis/go-redis/pull/3894), [#3872](https://github.com/redis/go-redis/pull/3872)), pool handoff ([#3876](https://github.com/redis/go-redis/pull/3876)), and `redisotel` ([#3881](https://github.com/redis/go-redis/pull/3881)) — and hardened the RESP parsers against malformed or unexpected replies: over-reads on nil replies ([#3874](https://github.com/redis/go-redis/pull/3874)), integer overflow when skipping map/attribute bodies ([#3877](https://github.com/redis/go-redis/pull/3877)), unhashable RESP3 map keys ([#3873](https://github.com/redis/go-redis/pull/3873)), odd-length flat replies ([#3900](https://github.com/redis/go-redis/pull/3900)), mismatched declared array lengths ([#3907](https://github.com/redis/go-redis/pull/3907)), unexpected extra reply frames ([#3884](https://github.com/redis/go-redis/pull/3884)), and nil elements in numeric/bool slice replies ([#3922](https://github.com/redis/go-redis/pull/3922)).

### PubSub `Receive` Hang Fix

`PeekPushNotificationName` blocked until 36 bytes were buffered, so a short subscribe confirmation (channel name of six or fewer characters) on an otherwise idle connection hung `PubSub.Receive` forever — a regression introduced in 9.20.1 by [#3842](https://github.com/redis/go-redis/pull/3842). The peek now parses whatever is already buffered and only waits for one more byte when the frame prefix is valid but incomplete. Fixes [#3935](https://github.com/redis/go-redis/issues/3935).

([#3936](https://github.com/redis/go-redis/pull/3936)) by [@ndyakov](https://github.com/ndyakov)

### Correct Cluster Transaction Retries

The cluster transaction pipeline treated a `MULTI`...`EXEC` block as independently retryable commands, which could scatter a transaction across nodes or send malformed transactions on retry. Redirects (`MOVED`/`ASK`/`TRYAGAIN`) and aborts are now handled at the whole-transaction level, matching Redis transaction semantics: the transaction is re-routed and retried as a unit, never partially ([#3909](https://github.com/redis/go-redis/pull/3909)) by [@cxljs](https://github.com/cxljs).

### Credential Redaction in Command Tracing

`rediscmd.AppendCmd` — used by `redisotel` and `rediscensus` to render commands into span attributes — now redacts credential arguments as `<redacted>`: `AUTH`, `HELLO ... AUTH`, `CONFIG SET` of `requirepass` / `masterauth` / TLS key passphrases, `ACL SETUSER` password rules, and `MIGRATE ... AUTH`/`AUTH2`. The client sends `HELLO ... AUTH` on every handshake and `AUTH` on every streaming-credentials rotation through the regular hook chain, so tracing hooks previously captured credentials even when the application never issued an auth command itself ([#3939](https://github.com/redis/go-redis/pull/3939)) by [@saddamr3e](https://github.com/saddamr3e).

## ✨ New Features

- **`HIMPORT` command family**: `HImportPrepare` / `HImportSet` / `HImportDiscard` / `HImportDiscardAll` with lazy per-connection fieldset prepare replay ([#3919](https://github.com/redis/go-redis/pull/3919)) by [@ndyakov](https://github.com/ndyakov)
- **`LMOVEM` / `BLMOVEM`**: move multiple list elements in one call, with `COUNT` (up to N) or `EXACTLY` (all-or-nothing) semantics via `LMoveMArgs` ([#3913](https://github.com/redis/go-redis/pull/3913)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`SUnionCard` / `SDiffCard`**: cardinality of set union/difference ([#3897](https://github.com/redis/go-redis/pull/3897)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`XRead` / `XReadGroup` `MAXCOUNT` / `MAXSIZE`**: bound stream read responses by entry count or payload size ([#3898](https://github.com/redis/go-redis/pull/3898)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`TS.READ`**: read samples from a series starting at a given timestamp, with `TSReadEarliest` (`-`), `TSReadLatest` (`+`), and `TSReadNew` (`$`) sentinels ([#3896](https://github.com/redis/go-redis/pull/3896)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`TS.QUERYLABELS`**: query label names/values across time series ([#3926](https://github.com/redis/go-redis/pull/3926)) by [@ndyakov](https://github.com/ndyakov)
- **`TS.NRANGE` / `TS.NREVRANGE`**: range queries across multiple series ([#3870](https://github.com/redis/go-redis/pull/3870)) by [@ofekshenawa](https://github.com/ofekshenawa), with multiple aggregators per key ([#3937](https://github.com/redis/go-redis/pull/3937)) by [@ndyakov](https://github.com/ndyakov)
- **`TS.MRANGE` / `TS.MREVRANGE` `EXCLUDEEMPTY`**: skip series with no samples in the result ([#3912](https://github.com/redis/go-redis/pull/3912)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.ALIASLIST`**: list all index aliases ([#3925](https://github.com/redis/go-redis/pull/3925)) by [@ndyakov](https://github.com/ndyakov)
- **`FT.AGGREGATE` `COLLECT` reducer**: collect grouped values into an array ([#3886](https://github.com/redis/go-redis/pull/3886)) by [@ndyakov](https://github.com/ndyakov)
- **`FT.CREATE` `RERANK`**: `RERANK` parameter on HNSW vector field definitions ([#3927](https://github.com/redis/go-redis/pull/3927)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.HYBRID` timeout warnings**: timeout warnings are now populated in hybrid search results ([#3911](https://github.com/redis/go-redis/pull/3911)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.HYBRID` KNN `SHARD_K_RATIO`** (Redis 8.8+): per-shard K ratio for KNN clauses ([#3841](https://github.com/redis/go-redis/pull/3841)) by [@ndyakov](https://github.com/ndyakov)

## 🐛 Bug Fixes

- **PubSub `Receive` hang**: peek push-notification names without demanding 36 buffered bytes, fixing a hang on short subscribe confirmations (fixes [#3935](https://github.com/redis/go-redis/issues/3935), regression from 9.20.1) ([#3936](https://github.com/redis/go-redis/pull/3936)) by [@ndyakov](https://github.com/ndyakov)
- **Cluster transactions**: re-route the whole tx pipeline on redirect/abort instead of per-command ([#3909](https://github.com/redis/go-redis/pull/3909)) by [@cxljs](https://github.com/cxljs)
- **Credential leak in traces**: `rediscmd.AppendCmd` redacts credential arguments (`AUTH`, `HELLO ... AUTH`, `CONFIG SET` secret params, `ACL SETUSER` password rules, `MIGRATE AUTH`/`AUTH2`), so `redisotel` / `rediscensus` span attributes no longer contain passwords ([#3939](https://github.com/redis/go-redis/pull/3939)) by [@saddamr3e](https://github.com/saddamr3e)
- **`WaitAOF` return type**: returns `*IntSliceCmd` matching the two-integer `WAITAOF` reply ([#3888](https://github.com/redis/go-redis/pull/3888)) by [@CipherN9](https://github.com/CipherN9)
- **`Ring.Publish` routing**: publish to the shard that owns the topic instead of a round-robined one ([#3893](https://github.com/redis/go-redis/pull/3893)) by [@dkindel](https://github.com/dkindel)
- **Pool `OnRemove` hooks**: fire `OnRemove` on `putConn` eviction paths so removal hooks see every evicted connection ([#3932](https://github.com/redis/go-redis/pull/3932)) by [@cxljs](https://github.com/cxljs)
- **`UniversalClient` `InfoMap`**: added `InfoMap` to the `Cmdable` interface ([#3904](https://github.com/redis/go-redis/pull/3904)) by [@nazarli-shabnam](https://github.com/nazarli-shabnam)
- **`SlowLogGet` context**: pass the caller's context instead of a background one ([#3915](https://github.com/redis/go-redis/pull/3915)) by [@sonnemusk](https://github.com/sonnemusk)
- **`ModuleLoadex` nil config**: return an error instead of panicking on nil config ([#3916](https://github.com/redis/go-redis/pull/3916)) by [@sonnemusk](https://github.com/sonnemusk)
- **`ParseURL` IPv6 hosts**: keep single brackets for IPv6 hosts without a port ([#3882](https://github.com/redis/go-redis/pull/3882)) by [@sueun-dev](https://github.com/sueun-dev)
- **`ParseURL` durations**: treat unit durations `<= 0` as disabled ([#3866](https://github.com/redis/go-redis/pull/3866)) by [@sueun-dev](https://github.com/sueun-dev)
- **Nil `*uint8` encoding**: encode nil `*uint8` as `"0"` like other numeric pointers ([#3869](https://github.com/redis/go-redis/pull/3869)) by [@sueun-dev](https://github.com/sueun-dev)
- **`JSONSliceCmd` read errors**: return the read error from `readReply` instead of swallowing it ([#3903](https://github.com/redis/go-redis/pull/3903)) by [@saddamr3e](https://github.com/saddamr3e)
- **RESP parser hardening**: reconcile declared entry-array lengths ([#3907](https://github.com/redis/go-redis/pull/3907)), handle nil elements in int/uint/bool slice parsers ([#3922](https://github.com/redis/go-redis/pull/3922)), drain unexpected reply frames ([#3884](https://github.com/redis/go-redis/pull/3884)), reject odd-length flat replies in Z/KeyValue parsers ([#3900](https://github.com/redis/go-redis/pull/3900)), avoid int overflow when skipping map/attr bodies ([#3877](https://github.com/redis/go-redis/pull/3877)), don't over-read nil replies in `Reader.Discard` ([#3874](https://github.com/redis/go-redis/pull/3874)) by [@saddamr3e](https://github.com/saddamr3e); reject unhashable keys in RESP3 map parsing ([#3873](https://github.com/redis/go-redis/pull/3873)) by [@iabdullah215](https://github.com/iabdullah215)
- **Data races**: hook state during `AddHook` ([#3868](https://github.com/redis/go-redis/pull/3868)), `onNewNode` during `Ring.SetAddrs` ([#3862](https://github.com/redis/go-redis/pull/3862)), shared masters/slaves slices in cluster ([#3861](https://github.com/redis/go-redis/pull/3861)), shared `opt.Addr` during pub/sub reconnect ([#3906](https://github.com/redis/go-redis/pull/3906)), `clusterStateReloadCallback` in maintnotifications ([#3894](https://github.com/redis/go-redis/pull/3894)), conn reader in `isHealthyConn` during handoff ([#3876](https://github.com/redis/go-redis/pull/3876)) by [@saddamr3e](https://github.com/saddamr3e); handoff race window in maintnotifications ([#3872](https://github.com/redis/go-redis/pull/3872)) by [@ndyakov](https://github.com/ndyakov)
- **`redisotel`**: use `ObservableCounter` for cumulative pool stats ([#3914](https://github.com/redis/go-redis/pull/3914)) by [@Solaris-star](https://github.com/Solaris-star); avoid a data race on shared attributes during `MinIdleConns` warmup ([#3881](https://github.com/redis/go-redis/pull/3881)) by [@ndyakov](https://github.com/ndyakov)

## 🧰 Maintenance

- **Cross-SDK default alignment**: new defaults for timeouts, retry backoff, cluster state reload, and TCP keep-alive ([#3918](https://github.com/redis/go-redis/pull/3918)) by [@ndyakov](https://github.com/ndyakov)
- **CI on Redis 8.10**: 8.10 made the default test version ([#3920](https://github.com/redis/go-redis/pull/3920)) with version gating by major.minor ([#3908](https://github.com/redis/go-redis/pull/3908)) by [@ofekshenawa](https://github.com/ofekshenawa); the test stack now runs the GA `redislabs/client-libs-test:8.10.0` image and 8.8 was dropped from the CI matrix ([#3940](https://github.com/redis/go-redis/pull/3940))
- **Type-safe atomics**: use typed `sync/atomic` value types ([#3860](https://github.com/redis/go-redis/pull/3860)) and remove the dead `assertUnstableCommand` RESP3 path ([#3928](https://github.com/redis/go-redis/pull/3928)) by [@cxljs](https://github.com/cxljs)
- **Docs**: clarify that `ExpireTime` / `PExpireTime` return Unix timestamps ([#3917](https://github.com/redis/go-redis/pull/3917)) by [@sonnemusk](https://github.com/sonnemusk); remove a duplicate example step ([#3875](https://github.com/redis/go-redis/pull/3875)) by [@andy-stark-redis](https://github.com/andy-stark-redis)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@andy-stark-redis](https://github.com/andy-stark-redis), [@CipherN9](https://github.com/CipherN9), [@cxljs](https://github.com/cxljs), [@dkindel](https://github.com/dkindel), [@iabdullah215](https://github.com/iabdullah215), [@nazarli-shabnam](https://github.com/nazarli-shabnam), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@saddamr3e](https://github.com/saddamr3e), [@Solaris-star](https://github.com/Solaris-star), [@sonnemusk](https://github.com/sonnemusk), [@sueun-dev](https://github.com/sueun-dev)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.21.0...v9.22.0-beta.1

# 9.21.0 (2026-06-18)

This is a minor release adding new features and bug fixes. There are no breaking changes; upgrading from 9.20.x is a drop-in replacement.

## 🚀 Highlights

### Zero-copy `GetToBuffer` / `SetFromBuffer`

Two new `StringCmdable` methods let callers read and write Redis string values directly into and from pre-allocated byte buffers, eliminating the per-call payload allocation that `Get`/`Set` incur:

```go
GetToBuffer(ctx, key, buf) *ZeroCopyStringCmd   // reads into buf; ZeroCopyStringCmd { Val() int; Bytes() []byte; Result() (int, error) }
SetFromBuffer(ctx, key, buf) *StatusCmd
```

`GetToBuffer` decodes the bulk reply straight into the caller-owned `buf` (no intermediate allocation); a buffer that is too small returns an error after draining the payload, so the connection stays aligned for the next reply. `SetFromBuffer` is provided for API symmetry — it dispatches to the same `[]byte` writer path as `Set(ctx, key, buf, 0)` and produces byte-identical output on the wire. Available on `*Client`, `*ClusterClient`, `*Ring`, `*Conn` and `Pipeliner`.

([#3834](https://github.com/redis/go-redis/pull/3834)) by [@ndyakov](https://github.com/ndyakov)

### Explicit `LIMIT 0` for stream trimming

Redis treats `XTRIM`/`XADD` approximate-trim (`~`) `LIMIT 0` as "disable the trimming effort cap entirely", which differs from omitting `LIMIT` (the implicit `100 * stream-node-max-entries` default). The command builders previously only emitted `LIMIT` when `limit > 0`, so callers could never send an explicit `LIMIT 0`. Following the `KeepTTL = -1` precedent, the new `XTrimLimitDisabled = -1` sentinel now emits an explicit `LIMIT 0`; `limit == 0` keeps the historical no-`LIMIT` behavior, so existing callers produce byte-identical commands.

([#3848](https://github.com/redis/go-redis/pull/3848)) by [@TheRealMal](https://github.com/TheRealMal)

## ✨ New Features

- **Zero-copy buffer string commands**: new `GetToBuffer` / `SetFromBuffer` on `StringCmdable` and the `ZeroCopyStringCmd` result type, reading/writing string values into caller-owned buffers without per-call payload allocation ([#3834](https://github.com/redis/go-redis/pull/3834)) by [@ndyakov](https://github.com/ndyakov)
- **`XTrimLimitDisabled` sentinel**: `XTRIM`/`XADD` approximate trimming can now send an explicit `LIMIT 0` to disable the trim effort cap, via the new `XTrimLimitDisabled = -1` sentinel ([#3848](https://github.com/redis/go-redis/pull/3848)) by [@TheRealMal](https://github.com/TheRealMal)
- **PubSub health-check timeouts**: `channel.initHealthCheck` now bounds the `Ping` it issues with a fresh per-check timeout context (the exported `pingTimeout` / `reconnectTimeout`) instead of `context.TODO()`, so a stuck health-check Ping can no longer block indefinitely ([#3819](https://github.com/redis/go-redis/pull/3819)) by [@abdellani](https://github.com/abdellani)
- **Skip redundant `UNWATCH` in `Tx.Close`**: a transaction now tracks whether a `WATCH` is still active (`watchArmed`) and only issues `UNWATCH` on `Close` when it is, removing an extra round trip on the common `WATCH`/.../`EXEC` and no-key `Watch` paths while never returning a connection to the pool with an active watch ([#3854](https://github.com/redis/go-redis/pull/3854)) by [@fcostaoliveira](https://github.com/fcostaoliveira)

## 🐛 Bug Fixes

- **`maintnotifications` `ModeAuto` fail-open**: `ModeAuto` now stays fail-open when the server does not support maintenance notifications — connections are retired and tracking is guarded during downgrade so the client keeps working instead of erroring ([#3853](https://github.com/redis/go-redis/pull/3853)) by [@terrorobe](https://github.com/terrorobe)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@abdellani](https://github.com/abdellani), [@fcostaoliveira](https://github.com/fcostaoliveira), [@ndyakov](https://github.com/ndyakov), [@terrorobe](https://github.com/terrorobe), [@TheRealMal](https://github.com/TheRealMal)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.20.1...v9.21.0

# 9.20.1 (2026-06-11)

This is a patch release containing bug fixes only. There are no new features or breaking changes; upgrading from 9.20.0 is a drop-in replacement.

## 🚀 Highlights

### RESP3 pub/sub message loss fixed

`PeekPushNotificationName` previously inspected only the bytes already buffered by `bufio`, so when a push frame header straddled a buffer fill boundary it could return a **truncated** notification name (e.g. `"messa"` instead of `"message"`). The push processor then mis-routed the frame and `ReadReply` silently dropped it, causing intermittent RESP3 pub/sub message loss. The peek now grows its window (36 bytes → up to 4 KiB) and reads more from the connection until the header is complete, cleanly separating incomplete prefixes from corrupt frames (including overflow-safe bulk-length handling). Fixes [#3839](https://github.com/redis/go-redis/issues/3839).

([#3842](https://github.com/redis/go-redis/pull/3842)) by [@ndyakov](https://github.com/ndyakov)

## 🐛 Bug Fixes

- **RESP3 push peeking**: `PeekPushNotificationName` no longer returns a truncated notification name when a push frame header spans a buffer boundary, preventing silent RESP3 pub/sub message loss (fixes [#3839](https://github.com/redis/go-redis/issues/3839)) ([#3842](https://github.com/redis/go-redis/pull/3842)) by [@ndyakov](https://github.com/ndyakov)
- **`FT.HYBRID` vector params**: Vector data is now always sent via `PARAMS` with auto-generated param names (`__vector_param_N`, with collision avoidance) when `VectorParamName` is omitted, since Redis no longer accepts inline vector blobs; the `FTHybridOptions.Params` map is no longer mutated, so the same options struct can be reused across calls ([#3844](https://github.com/redis/go-redis/pull/3844)) by [@ndyakov](https://github.com/ndyakov)
- **`CLUSTER SHARDS` forward compatibility**: Unknown shard- and node-level attributes in the `CLUSTER SHARDS` reply are now skipped via `DiscardNext()` instead of erroring, so clients keep working when the server introduces new fields ([#3843](https://github.com/redis/go-redis/pull/3843)) by [@madolson](https://github.com/madolson)
- **PubSub double reconnect**: `PubSub.releaseConn` no longer reconnects twice when a connection is both unusable (or pending handoff) and reports a bad-connection error, avoiding a wasted connection establish-then-close cycle ([#3833](https://github.com/redis/go-redis/pull/3833)) by [@cxljs](https://github.com/cxljs)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@cxljs](https://github.com/cxljs), [@madolson](https://github.com/madolson), [@ndyakov](https://github.com/ndyakov)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.20.0...v9.20.1

# 9.20.0 (2026-05-28)

## 🚀 Highlights

### Redis 8.8 Support

This release adds support for **Redis 8.8**. The README's supported-versions list now includes Redis 8.8 alongside 8.0/8.2/8.4, and CI exercises the `8.8-rc1` client-libs-test image across the full suite (Makefile, build workflow, doctests, run-tests action, and docker-compose).

Coverage for the new commands that ship in the 8.x line, rounded out in this release:

- **`AR*` array data type** ([#3813](https://github.com/redis/go-redis/pull/3813)) — new array data structure, exposed via the `ArrayCmdable` interface (see the experimental-features highlight below).
- **`INCREX`** ([#3816](https://github.com/redis/go-redis/pull/3816)) — atomic increment with expiration in a single round-trip.
- **`XNACK`** ([#3790](https://github.com/redis/go-redis/pull/3790)) — explicit negative-acknowledge of pending stream entries.
- **`XAUTOCLAIM` PEL deletes** ([#3798](https://github.com/redis/go-redis/pull/3798)) — `XAUTOCLAIM`/`XAUTOCLAIMJUSTID` now return the list of deleted message IDs from the pending entries list.
- **`TS.RANGE` multiple aggregators** ([#3791](https://github.com/redis/go-redis/pull/3791)) — `TS.RANGE`/`TS.REVRANGE`/`TS.MRANGE`/`TS.MREVRANGE` accept multiple aggregators in a single call.
- **`Z(UNION|INTER|DIFF)` `COUNT` aggregator** ([#3802](https://github.com/redis/go-redis/pull/3802)) — `COUNT` reducer for sorted-set set operations.
- **`JSON.SET FPHA`** ([#3797](https://github.com/redis/go-redis/pull/3797)) — new `FPHA` argument that specifies the floating-point type for homogeneous FP arrays.

CI image bump ([#3814](https://github.com/redis/go-redis/pull/3814)) by [@ofekshenawa](https://github.com/ofekshenawa). Command coverage contributions by [@cxljs](https://github.com/cxljs), [@elena-kolevska](https://github.com/elena-kolevska), [@Khukharr](https://github.com/Khukharr), [@ndyakov](https://github.com/ndyakov), and [@ofekshenawa](https://github.com/ofekshenawa).

### Stable RESP3 for RediSearch (`UnstableResp3` deprecated)

`FT.SEARCH`, `FT.AGGREGATE`, `FT.INFO`, `FT.SPELLCHECK`, and `FT.SYNDUMP` now parse RESP3 (map) responses into the same typed result objects as RESP2 — `Val()` and `Result()` work uniformly on both protocols, no flag required. Previously, RESP3 search responses required `UnstableResp3: true` and were returned as opaque maps accessible only via `RawResult()` / `RawVal()`.

As a result, the `UnstableResp3` option is now a **no-op** across every options struct (`Options`, `ClusterOptions`, `UniversalOptions`, `FailoverOptions`, `RingOptions`) and has been marked `// Deprecated:`. The field is retained for backwards compatibility — existing code that sets `UnstableResp3: true` will continue to compile and behave identically — but it will be removed in a future release and new code should not set it. `RawResult()` / `RawVal()` continue to work for callers that prefer the raw RESP payload.

([#3741](https://github.com/redis/go-redis/pull/3741)) by [@ndyakov](https://github.com/ndyakov)

### Experimental Array Data Structure Commands

Adds an experimental `ArrayCmdable` interface with the `AR*` command family (`ARSet`, `ARGet`, `ARGetRange`, `ARMSet`, `ARMGet`, `ARDel`, `ARDelRange`, `ARScan`, `ARSeek`, `ARNext`, `ARLastItems`, `ARGrep`, `ARGrepWithValues`, `ARInfo`/`ARInfoFull`, and typed reducers `AROpSum`/`AROpMin`/`AROpMax`/`AROpAnd`/`AROpOr`/`AROpXor`/`AROpMatch`/`AROpUsed`) for working with Redis 8.8's new array data type. **API is experimental and may change in a future release.**

([#3813](https://github.com/redis/go-redis/pull/3813)) by [@cxljs](https://github.com/cxljs)

## ✨ New Features

- **RESP3 search parser**: First-class RESP3 parsing for `FT.SEARCH`/`FT.AGGREGATE`/`FT.INFO`/`FT.SPELLCHECK`/`FT.SYNDUMP` responses with backwards compatibility for RESP2 ([#3741](https://github.com/redis/go-redis/pull/3741)) by [@ndyakov](https://github.com/ndyakov)
- **INCREX**: New `INCREX` command support — atomic increment with expiration ([#3816](https://github.com/redis/go-redis/pull/3816)) by [@ndyakov](https://github.com/ndyakov)
- **XNACK**: Client support for the `XNACK` stream command for explicitly negative-acknowledging pending entries ([#3790](https://github.com/redis/go-redis/pull/3790)) by [@elena-kolevska](https://github.com/elena-kolevska)
- **TS range multiple aggregators**: `TS.RANGE`/`TS.REVRANGE`/`TS.MRANGE`/`TS.MREVRANGE` now accept multiple aggregators in a single call ([#3791](https://github.com/redis/go-redis/pull/3791)) by [@elena-kolevska](https://github.com/elena-kolevska)
- **`XAutoClaim` deleted IDs**: `XAUTOCLAIM`/`XAUTOCLAIMJUSTID` now return the list of deleted message IDs from the PEL ([#3798](https://github.com/redis/go-redis/pull/3798)) by [@Khukharr](https://github.com/Khukharr)
- **`JSON.SET FPHA`**: `JSON.SET` accepts a new `FPHA` argument that specifies the floating-point type for homogeneous floating-point arrays ([#3797](https://github.com/redis/go-redis/pull/3797)) by [@ndyakov](https://github.com/ndyakov)
- **Sorted-set union/intersection COUNT**: `ZUNION`/`ZINTER`/`ZDIFF` aggregator now supports `COUNT` ([#3802](https://github.com/redis/go-redis/pull/3802)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **`FT.HYBRID` vector validation**: Validates hybrid-search vector input types and adds proper typed vector parameters ([#3756](https://github.com/redis/go-redis/pull/3756)) by [@DengY11](https://github.com/DengY11)
- **Cluster pool wait stats**: `ClusterClient.PoolStats()` now accumulates `WaitCount` and `WaitDurationNs` across all node pools (previously always zero) ([#3809](https://github.com/redis/go-redis/pull/3809)) by [@LINKIWI](https://github.com/LINKIWI)

## 🐛 Bug Fixes

- **TLS-only Cluster PubSub**: `CLUSTER SLOTS` port-0 entries now fall back to the origin endpoint's port, fixing `dial tcp <ip>:0: connection refused` on TLS-only clusters started with `--port 0 --tls-port <port>` (fixes [#3726](https://github.com/redis/go-redis/issues/3726)) ([#3828](https://github.com/redis/go-redis/pull/3828)) by [@ndyakov](https://github.com/ndyakov)
- **Sharded PubSub reconnect routing**: `PubSub.conn()` now passes both regular (`c.channels`) and sharded (`c.schannels`) channels into the per-PubSub `newConn` closure. Previously, `ClusterClient.SSubscribe`-only PubSubs reconnected to a random node (because the routing closure saw an empty channel list), the `SSUBSCRIBE` was sent to the wrong shard, and the resulting `MOVED` reply was silently dropped ([#3829](https://github.com/redis/go-redis/pull/3829)) by [@ndyakov](https://github.com/ndyakov)
- **ClusterClient `Watch` retry**: User errors returned from a `Watch` callback are no longer subjected to cluster-retry classification; transient cluster errors still retry, but a callback returning e.g. `net.ErrClosed` short-circuits immediately ([#3821](https://github.com/redis/go-redis/pull/3821)) by [@obiyang](https://github.com/obiyang)
- **Sentinel concurrent-probe leak**: `MasterAddr`'s concurrent sentinel probe now closes the non-winning sentinel clients instead of leaking them ([#3827](https://github.com/redis/go-redis/pull/3827)) by [@cxljs](https://github.com/cxljs)
- **Sentinel rediscovery loop on master-only setups**: `replicaAddrs` no longer tears down the cached sentinel client when the replica list is empty, eliminating a continuous rediscovery loop on master-only Sentinel deployments that flooded logs and added per-operation latency ([#3795](https://github.com/redis/go-redis/pull/3795)) by [@shahyash2609](https://github.com/shahyash2609)
- **Pool `CloseConn` hooks**: `Pool.CloseConn` now triggers registered hooks, fixing a memory leak when connections are closed explicitly rather than via the normal removal path ([#3818](https://github.com/redis/go-redis/pull/3818)) by [@ndyakov](https://github.com/ndyakov)
- **Dial TCP error redirection**: Wrapped `dial tcp` errors are now correctly classified as redirectable so cluster routing can recover from a single unreachable node ([#3810](https://github.com/redis/go-redis/pull/3810)) by [@vladisa88](https://github.com/vladisa88)
- **Pool `Close` health checks**: `ConnPool.Close` now only runs health checks against idle connections, avoiding spurious activity on connections still in use ([#3805](https://github.com/redis/go-redis/pull/3805)) by [@ndyakov](https://github.com/ndyakov)
- **VLinks return type**: Fixed the return type of `VLINKS`/`VLINKSWITHSCORES` vector-set replies ([#3820](https://github.com/redis/go-redis/pull/3820)) by [@romanpovol](https://github.com/romanpovol)

## 🧪 Testing & Infrastructure

- **Flaky tests**: Stabilized several flaky tests in the sentinel and pool suites ([#3815](https://github.com/redis/go-redis/pull/3815)) by [@ndyakov](https://github.com/ndyakov)
- **Sentinel failover metric race**: Fixed a data race in the sentinel failover metric test ([#3824](https://github.com/redis/go-redis/pull/3824)) by [@cxljs](https://github.com/cxljs)
- **`waitForSentinelClusterStable` post-conditions**: The sentinel test harness now waits for replicas to be fully connected (not just present in the count) and is robust to randomized spec ordering after failover specs, eliminating an intermittent `Expected master to equal slave` flake ([#3830](https://github.com/redis/go-redis/pull/3830)) by [@ndyakov](https://github.com/ndyakov)
- **`govulncheck` workflow**: New scheduled GitHub Actions workflow runs `govulncheck` on every push, PR, and weekly, surfacing newly disclosed Go vulnerabilities even when no code changes ([#3779](https://github.com/redis/go-redis/pull/3779)) by [@solardome](https://github.com/solardome)
- **CI Redis 8.8-rc1**: CI now exercises the 8.8-rc1 Redis image ([#3814](https://github.com/redis/go-redis/pull/3814)) by [@ofekshenawa](https://github.com/ofekshenawa)

## 🧰 Maintenance

- **`Cmd.Slot()` lookup refactor**: Caches the per-command `CommandInfo` and short-circuits keyless commands before the switch dispatch, removing redundant `Peek` calls ([#3804](https://github.com/redis/go-redis/pull/3804)) by [@retr0-kernel](https://github.com/retr0-kernel)
- **stdlib `math/rand`**: Replaced `internal/rand` with `math/rand` from the standard library now that the minimum Go version is 1.24 ([#3823](https://github.com/redis/go-redis/pull/3823)) by [@cxljs](https://github.com/cxljs)
- **ConnPool queue channel**: Removed the unused queue channel from `ConnPool`, trimming the pool's footprint ([#3826](https://github.com/redis/go-redis/pull/3826)) by [@cxljs](https://github.com/cxljs)
- **Extra packages LICENSE**: Added a LICENSE file to each `extra/*` package ([#3817](https://github.com/redis/go-redis/pull/3817)) by [@ndyakov](https://github.com/ndyakov)
- **README & CI image**: Documentation refresh and bumped the default CI image tag ([#3822](https://github.com/redis/go-redis/pull/3822)) by [@ndyakov](https://github.com/ndyakov)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@cxljs](https://github.com/cxljs), [@DengY11](https://github.com/DengY11), [@elena-kolevska](https://github.com/elena-kolevska), [@Khukharr](https://github.com/Khukharr), [@LINKIWI](https://github.com/LINKIWI), [@ndyakov](https://github.com/ndyakov), [@obiyang](https://github.com/obiyang), [@ofekshenawa](https://github.com/ofekshenawa), [@retr0-kernel](https://github.com/retr0-kernel), [@romanpovol](https://github.com/romanpovol), [@shahyash2609](https://github.com/shahyash2609), [@solardome](https://github.com/solardome), [@vladisa88](https://github.com/vladisa88)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.19.0...v9.20.0

# 9.19.0 (2026-04-27)

## 🚀 Highlights

### FIPS-Compatible Script Helper

`Script` now supports a FIPS-safe execution mode that avoids client-side SHA-1 computation, which is blocked in strict FIPS environments. A new `NewScriptServerSHA` constructor uses `SCRIPT LOAD` to obtain and cache the digest from the server, then runs commands via `EVALSHA`/`EVALSHA_RO`. Falls back to `EVAL`/`EVALRO` if loading fails, and transparently retries once on `NOSCRIPT`. The default behavior is unchanged for existing users.

([#3700](https://github.com/redis/go-redis/pull/3700)) by [@chaitanyabodlapati](https://github.com/chaitanyabodlapati)

### FT.AGGREGATE Step-Based Pipeline Builder

Added a new step-based `FT.AGGREGATE` pipeline API via `FTAggregateOptions.Steps`, allowing `LOAD`, `APPLY`, `GROUPBY`, and `SORTBY` (with per-step `MAX`) to be repeated and interleaved in arbitrary order — matching Redis's native multi-stage aggregation semantics. The legacy `Load`/`Apply`/`GroupBy`/`SortBy`/`SortByMax` fields are now deprecated.

([#3782](https://github.com/redis/go-redis/pull/3782)) by [@ndyakov](https://github.com/ndyakov)

### Raw RESP Protocol Access

Added `DoRaw` and `DoRawWriteTo` methods for executing arbitrary commands and reading the raw RESP response. Useful for proxying, custom protocol inspection, and working with commands not yet wrapped by go-redis.

([#3713](https://github.com/redis/go-redis/pull/3713)) by [@ofekshenawa](https://github.com/ofekshenawa)

### Configurable Dial Retry Backoff

Added `DialerRetryBackoff` option (plumbed through `Options`, `ClusterOptions`, `RingOptions`, `FailoverOptions`) to let callers customize the delay between failed dial attempts. Helpers `DialRetryBackoffConstant` and `DialRetryBackoffExponential` (with jitter and cap) are provided out of the box. Dial timeout is now also applied **per attempt** rather than across all retries.

([#3706](https://github.com/redis/go-redis/pull/3706), [#3705](https://github.com/redis/go-redis/pull/3705)) by [@mwhooker](https://github.com/mwhooker)

## ✨ New Features

- **FT.AGGREGATE Steps**: Step-based pipeline builder for `FT.AGGREGATE` with support for repeated/interleaved `LOAD`, `APPLY`, `GROUPBY`, and `SORTBY` stages ([#3782](https://github.com/redis/go-redis/pull/3782)) by [@ndyakov](https://github.com/ndyakov)
- **VectorSet commands**: Added `VISMEMBER` and `WITHATTRIBS` support ([#3753](https://github.com/redis/go-redis/pull/3753)) by [@romanpovol](https://github.com/romanpovol)
- **FIPS-safe Script**: `NewScriptServerSHA` uses `SCRIPT LOAD` to obtain the digest from the server, avoiding client-side SHA-1 ([#3700](https://github.com/redis/go-redis/pull/3700)) by [@chaitanyabodlapati](https://github.com/chaitanyabodlapati)
- **Raw RESP access**: `DoRaw` and `DoRawWriteTo` for raw RESP protocol access ([#3713](https://github.com/redis/go-redis/pull/3713)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **Dial retry backoff**: `DialerRetryBackoff` function option with constant and exponential helpers ([#3706](https://github.com/redis/go-redis/pull/3706)) by [@mwhooker](https://github.com/mwhooker)
- **Typed NOSCRIPT error**: Redis `NOSCRIPT` replies are now surfaced as a typed error for easier handling ([#3738](https://github.com/redis/go-redis/pull/3738)) by [@LINKIWI](https://github.com/LINKIWI)
- **PubSub ClientSetName**: Added `ClientSetName` method to `PubSub` ([#3727](https://github.com/redis/go-redis/pull/3727)) by [@Flack74](https://github.com/Flack74)
- **ReplicaOf**: New `ReplicaOf` method replaces the deprecated `SlaveOf` ([#3720](https://github.com/redis/go-redis/pull/3720)) by [@Copilot](https://github.com/apps/copilot-swe-agent)
- **HSCAN BinaryUnmarshaler**: `HScan` now supports types implementing `encoding.BinaryUnmarshaler` ([#3768](https://github.com/redis/go-redis/pull/3768)) by [@Aaditya-dubey1](https://github.com/Aaditya-dubey1)

## 🐛 Bug Fixes

- **Auto hostname type detection**: Improved endpoint type detection for maintenance notifications using DNS-based classification; handles empty hosts and expanded private-IP ranges ([#3789](https://github.com/redis/go-redis/pull/3789)) by [@ndyakov](https://github.com/ndyakov)
- **HELLO fallback**: Don't send `CLIENT MAINT_NOTIFICATIONS` handshake when `HELLO` fails and connection falls back to RESP2; fail fast when explicitly enabled with RESP3 ([#3788](https://github.com/redis/go-redis/pull/3788)) by [@ndyakov](https://github.com/ndyakov)
- **Dial TCP retry**: `ShouldRetry` now treats `net.OpError` with `Op == "dial"` timeout errors as safe to retry since no command was sent ([#3787](https://github.com/redis/go-redis/pull/3787)) by [@vladisa88](https://github.com/vladisa88)
- **wrappedOnClose leak**: Fixed resource leak caused by repeatedly wrapping `baseClient` close logic; replaced with a bounded, concurrency-safe named-hook registry ([#3785](https://github.com/redis/go-redis/pull/3785)) by [@ndyakov](https://github.com/ndyakov)
- **Pool Close() on stale connections**: Suppress close errors (e.g., TLS `closeNotify` timeouts) for connections already dropped by the server due to idle timeout ([#3778](https://github.com/redis/go-redis/pull/3778)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **FIFO waiter ordering**: Fixed race in `ConnStateMachine.notifyWaiters` that could wake multiple waiters under a single mutex hold and violate FIFO ordering ([#3777](https://github.com/redis/go-redis/pull/3777)) by [@0x48core](https://github.com/0x48core)
- **Lua READONLY detection**: Detect `READONLY` errors embedded in Lua script error messages on read-only replicas so commands are correctly retried ([#3769](https://github.com/redis/go-redis/pull/3769)) by [@zhengjilei](https://github.com/zhengjilei)
- **VectorScoreSliceCmd RESP2**: Fixed `VSimWithScores`, `VSimWithArgsWithScores`, and `VLinksWithScores` which were broken on RESP2 connections returning flat arrays instead of maps ([#3767](https://github.com/redis/go-redis/pull/3767)) by [@Copilot](https://github.com/apps/copilot-swe-agent)
- **Closed connection handling**: Two fixes for closed connection handling in the pool ([#3764](https://github.com/redis/go-redis/pull/3764)) by [@cxljs](https://github.com/cxljs)
- **ZRangeArgs Rev**: Fixed `ZRangeArgs` with `Rev` + `ByScore`/`ByLex` incorrectly swapping `Start`/`Stop`, breaking `ZRANGESTORE` ([#3751](https://github.com/redis/go-redis/pull/3751)) by [@Copilot](https://github.com/apps/copilot-swe-agent)
- **OTel metric instrument types**: Fixed metric instrument types in `redisotel-native` ([#3743](https://github.com/redis/go-redis/pull/3743)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **Options.clone() data race**: Fixed data race when cloning `Options` ([#3739](https://github.com/redis/go-redis/pull/3739)) by [@rubensayshi](https://github.com/rubensayshi)
- **Connection closure metrics**: Fixed connection closure metrics and enabled all metric groups by default in `redisotel-native` ([#3735](https://github.com/redis/go-redis/pull/3735)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **OTel semconv v1.38.0**: Use metric definition from `otel/semconv/v1.38.0` in `redisotel-native` ([#3731](https://github.com/redis/go-redis/pull/3731)) by [@wzy9607](https://github.com/wzy9607)
- **SETNX semantics**: Use `SET ... NX` instead of the deprecated `SETNX` command ([#3723](https://github.com/redis/go-redis/pull/3723)) by [@ndyakov](https://github.com/ndyakov)
- **TIME keyless routing**: Mark `TIME` as a keyless command for correct cluster routing ([#3722](https://github.com/redis/go-redis/pull/3722)) by [@fatal10110](https://github.com/fatal10110)
- **Dial timeout per retry**: Dial timeout now applies per attempt instead of across all retry attempts combined ([#3705](https://github.com/redis/go-redis/pull/3705)) by [@mwhooker](https://github.com/mwhooker)
- **Cluster metrics attributes**: Fixed `pool.name` being appended per node, which corrupted and dropped user-provided custom attributes ([#3699](https://github.com/redis/go-redis/pull/3699)) by [@Jesse-Bonfire](https://github.com/Jesse-Bonfire)
- **initConn nil dereference**: Fixed nil pointer dereference and potential deadlock in `*baseClient.initConn()`; added explicit nil option guards to client constructors ([#3676](https://github.com/redis/go-redis/pull/3676)) by [@olde-ducke](https://github.com/olde-ducke)

## ⚡ Performance

- **RESP reader**: Optimized RESP reader by eliminating intermediate string allocations ([#3774](https://github.com/redis/go-redis/pull/3774)) by [@Aaditya-dubey1](https://github.com/Aaditya-dubey1)
- **Inline rendezvous hashing**: Replaced `github.com/dgryski/go-rendezvous` dependency with an in-repo implementation in `internal/hashtag`, reducing the dependency graph while preserving algorithm parity ([#3762](https://github.com/redis/go-redis/pull/3762)) by [@bigsk05](https://github.com/bigsk05)

## 🧪 Testing & Infrastructure

- **Release automation**: Added `repository`, `ref`, and `client-libs-test-image-tag` inputs to the `run-tests` composite action; `redis-version` is now optional so unstable builds use `REDIS_VERSION` from the Makefile ([#3749](https://github.com/redis/go-redis/pull/3749)) by [@dariaguy](https://github.com/dariaguy)
- **Go 1.24**: Updated minimum Go version to 1.24 and use `-compat=1.24` in release scripts ([#3714](https://github.com/redis/go-redis/pull/3714), [#3754](https://github.com/redis/go-redis/pull/3754)) by [@ndyakov](https://github.com/ndyakov), [@cxljs](https://github.com/cxljs)

## 🧰 Maintenance

- **Pool state machine**: Removed redundant `Conn.closed` atomic field in favor of the state machine's `StateClosed` ([#3783](https://github.com/redis/go-redis/pull/3783)) by [@cxljs](https://github.com/cxljs)
- **OTel SDK**: Updated OpenTelemetry SDK dependencies in `redisotel`/`redisotel-native` ([#3770](https://github.com/redis/go-redis/pull/3770)) by [@ndyakov](https://github.com/ndyakov)
- **Go 1.21+ built-ins**: Use `maps.Keys`, `slices.Collect`, `slices.Contains`, `clear()`, and `slices.SortFunc` instead of custom helpers ([#3758](https://github.com/redis/go-redis/pull/3758), [#3746](https://github.com/redis/go-redis/pull/3746)) by [@cxljs](https://github.com/cxljs)
- **HGetAll docs**: Added Go doc comment to `HGetAll` describing behavior and complexity ([#3776](https://github.com/redis/go-redis/pull/3776)) by [@0x48core](https://github.com/0x48core)
- **Docs links**: Fixed irrelevant docs links ([#3724](https://github.com/redis/go-redis/pull/3724)) by [@olzhas-sabiyev](https://github.com/olzhas-sabiyev)
- **Examples cleanup**: Removed throughput binary from examples ([#3733](https://github.com/redis/go-redis/pull/3733)) by [@ndyakov](https://github.com/ndyakov)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@0x48core](https://github.com/0x48core), [@Aaditya-dubey1](https://github.com/Aaditya-dubey1), [@Copilot](https://github.com/apps/copilot-swe-agent), [@Flack74](https://github.com/Flack74), [@Jesse-Bonfire](https://github.com/Jesse-Bonfire), [@LINKIWI](https://github.com/LINKIWI), [@bigsk05](https://github.com/bigsk05), [@chaitanyabodlapati](https://github.com/chaitanyabodlapati), [@cxljs](https://github.com/cxljs), [@dariaguy](https://github.com/dariaguy), [@fatal10110](https://github.com/fatal10110), [@mwhooker](https://github.com/mwhooker), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@olde-ducke](https://github.com/olde-ducke), [@olzhas-sabiyev](https://github.com/olzhas-sabiyev), [@romanpovol](https://github.com/romanpovol), [@rubensayshi](https://github.com/rubensayshi), [@vladisa88](https://github.com/vladisa88), [@wzy9607](https://github.com/wzy9607), [@zhengjilei](https://github.com/zhengjilei)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.18.0...v9.19.0

# 9.18.0 (2026-02-16)

## 🚀 Highlights

### Redis 8.6 Support

Added support for Redis 8.6, including new commands and features for streams idempotent production and HOTKEYS.

### Smart Client Handoff (Maintenance Notifications) for Cluster

This release introduces comprehensive support for Redis Cluster maintenance notifications via SMIGRATING/SMIGRATED push notifications. The client now automatically handles slot migrations by:
- **Relaxing timeouts during migration** (SMIGRATING) to prevent false failures
- **Triggering lazy cluster state reloads** upon completion (SMIGRATED)
- Enabling seamless operations during Redis Enterprise maintenance windows

([#3643](https://github.com/redis/go-redis/pull/3643)) by [@ndyakov](https://github.com/ndyakov)

### OpenTelemetry Native Metrics Support

Added comprehensive OpenTelemetry metrics support following the [OpenTelemetry Database Client Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/database/database-metrics/). The implementation uses a Bridge Pattern to keep the core library dependency-free while providing optional metrics instrumentation through the new `extra/redisotel-native` package.

**Metric groups include:**
- Command metrics: Operation duration with retry tracking
- Connection basic: Connection count and creation time
- Resiliency: Errors, handoffs, timeout relaxation
- Connection advanced: Wait time and use time
- Pubsub metrics: Published and received messages
- Stream metrics: Processing duration and maintenance notifications

([#3637](https://github.com/redis/go-redis/pull/3637)) by [@ofekshenawa](https://github.com/ofekshenawa)

## ✨ New Features

- **HOTKEYS Commands**: Added support for Redis HOTKEYS feature for identifying hot keys based on CPU consumption and network utilization ([#3695](https://github.com/redis/go-redis/pull/3695)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **Streams Idempotent Production**: Added support for Redis 8.6+ Streams Idempotent Production with `ProducerID`, `IdempotentID`, `IdempotentAuto` in `XAddArgs` and new `XCFGSET` command ([#3693](https://github.com/redis/go-redis/pull/3693)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **NaN Values for TimeSeries**: Added support for NaN (Not a Number) values in Redis time series commands ([#3687](https://github.com/redis/go-redis/pull/3687)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **DialerRetries Options**: Added `DialerRetries` and `DialerRetryTimeout` to `ClusterOptions`, `RingOptions`, and `FailoverOptions` ([#3686](https://github.com/redis/go-redis/pull/3686)) by [@naveenchander30](https://github.com/naveenchander30)
- **ConnMaxLifetimeJitter**: Added jitter configuration to distribute connection expiration times and prevent thundering herd ([#3666](https://github.com/redis/go-redis/pull/3666)) by [@cyningsun](https://github.com/cyningsun)
- **Digest Helper Functions**: Added `DigestString` and `DigestBytes` helper functions for client-side xxh3 hashing compatible with Redis DIGEST command ([#3679](https://github.com/redis/go-redis/pull/3679)) by [@ofekshenawa](https://github.com/ofekshenawa)
- **SMIGRATED New Format**: Updated SMIGRATED parser to support new format and remember original host:port ([#3697](https://github.com/redis/go-redis/pull/3697)) by [@ndyakov](https://github.com/ndyakov)
- **Cluster State Reload Interval**: Added cluster state reload interval option for maintenance notifications ([#3663](https://github.com/redis/go-redis/pull/3663)) by [@ndyakov](https://github.com/ndyakov)

## 🐛 Bug Fixes

- **PubSub nil pointer dereference**: Fixed nil pointer dereference in PubSub after `WithTimeout()` - `pubSubPool` is now properly cloned ([#3710](https://github.com/redis/go-redis/pull/3710)) by [@Copilot](https://github.com/apps/copilot-swe-agent)
- **MaintNotificationsConfig nil check**: Guard against nil `MaintNotificationsConfig` in `initConn` ([#3707](https://github.com/redis/go-redis/pull/3707)) by [@veeceey](https://github.com/veeceey)
- **wantConnQueue zombie elements**: Fixed zombie `wantConn` elements accumulation in `wantConnQueue` ([#3680](https://github.com/redis/go-redis/pull/3680)) by [@cyningsun](https://github.com/cyningsun)
- **XADD/XTRIM approx flag**: Fixed XADD and XTRIM to use `=` when approx is false ([#3684](https://github.com/redis/go-redis/pull/3684)) by [@ndyakov](https://github.com/ndyakov)
- **Sentinel timeout retry**: When connection to a sentinel times out, attempt to connect to other sentinels ([#3654](https://github.com/redis/go-redis/pull/3654)) by [@cxljs](https://github.com/cxljs)

## ⚡ Performance

- **Fuzz test optimization**: Eliminated repeated string conversions, used functional approach for cleaner operation selection ([#3692](https://github.com/redis/go-redis/pull/3692)) by [@feiguoL](https://github.com/feiguoL)
- **Pre-allocate capacity**: Pre-allocate slice capacity to prevent multiple capacity expansions ([#3689](https://github.com/redis/go-redis/pull/3689)) by [@feelshu](https://github.com/feelshu)

## 🧪 Testing

- **Comprehensive TLS tests**: Added comprehensive TLS tests and example for standalone, cluster, and certificate authentication ([#3681](https://github.com/redis/go-redis/pull/3681)) by [@ndyakov](https://github.com/ndyakov)
- **Redis 8.6**: Updated CI to use Redis 8.6-pre ([#3685](https://github.com/redis/go-redis/pull/3685)) by [@ndyakov](https://github.com/ndyakov)

## 🧰 Maintenance

- **Deprecation warnings**: Added deprecation warnings for commands based on Redis documentation ([#3673](https://github.com/redis/go-redis/pull/3673)) by [@ndyakov](https://github.com/ndyakov)
- **Use errors.Join()**: Replaced custom error join function with standard library `errors.Join()` ([#3653](https://github.com/redis/go-redis/pull/3653)) by [@cxljs](https://github.com/cxljs)
- **Use Go 1.21 min/max**: Use Go 1.21's built-in min/max functions ([#3656](https://github.com/redis/go-redis/pull/3656)) by [@cxljs](https://github.com/cxljs)
- **Proper formatting**: Code formatting improvements ([#3670](https://github.com/redis/go-redis/pull/3670)) by [@12ya](https://github.com/12ya)
- **Set commands documentation**: Added comprehensive documentation to all set command methods ([#3642](https://github.com/redis/go-redis/pull/3642)) by [@iamamirsalehi](https://github.com/iamamirsalehi)
- **MaxActiveConns docs**: Added default value documentation for `MaxActiveConns` ([#3674](https://github.com/redis/go-redis/pull/3674)) by [@codykaup](https://github.com/codykaup)
- **README example update**: Updated README example ([#3657](https://github.com/redis/go-redis/pull/3657)) by [@cxljs](https://github.com/cxljs)
- **Cluster maintnotif example**: Added example application for cluster maintenance notifications ([#3651](https://github.com/redis/go-redis/pull/3651)) by [@ndyakov](https://github.com/ndyakov)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@12ya](https://github.com/12ya), [@Copilot](https://github.com/apps/copilot-swe-agent), [@codykaup](https://github.com/codykaup), [@cxljs](https://github.com/cxljs), [@cyningsun](https://github.com/cyningsun), [@feelshu](https://github.com/feelshu), [@feiguoL](https://github.com/feiguoL), [@iamamirsalehi](https://github.com/iamamirsalehi), [@naveenchander30](https://github.com/naveenchander30), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@veeceey](https://github.com/veeceey)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.17.0...v9.18.0

# 9.18.0-beta.2 (2025-12-09)

## 🚀 Highlights

### Go Version Update

This release updates the minimum required Go version to 1.21. This is part of a gradual migration strategy where the minimum supported Go version will be three versions behind the latest release. With each new Go version release, we will bump the minimum version by one, ensuring compatibility while staying current with the Go ecosystem.

### Stability Improvements

This release includes several important stability fixes:
- Fixed a critical panic in the handoff worker manager that could occur when handling nil errors
- Improved test reliability for Smart Client Handoff functionality
- Fixed logging format issues that could cause runtime errors

## ✨ New Features

- OpenTelemetry metrics improvements for nil response handling ([#3638](https://github.com/redis/go-redis/pull/3638)) by [@fengve](https://github.com/fengve)

## 🐛 Bug Fixes

- Fixed panic on nil error in handoffWorkerManager closeConnFromRequest ([#3633](https://github.com/redis/go-redis/pull/3633)) by [@ccoVeille](https://github.com/ccoVeille)
- Fixed bad sprintf syntax in logging ([#3632](https://github.com/redis/go-redis/pull/3632)) by [@ccoVeille](https://github.com/ccoVeille)

## 🧰 Maintenance

- Updated minimum Go version to 1.21 ([#3640](https://github.com/redis/go-redis/pull/3640)) by [@ndyakov](https://github.com/ndyakov)
- Use Go 1.20 idiomatic string<->byte conversion ([#3435](https://github.com/redis/go-redis/pull/3435)) by [@justinhwang](https://github.com/justinhwang)
- Reduce flakiness of Smart Client Handoff test ([#3641](https://github.com/redis/go-redis/pull/3641)) by [@kiryazovi-redis](https://github.com/kiryazovi-redis)
- Revert PR #3634 (Observability metrics phase1) ([#3635](https://github.com/redis/go-redis/pull/3635)) by [@ofekshenawa](https://github.com/ofekshenawa)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@justinhwang](https://github.com/justinhwang), [@ndyakov](https://github.com/ndyakov), [@kiryazovi-redis](https://github.com/kiryazovi-redis), [@fengve](https://github.com/fengve), [@ccoVeille](https://github.com/ccoVeille), [@ofekshenawa](https://github.com/ofekshenawa)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.18.0-beta.1...v9.18.0-beta.2

# 9.18.0-beta.1 (2025-12-01)

## 🚀 Highlights

### Request and Response Policy Based Routing in Cluster Mode

This beta release introduces comprehensive support for Redis COMMAND-based request and response policy routing for cluster clients. This feature enables intelligent command routing and response aggregation based on Redis command metadata.

**Key Features:**
- **Command Policy Loader**: Automatically parses and caches COMMAND metadata with routing/aggregation hints
- **Enhanced Routing Engine**: Supports all request policies including:
  - `default(keyless)` - Commands without keys
  - `default(hashslot)` - Commands with hash slot routing
  - `all_shards` - Commands that need to run on all shards
  - `all_nodes` - Commands that need to run on all nodes
  - `multi_shard` - Commands that span multiple shards
  - `special` - Commands with custom routing logic
- **Response Aggregator**: Intelligently combines multi-shard replies based on response policies:
  - `all_succeeded` - All shards must succeed
  - `one_succeeded` - At least one shard must succeed
  - `agg_sum` - Aggregate numeric responses
  - `special` - Custom aggregation logic (e.g., FT.CURSOR)
- **Raw Command Support**: Policies are enforced on `Client.Do(ctx, args...)`

This feature is particularly useful for Redis Stack commands like RediSearch that need to operate across multiple shards in a cluster.

### Connection Pool Improvements

Fixed a critical defect in the connection pool's turn management mechanism that could lead to connection leaks under certain conditions. The fix ensures proper 1:1 correspondence between turns and connections.

## ✨ New Features

- Request and Response Policy Based Routing in Cluster Mode ([#3422](https://github.com/redis/go-redis/pull/3422)) by [@ofekshenawa](https://github.com/ofekshenawa)

## 🐛 Bug Fixes

- Fixed connection pool turn management to prevent connection leaks ([#3626](https://github.com/redis/go-redis/pull/3626)) by [@cyningsun](https://github.com/cyningsun)

## 🧰 Maintenance

- chore(deps): bump rojopolis/spellcheck-github-actions from 0.54.0 to 0.55.0 ([#3627](https://github.com/redis/go-redis/pull/3627))

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@cyningsun](https://github.com/cyningsun), [@ofekshenawa](https://github.com/ofekshenawa), [@ndyakov](https://github.com/ndyakov)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.17.1...v9.18.0-beta.1

# 9.17.1 (2025-11-25)

## 🐛 Bug Fixes

- add wait to keyless commands list ([#3615](https://github.com/redis/go-redis/pull/3615)) by [@marcoferrer](https://github.com/marcoferrer)
- fix(time): remove cached time optimization ([#3611](https://github.com/redis/go-redis/pull/3611)) by [@ndyakov](https://github.com/ndyakov)

## 🧰 Maintenance

- chore(deps): bump golangci/golangci-lint-action from 9.0.0 to 9.1.0 ([#3609](https://github.com/redis/go-redis/pull/3609))
- chore(deps): bump actions/checkout from 5 to 6 ([#3610](https://github.com/redis/go-redis/pull/3610))
- chore(script): fix help call in tag.sh ([#3606](https://github.com/redis/go-redis/pull/3606)) by [@ndyakov](https://github.com/ndyakov)

## Contributors
We'd like to thank all the contributors who worked on this release!

[@marcoferrer](https://github.com/marcoferrer) and [@ndyakov](https://github.com/ndyakov)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.17.0...v9.17.1

# 9.17.0 (2025-11-19)

## 🚀 Highlights

### Redis 8.4 Support
Added support for Redis 8.4, including new commands and features ([#3572](https://github.com/redis/go-redis/pull/3572))

### Typed Errors
Introduced typed errors for better error handling using `errors.As` instead of string checks. Errors can now be wrapped and set to commands in hooks without breaking library functionality ([#3602](https://github.com/redis/go-redis/pull/3602))

### New Commands
- **CAS/CAD Commands**: Added support for Compare-And-Set/Compare-And-Delete operations with conditional matching (`IFEQ`, `IFNE`, `IFDEQ`, `IFDNE`) ([#3583](https://github.com/redis/go-redis/pull/3583), [#3595](https://github.com/redis/go-redis/pull/3595))
- **MSETEX**: Atomically set multiple key-value pairs with expiration options and conditional modes ([#3580](https://github.com/redis/go-redis/pull/3580))
- **XReadGroup CLAIM**: Consume both incoming and idle pending entries from streams in a single call ([#3578](https://github.com/redis/go-redis/pull/3578))
- **ACL Commands**: Added `ACLGenPass`, `ACLUsers`, and `ACLWhoAmI` ([#3576](https://github.com/redis/go-redis/pull/3576))
- **SLOWLOG Commands**: Added `SLOWLOG LEN` and `SLOWLOG RESET` ([#3585](https://github.com/redis/go-redis/pull/3585))
- **LATENCY Commands**: Added `LATENCY LATEST` and `LATENCY RESET` ([#3584](https://github.com/redis/go-redis/pull/3584))

### Search & Vector Improvements
- **Hybrid Search**: Added  **EXPERIMENTAL** support for the new `FT.HYBRID` command ([#3573](https://github.com/redis/go-redis/pull/3573))
- **Vector Range**: Added `VRANGE` command for vector sets ([#3543](https://github.com/redis/go-redis/pull/3543))
- **FT.INFO Enhancements**: Added vector-specific attributes in FT.INFO response ([#3596](https://github.com/redis/go-redis/pull/3596))

### Connection Pool Improvements
- **Improved Connection Success Rate**: Implemented FIFO queue-based fairness and context pattern for connection creation to prevent premature cancellation under high concurrency ([#3518](https://github.com/redis/go-redis/pull/3518))
- **Connection State Machine**: Resolved race conditions and improved pool performance with proper state tracking ([#3559](https://github.com/redis/go-redis/pull/3559))
- **Pool Performance**: Significant performance improvements with faster semaphores, lockless hook manager, and reduced allocations (47-67% faster Get/Put operations) ([#3565](https://github.com/redis/go-redis/pull/3565))

### Metrics & Observability
- **Canceled Metric Attribute**: Added 'canceled' metrics attribute to distinguish context cancellation errors from other errors ([#3566](https://github.com/redis/go-redis/pull/3566))

## ✨ New Features

- Typed errors with wrapping support ([#3602](https://github.com/redis/go-redis/pull/3602)) by [@ndyakov](https://github.com/ndyakov)
- CAS/CAD commands (marked as experimental) ([#3583](https://github.com/redis/go-redis/pull/3583), [#3595](https://github.com/redis/go-redis/pull/3595)) by [@ndyakov](https://github.com/ndyakov), [@htemelski-redis](https://github.com/htemelski-redis)
- MSETEX command support ([#3580](https://github.com/redis/go-redis/pull/3580)) by [@ofekshenawa](https://github.com/ofekshenawa)
- XReadGroup CLAIM argument ([#3578](https://github.com/redis/go-redis/pull/3578)) by [@ofekshenawa](https://github.com/ofekshenawa)
- ACL commands: GenPass, Users, WhoAmI ([#3576](https://github.com/redis/go-redis/pull/3576)) by [@destinyoooo](https://github.com/destinyoooo)
- SLOWLOG commands: LEN, RESET ([#3585](https://github.com/redis/go-redis/pull/3585)) by [@destinyoooo](https://github.com/destinyoooo)
- LATENCY commands: LATEST, RESET ([#3584](https://github.com/redis/go-redis/pull/3584)) by [@destinyoooo](https://github.com/destinyoooo)
- Hybrid search command (FT.HYBRID) ([#3573](https://github.com/redis/go-redis/pull/3573)) by [@htemelski-redis](https://github.com/htemelski-redis)
- Vector range command (VRANGE) ([#3543](https://github.com/redis/go-redis/pull/3543)) by [@cxljs](https://github.com/cxljs)
- Vector-specific attributes in FT.INFO ([#3596](https://github.com/redis/go-redis/pull/3596)) by [@ndyakov](https://github.com/ndyakov)
- Improved connection pool success rate with FIFO queue ([#3518](https://github.com/redis/go-redis/pull/3518)) by [@cyningsun](https://github.com/cyningsun)
- Canceled metrics attribute for context errors ([#3566](https://github.com/redis/go-redis/pull/3566)) by [@pvragov](https://github.com/pvragov)

## 🐛 Bug Fixes

- Fixed Failover Client MaintNotificationsConfig ([#3600](https://github.com/redis/go-redis/pull/3600)) by [@ajax16384](https://github.com/ajax16384)
- Fixed ACLGenPass function to use the bit parameter ([#3597](https://github.com/redis/go-redis/pull/3597)) by [@destinyoooo](https://github.com/destinyoooo)
- Return error instead of panic from commands ([#3568](https://github.com/redis/go-redis/pull/3568)) by [@dragneelfps](https://github.com/dragneelfps)
- Safety harness in `joinErrors` to prevent panic ([#3577](https://github.com/redis/go-redis/pull/3577)) by [@manisharma](https://github.com/manisharma)

## ⚡ Performance

- Connection state machine with race condition fixes ([#3559](https://github.com/redis/go-redis/pull/3559)) by [@ndyakov](https://github.com/ndyakov)
- Pool performance improvements: 47-67% faster Get/Put, 33% less memory, 50% fewer allocations ([#3565](https://github.com/redis/go-redis/pull/3565)) by [@ndyakov](https://github.com/ndyakov)

## 🧪 Testing & Infrastructure

- Updated to Redis 8.4.0 image ([#3603](https://github.com/redis/go-redis/pull/3603)) by [@ndyakov](https://github.com/ndyakov)
- Added Redis 8.4-RC1-pre to CI ([#3572](https://github.com/redis/go-redis/pull/3572)) by [@ndyakov](https://github.com/ndyakov)
- Refactored tests for idiomatic Go ([#3561](https://github.com/redis/go-redis/pull/3561), [#3562](https://github.com/redis/go-redis/pull/3562), [#3563](https://github.com/redis/go-redis/pull/3563)) by [@12ya](https://github.com/12ya)

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@12ya](https://github.com/12ya), [@ajax16384](https://github.com/ajax16384), [@cxljs](https://github.com/cxljs), [@cyningsun](https://github.com/cyningsun), [@destinyoooo](https://github.com/destinyoooo), [@dragneelfps](https://github.com/dragneelfps), [@htemelski-redis](https://github.com/htemelski-redis), [@manisharma](https://github.com/manisharma), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@pvragov](https://github.com/pvragov)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.16.0...v9.17.0

# 9.16.0 (2025-10-23)

## 🚀 Highlights

### Maintenance Notifications Support

This release introduces comprehensive support for Redis maintenance notifications, enabling applications to handle server maintenance events gracefully. The new `maintnotifications` package provides:

- **RESP3 Push Notifications**: Full support for Redis RESP3 protocol push notifications
- **Connection Handoff**: Automatic connection migration during server maintenance with configurable retry policies and circuit breakers
- **Graceful Degradation**: Configurable timeout relaxation during maintenance windows to prevent false failures
- **Event-Driven Architecture**: Background workers with on-demand scaling for efficient handoff processing
- **Production-Ready**: Comprehensive E2E testing framework and monitoring capabilities

For detailed usage examples and configuration options, see the [maintenance notifications documentation](maintnotifications/README.md).

## ✨ New Features

- **Trace Filtering**: Add support for filtering traces for specific commands, including pipeline operations and dial operations ([#3519](https://github.com/redis/go-redis/pull/3519), [#3550](https://github.com/redis/go-redis/pull/3550))
  - New `TraceCmdFilter` option to selectively trace commands
  - Reduces overhead by excluding high-frequency or low-value commands from traces

## 🐛 Bug Fixes

- **Pipeline Error Handling**: Fix issue where pipeline repeatedly sets the same error ([#3525](https://github.com/redis/go-redis/pull/3525))
- **Connection Pool**: Ensure re-authentication does not interfere with connection handoff operations ([#3547](https://github.com/redis/go-redis/pull/3547))

## 🔧 Improvements

- **Hash Commands**: Update hash command implementations ([#3523](https://github.com/redis/go-redis/pull/3523))
- **OpenTelemetry**: Use `metric.WithAttributeSet` to avoid unnecessary attribute copying in redisotel ([#3552](https://github.com/redis/go-redis/pull/3552))

## 📚 Documentation

- **Cluster Client**: Add explanation for why `MaxRetries` is disabled for `ClusterClient` ([#3551](https://github.com/redis/go-redis/pull/3551))

## 🧪 Testing & Infrastructure

- **E2E Testing**: Upgrade E2E testing framework with improved reliability and coverage ([#3541](https://github.com/redis/go-redis/pull/3541))
- **Release Process**: Improved resiliency of the release process ([#3530](https://github.com/redis/go-redis/pull/3530))

## 📦 Dependencies

- Bump `rojopolis/spellcheck-github-actions` from 0.51.0 to 0.52.0 ([#3520](https://github.com/redis/go-redis/pull/3520))
- Bump `github/codeql-action` from 3 to 4 ([#3544](https://github.com/redis/go-redis/pull/3544))

## 👥 Contributors

We'd like to thank all the contributors who worked on this release!

[@ndyakov](https://github.com/ndyakov), [@htemelski-redis](https://github.com/htemelski-redis), [@Sovietaced](https://github.com/Sovietaced), [@Udhayarajan](https://github.com/Udhayarajan), [@boekkooi-impossiblecloud](https://github.com/boekkooi-impossiblecloud), [@Pika-Gopher](https://github.com/Pika-Gopher), [@cxljs](https://github.com/cxljs), [@huiyifyj](https://github.com/huiyifyj), [@omid-h70](https://github.com/omid-h70)

---

**Full Changelog**: https://github.com/redis/go-redis/compare/v9.14.0...v9.16.0


# 9.15.0 was accidentally released. Please use version 9.16.0 instead.

# 9.15.0-beta.3 (2025-09-26)

## Highlights
This beta release includes a pre-production version of processing push notifications and hitless upgrades.

# Changes

- chore: Update hash_commands.go ([#3523](https://github.com/redis/go-redis/pull/3523))

## 🚀 New Features

- feat: RESP3 notifications support & Hitless notifications handling ([#3418](https://github.com/redis/go-redis/pull/3418))

## 🐛 Bug Fixes

- fix: pipeline repeatedly sets the error ([#3525](https://github.com/redis/go-redis/pull/3525))

## 🧰 Maintenance

- chore(deps): bump rojopolis/spellcheck-github-actions from 0.51.0 to 0.52.0 ([#3520](https://github.com/redis/go-redis/pull/3520))
- feat(e2e-testing): maintnotifications e2e and refactor ([#3526](https://github.com/redis/go-redis/pull/3526))
- feat(tag.sh): Improved resiliency of the release process ([#3530](https://github.com/redis/go-redis/pull/3530))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@cxljs](https://github.com/cxljs), [@ndyakov](https://github.com/ndyakov), [@htemelski-redis](https://github.com/htemelski-redis), and [@omid-h70](https://github.com/omid-h70)


# 9.15.0-beta.1 (2025-09-10)

## Highlights
This beta release includes a pre-production version of processing push notifications and hitless upgrades.

### Hitless Upgrades
Hitless upgrades is a major new feature that allows for zero-downtime upgrades in Redis clusters.
You can find more information in the [Hitless Upgrades documentation](https://github.com/redis/go-redis/tree/master/hitless).

# Changes

## 🚀 New Features
- [CAE-1088] & [CAE-1072] feat: RESP3 notifications support & Hitless notifications handling ([#3418](https://github.com/redis/go-redis/pull/3418))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@ndyakov](https://github.com/ndyakov), [@htemelski-redis](https://github.com/htemelski-redis), [@ofekshenawa](https://github.com/ofekshenawa)


# 9.14.0 (2025-09-10)

## Highlights
- Added batch process method to the pipeline ([#3510](https://github.com/redis/go-redis/pull/3510))

# Changes

## 🚀 New Features

- Added batch process method to the pipeline ([#3510](https://github.com/redis/go-redis/pull/3510))

## 🐛 Bug Fixes

- fix: SetErr on Cmd if the command cannot be queued correctly in multi/exec ([#3509](https://github.com/redis/go-redis/pull/3509))

## 🧰 Maintenance

- Updates release drafter config to exclude dependabot ([#3511](https://github.com/redis/go-redis/pull/3511))
- chore(deps): bump actions/setup-go from 5 to 6 ([#3504](https://github.com/redis/go-redis/pull/3504))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@elena-kolevska](https://github.com/elena-kolevksa), [@htemelski-redis](https://github.com/htemelski-redis) and [@ndyakov](https://github.com/ndyakov)


# 9.13.0 (2025-09-03)

## Highlights
- Pipeliner expose queued commands ([#3496](https://github.com/redis/go-redis/pull/3496))
- Ensure that JSON.GET returns Nil response ([#3470](https://github.com/redis/go-redis/pull/3470))
- Fixes on Read and Write buffer sizes and UniversalOptions

## Changes
- Pipeliner expose queued commands ([#3496](https://github.com/redis/go-redis/pull/3496))
- fix(test): fix a timing issue in pubsub test ([#3498](https://github.com/redis/go-redis/pull/3498))
- Allow users to enable read-write splitting in failover mode. ([#3482](https://github.com/redis/go-redis/pull/3482))
- Set the read/write buffer size of the sentinel client to 4KiB ([#3476](https://github.com/redis/go-redis/pull/3476))

## 🚀 New Features

- fix(otel): register wait metrics ([#3499](https://github.com/redis/go-redis/pull/3499))
- Support subscriptions against cluster slave nodes ([#3480](https://github.com/redis/go-redis/pull/3480))
- Add wait metrics to otel ([#3493](https://github.com/redis/go-redis/pull/3493))
- Clean failing timeout implementation ([#3472](https://github.com/redis/go-redis/pull/3472))

## 🐛 Bug Fixes

- Do not assume that all non-IP hosts are loopbacks ([#3085](https://github.com/redis/go-redis/pull/3085))
- Ensure that JSON.GET returns Nil response ([#3470](https://github.com/redis/go-redis/pull/3470))

## 🧰 Maintenance

- fix(otel): register wait metrics ([#3499](https://github.com/redis/go-redis/pull/3499))
- fix(make test): Add default env in makefile ([#3491](https://github.com/redis/go-redis/pull/3491))
- Update the introduction to running tests in README.md ([#3495](https://github.com/redis/go-redis/pull/3495))
- test: Add comprehensive edge case tests for IncrByFloat command ([#3477](https://github.com/redis/go-redis/pull/3477))
- Set the default read/write buffer size of Redis connection to 32KiB ([#3483](https://github.com/redis/go-redis/pull/3483))
- Bumps test image to 8.2.1-pre ([#3478](https://github.com/redis/go-redis/pull/3478))
- fix UniversalOptions miss ReadBufferSize and WriteBufferSize options ([#3485](https://github.com/redis/go-redis/pull/3485))
- chore(deps): bump actions/checkout from 4 to 5 ([#3484](https://github.com/redis/go-redis/pull/3484))
- Removes dry run for stale issues policy ([#3471](https://github.com/redis/go-redis/pull/3471))
- Update otel metrics URL ([#3474](https://github.com/redis/go-redis/pull/3474))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@LINKIWI](https://github.com/LINKIWI), [@cxljs](https://github.com/cxljs), [@cybersmeashish](https://github.com/cybersmeashish), [@elena-kolevska](https://github.com/elena-kolevska), [@htemelski-redis](https://github.com/htemelski-redis), [@mwhooker](https://github.com/mwhooker), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@suever](https://github.com/suever)


# 9.12.1 (2025-08-11)
## 🚀 Highlights
In the last version (9.12.0) the client introduced bigger write and read buffer sized. The default value we set was 512KiB.
However, users reported that this is too big for most use cases and can lead to high memory usage.
In this version the default value is changed to 256KiB. The `README.md` was updated to reflect the
correct default value and include a note that the default value can be changed.

## 🐛 Bug Fixes

- fix(options): Add buffer sizes to failover. Update README ([#3468](https://github.com/redis/go-redis/pull/3468))

## 🧰 Maintenance

- fix(options): Add buffer sizes to failover. Update README ([#3468](https://github.com/redis/go-redis/pull/3468))
- chore: update & fix otel example ([#3466](https://github.com/redis/go-redis/pull/3466))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@ndyakov](https://github.com/ndyakov) and [@vmihailenco](https://github.com/vmihailenco)

# 9.12.0 (2025-08-05)

## 🚀 Highlights

- This release includes support for [Redis 8.2](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/release-notes/redisce/redisos-8.2-release-notes/).
- Introduces an experimental Query Builders for `FTSearch`, `FTAggregate` and other search commands.
- Adds support for `EPSILON` option in `FT.VSIM`.
- Includes bug fixes and improvements contributed by the community related to ring and [redisotel](https://github.com/redis/go-redis/tree/master/extra/redisotel).

## Changes
- Improve stale issue workflow ([#3458](https://github.com/redis/go-redis/pull/3458))
- chore(ci): Add 8.2 rc2 pre build for CI ([#3459](https://github.com/redis/go-redis/pull/3459))
- Added new stream commands ([#3450](https://github.com/redis/go-redis/pull/3450))
- feat: Add "skip_verify" to Sentinel ([#3428](https://github.com/redis/go-redis/pull/3428))
- fix: `errors.Join` requires Go 1.20 or later ([#3442](https://github.com/redis/go-redis/pull/3442))
- DOC-4344 document quickstart examples ([#3426](https://github.com/redis/go-redis/pull/3426))
- feat(bitop): add support for the new bitop operations ([#3409](https://github.com/redis/go-redis/pull/3409))

## 🚀 New Features

- feat: recover addIdleConn may occur panic ([#2445](https://github.com/redis/go-redis/pull/2445))
- feat(ring): specify custom health check func via HeartbeatFn option ([#2940](https://github.com/redis/go-redis/pull/2940))
- Add Query Builder for RediSearch commands ([#3436](https://github.com/redis/go-redis/pull/3436))
- add configurable buffer sizes for Redis connections ([#3453](https://github.com/redis/go-redis/pull/3453))
- Add VAMANA vector type to RediSearch ([#3449](https://github.com/redis/go-redis/pull/3449))
- VSIM add `EPSILON` option ([#3454](https://github.com/redis/go-redis/pull/3454))
- Add closing support to otel metrics instrumentation ([#3444](https://github.com/redis/go-redis/pull/3444))

## 🐛 Bug Fixes

- fix(redisotel): fix buggy append in reportPoolStats ([#3122](https://github.com/redis/go-redis/pull/3122))
- fix(search): return results even if doc is empty ([#3457](https://github.com/redis/go-redis/pull/3457))
- [ISSUE-3402]: Ring.Pipelined return dial timeout error ([#3403](https://github.com/redis/go-redis/pull/3403))

## 🧰 Maintenance

- Merges stale issues jobs into one job with two steps ([#3463](https://github.com/redis/go-redis/pull/3463))
- improve code readability ([#3446](https://github.com/redis/go-redis/pull/3446))
- chore(release): 9.12.0-beta.1 ([#3460](https://github.com/redis/go-redis/pull/3460))
- DOC-5472 time series doc examples ([#3443](https://github.com/redis/go-redis/pull/3443))
- Add VAMANA compression algorithm tests ([#3461](https://github.com/redis/go-redis/pull/3461))
- bumped redis 8.2 version used in the CI/CD ([#3451](https://github.com/redis/go-redis/pull/3451))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@andy-stark-redis](https://github.com/andy-stark-redis), [@cxljs](https://github.com/cxljs), [@elena-kolevska](https://github.com/elena-kolevska), [@htemelski-redis](https://github.com/htemelski-redis), [@jouir](https://github.com/jouir), [@monkey92t](https://github.com/monkey92t), [@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@rokn](https://github.com/rokn), [@smnvdev](https://github.com/smnvdev), [@strobil](https://github.com/strobil) and [@wzy9607](https://github.com/wzy9607)

## New Contributors
* [@htemelski-redis](https://github.com/htemelski-redis) made their first contribution in [#3409](https://github.com/redis/go-redis/pull/3409)
* [@smnvdev](https://github.com/smnvdev) made their first contribution in [#3403](https://github.com/redis/go-redis/pull/3403)
* [@rokn](https://github.com/rokn) made their first contribution in [#3444](https://github.com/redis/go-redis/pull/3444)

# 9.11.0 (2025-06-24)

## 🚀 Highlights

Fixes TxPipeline to work correctly in cluster scenarios, allowing execution of commands
only in the same slot.

# Changes

## 🚀 New Features

- Set cluster slot for `scan` commands, rather than random ([#2623](https://github.com/redis/go-redis/pull/2623))
- Add CredentialsProvider field to UniversalOptions ([#2927](https://github.com/redis/go-redis/pull/2927))
- feat(redisotel): add WithCallerEnabled option ([#3415](https://github.com/redis/go-redis/pull/3415))

## 🐛 Bug Fixes

- fix(txpipeline): keyless commands should take the slot of the keyed ([#3411](https://github.com/redis/go-redis/pull/3411))
- fix(loading): cache the loaded flag for slave nodes ([#3410](https://github.com/redis/go-redis/pull/3410))
- fix(txpipeline): should return error on multi/exec on multiple slots ([#3408](https://github.com/redis/go-redis/pull/3408))
- fix: check if the shard exists to avoid returning nil ([#3396](https://github.com/redis/go-redis/pull/3396))

## 🧰 Maintenance

- feat: optimize connection pool waitTurn ([#3412](https://github.com/redis/go-redis/pull/3412))
- chore(ci): update CI redis builds ([#3407](https://github.com/redis/go-redis/pull/3407))
- chore: remove a redundant method from `Ring`, `Client` and `ClusterClient` ([#3401](https://github.com/redis/go-redis/pull/3401))
- test: refactor TestBasicCredentials using table-driven tests ([#3406](https://github.com/redis/go-redis/pull/3406))
- perf: reduce unnecessary memory allocation operations ([#3399](https://github.com/redis/go-redis/pull/3399))
- fix: insert entry during iterating over a map ([#3398](https://github.com/redis/go-redis/pull/3398))
- DOC-5229 probabilistic data type examples ([#3413](https://github.com/redis/go-redis/pull/3413))
- chore(deps): bump rojopolis/spellcheck-github-actions from 0.49.0 to 0.51.0 ([#3414](https://github.com/redis/go-redis/pull/3414))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@andy-stark-redis](https://github.com/andy-stark-redis), [@boekkooi-impossiblecloud](https://github.com/boekkooi-impossiblecloud), [@cxljs](https://github.com/cxljs), [@dcherubini](https://github.com/dcherubini), [@dependabot[bot]](https://github.com/apps/dependabot), [@iamamirsalehi](https://github.com/iamamirsalehi), [@ndyakov](https://github.com/ndyakov), [@pete-woods](https://github.com/pete-woods), [@twz915](https://github.com/twz915) and [dependabot[bot]](https://github.com/apps/dependabot)

# 9.10.0 (2025-06-06)

## 🚀 Highlights

`go-redis` now supports [vector sets](https://redis.io/docs/latest/develop/data-types/vector-sets/). This data type is marked
as "in preview" in Redis and its support in `go-redis` is marked as experimental. You can find examples in the documentation and
in the `doctests` folder.

# Changes

## 🚀 New Features

- feat: support vectorset ([#3375](https://github.com/redis/go-redis/pull/3375))

## 🧰 Maintenance

- Add the missing NewFloatSliceResult for testing ([#3393](https://github.com/redis/go-redis/pull/3393))
- DOC-5078 vector set examples ([#3394](https://github.com/redis/go-redis/pull/3394))

## Contributors
We'd like to thank all the contributors who worked on this release!

[@AndBobsYourUncle](https://github.com/AndBobsYourUncle), [@andy-stark-redis](https://github.com/andy-stark-redis), [@fukua95](https://github.com/fukua95) and [@ndyakov](https://github.com/ndyakov)



# 9.9.0 (2025-05-27)

## 🚀 Highlights
- **Token-based Authentication**: Added `StreamingCredentialsProvider` for dynamic credential updates (experimental)
  - Can be used with [go-redis-entraid](https://github.com/redis/go-redis-entraid) for Azure AD authentication
- **Connection Statistics**: Added connection waiting statistics for better monitoring
- **Failover Improvements**: Added `ParseFailoverURL` for easier failover configuration
- **Ring Client Enhancements**: Added shard access methods for better Pub/Sub management

## ✨ New Features
- Added `StreamingCredentialsProvider` for token-based authentication ([#3320](https://github.com/redis/go-redis/pull/3320))
  - Supports dynamic credential updates
  - Includes connection close hooks
  - Note: Currently marked as experimental
- Added `ParseFailoverURL` for parsing failover URLs ([#3362](https://github.com/redis/go-redis/pull/3362))
- Added connection waiting statistics ([#2804](https://github.com/redis/go-redis/pull/2804))
- Added new utility functions:
  - `ParseFloat` and `MustParseFloat` in public utils package ([#3371](https://github.com/redis/go-redis/pull/3371))
  - Unit tests for `Atoi`, `ParseInt`, `ParseUint`, and `ParseFloat` ([#3377](https://github.com/redis/go-redis/pull/3377))
- Added Ring client shard access methods:
  - `GetShardClients()` to retrieve all active shard clients
  - `GetShardClientForKey(key string)` to get the shard client for a specific key ([#3388](https://github.com/redis/go-redis/pull/3388))

## 🐛 Bug Fixes
- Fixed routing reads to loading slave nodes ([#3370](https://github.com/redis/go-redis/pull/3370))
- Added support for nil lag in XINFO GROUPS ([#3369](https://github.com/redis/go-redis/pull/3369))
- Fixed pool acquisition timeout issues ([#3381](https://github.com/redis/go-redis/pull/3381))
- Optimized unnecessary copy operations ([#3376](https://github.com/redis/go-redis/pull/3376))

## 📚 Documentation
- Updated documentation for XINFO GROUPS with nil lag support ([#3369](https://github.com/redis/go-redis/pull/3369))
- Added package-level comments for new features

## ⚡ Performance and Reliability
- Optimized `ReplaceSpaces` function ([#3383](https://github.com/redis/go-redis/pull/3383))
- Set default value for `Options.Protocol` in `init()` ([#3387](https://github.com/redis/go-redis/pull/3387))
- Exported pool errors for public consumption ([#3380](https://github.com/redis/go-redis/pull/3380))

## 🔧 Dependencies and Infrastructure
- Updated Redis CI to version 8.0.1 ([#3372](https://github.com/redis/go-redis/pull/3372))
- Updated spellcheck GitHub Actions ([#3389](https://github.com/redis/go-redis/pull/3389))
- Removed unused parameters ([#3382](https://github.com/redis/go-redis/pull/3382), [#3384](https://github.com/redis/go-redis/pull/3384))

## 🧪 Testing
- Added unit tests for pool acquisition timeout ([#3381](https://github.com/redis/go-redis/pull/3381))
- Added unit tests for utility functions ([#3377](https://github.com/redis/go-redis/pull/3377))

## 👥 Contributors

We would like to thank all the contributors who made this release possible:

[@ndyakov](https://github.com/ndyakov), [@ofekshenawa](https://github.com/ofekshenawa), [@LINKIWI](https://github.com/LINKIWI), [@iamamirsalehi](https://github.com/iamamirsalehi), [@fukua95](https://github.com/fukua95), [@lzakharov](https://github.com/lzakharov), [@DengY11](https://github.com/DengY11)

## 📝 Changelog

For a complete list of changes, see the [full changelog](https://github.com/redis/go-redis/compare/v9.8.0...v9.9.0).

# 9.8.0 (2025-04-30)

## 🚀 Highlights
- **Redis 8 Support**: Full compatibility with Redis 8.0, including testing and CI integration
- **Enhanced Hash Operations**: Added support for new hash commands (`HGETDEL`, `HGETEX`, `HSETEX`) and `HSTRLEN` command
- **Search Improvements**: Enabled Search DIALECT 2 by default and added `CountOnly` argument for `FT.Search`

## ✨ New Features
- Added support for new hash commands: `HGETDEL`, `HGETEX`, `HSETEX` ([#3305](https://github.com/redis/go-redis/pull/3305))
- Added `HSTRLEN` command for hash operations ([#2843](https://github.com/redis/go-redis/pull/2843))
- Added `Do` method for raw query by single connection from `pool.Conn()` ([#3182](https://github.com/redis/go-redis/pull/3182))
- Prevent false-positive marshaling by treating zero time.Time as empty in isEmptyValue ([#3273](https://github.com/redis/go-redis/pull/3273))
- Added FailoverClusterClient support for Universal client ([#2794](https://github.com/redis/go-redis/pull/2794))
- Added support for cluster mode with `IsClusterMode` config parameter ([#3255](https://github.com/redis/go-redis/pull/3255))
- Added client name support in `HELLO` RESP handshake ([#3294](https://github.com/redis/go-redis/pull/3294))
- **Enabled Search DIALECT 2 by default** ([#3213](https://github.com/redis/go-redis/pull/3213))
- Added read-only option for failover configurations ([#3281](https://github.com/redis/go-redis/pull/3281))
- Added `CountOnly` argument for `FT.Search` to use `LIMIT 0 0` ([#3338](https://github.com/redis/go-redis/pull/3338))
- Added `DB` option support in `NewFailoverClusterClient` ([#3342](https://github.com/redis/go-redis/pull/3342))
- Added `nil` check for the options when creating a client ([#3363](https://github.com/redis/go-redis/pull/3363))

## 🐛 Bug Fixes
- Fixed `PubSub` concurrency safety issues ([#3360](https://github.com/redis/go-redis/pull/3360))
- Fixed panic caused when argument is `nil` ([#3353](https://github.com/redis/go-redis/pull/3353))
- Improved error handling when fetching master node from sentinels ([#3349](https://github.com/redis/go-redis/pull/3349))
- Fixed connection pool timeout issues and increased retries ([#3298](https://github.com/redis/go-redis/pull/3298))
- Fixed context cancellation error leading to connection spikes on Primary instances ([#3190](https://github.com/redis/go-redis/pull/3190))
- Fixed RedisCluster client to consider `MASTERDOWN` a retriable error ([#3164](https://github.com/redis/go-redis/pull/3164))
- Fixed tracing to show complete commands instead of truncated versions ([#3290](https://github.com/redis/go-redis/pull/3290))
- Fixed OpenTelemetry instrumentation to prevent multiple span reporting ([#3168](https://github.com/redis/go-redis/pull/3168))
- Fixed `FT.Search` Limit argument and added `CountOnly` argument for limit 0 0 ([#3338](https://github.com/redis/go-redis/pull/3338))
- Fixed missing command in interface ([#3344](https://github.com/redis/go-redis/pull/3344))
- Fixed slot calculation for `COUNTKEYSINSLOT` command ([#3327](https://github.com/redis/go-redis/pull/3327))
- Updated PubSub implementation with correct context ([#3329](https://github.com/redis/go-redis/pull/3329))

## 📚 Documentation
- Added hash search examples ([#3357](https://github.com/redis/go-redis/pull/3357))
- Fixed documentation comments ([#3351](https://github.com/redis/go-redis/pull/3351))
- Added `CountOnly` search example ([#3345](https://github.com/redis/go-redis/pull/3345))
- Added examples for list commands: `LLEN`, `LPOP`, `LPUSH`, `LRANGE`, `RPOP`, `RPUSH` ([#3234](https://github.com/redis/go-redis/pull/3234))
- Added `SADD` and `SMEMBERS` command examples ([#3242](https://github.com/redis/go-redis/pull/3242))
- Updated `README.md` to use Redis Discord guild ([#3331](https://github.com/redis/go-redis/pull/3331))
- Updated `HExpire` command documentation ([#3355](https://github.com/redis/go-redis/pull/3355))
- Featured OpenTelemetry instrumentation more prominently ([#3316](https://github.com/redis/go-redis/pull/3316))
- Updated `README.md` with additional information ([#310ce55](https://github.com/redis/go-redis/commit/310ce55))

## ⚡ Performance and Reliability
- Bound connection pool background dials to configured dial timeout ([#3089](https://github.com/redis/go-redis/pull/3089))
- Ensured context isn't exhausted via concurrent query ([#3334](https://github.com/redis/go-redis/pull/3334))

## 🔧 Dependencies and Infrastructure
- Updated testing image to Redis 8.0-RC2 ([#3361](https://github.com/redis/go-redis/pull/3361))
- Enabled CI for Redis CE 8.0 ([#3274](https://github.com/redis/go-redis/pull/3274))
- Updated various dependencies:
  - Bumped golangci/golangci-lint-action from 6.5.0 to 7.0.0 ([#3354](https://github.com/redis/go-redis/pull/3354))
  - Bumped rojopolis/spellcheck-github-actions ([#3336](https://github.com/redis/go-redis/pull/3336))
  - Bumped golang.org/x/net in example/otel ([#3308](https://github.com/redis/go-redis/pull/3308))
- Migrated golangci-lint configuration to v2 format ([#3354](https://github.com/redis/go-redis/pull/3354))

## ⚠️ Breaking Changes
- **Enabled Search DIALECT 2 by default** ([#3213](https://github.com/redis/go-redis/pull/3213))
- Dropped RedisGears (Triggers and Functions) support ([#3321](https://github.com/redis/go-redis/pull/3321))
- Dropped FT.PROFILE command that was never enabled ([#3323](https://github.com/redis/go-redis/pull/3323))

## 🔒 Security
- Fixed network error handling on SETINFO (CVE-2025-29923) ([#3295](https://github.com/redis/go-redis/pull/3295))

## 🧪 Testing
- Added integration tests for Redis 8 behavior changes in Redis Search ([#3337](https://github.com/redis/go-redis/pull/3337))
- Added vector types INT8 and UINT8 tests ([#3299](https://github.com/redis/go-redis/pull/3299))
- Added test codes for search_commands.go ([#3285](https://github.com/redis/go-redis/pull/3285))
- Fixed example test sorting ([#3292](https://github.com/redis/go-redis/pull/3292))

## 👥 Contributors

We would like to thank all the contributors who made this release possible:

[@alexander-menshchikov](https://github.com/alexander-menshchikov), [@EXPEbdodla](https://github.com/EXPEbdodla), [@afti](https://github.com/afti), [@dmaier-redislabs](https://github.com/dmaier-redislabs), [@four_leaf_clover](https://github.com/four_leaf_clover), [@alohaglenn](https://github.com/alohaglenn), [@gh73962](https://github.com/gh73962), [@justinmir](https://github.com/justinmir), [@LINKIWI](https://github.com/LINKIWI), [@liushuangbill](https://github.com/liushuangbill), [@golang88](https://github.com/golang88), [@gnpaone](https://github.com/gnpaone), [@ndyakov](https://github.com/ndyakov), [@nikolaydubina](https://github.com/nikolaydubina), [@oleglacto](https://github.com/oleglacto), [@andy-stark-redis](https://github.com/andy-stark-redis), [@rodneyosodo](https://github.com/rodneyosodo), [@dependabot](https://github.com/dependabot), [@rfyiamcool](https://github.com/rfyiamcool), [@frankxjkuang](https://github.com/frankxjkuang), [@fukua95](https://github.com/fukua95), [@soleymani-milad](https://github.com/soleymani-milad), [@ofekshenawa](https://github.com/ofekshenawa), [@khasanovbi](https://github.com/khasanovbi)


# Old Changelog
## Unreleased

### Changed

* `go-redis` won't skip span creation if the parent spans is not recording. ([#2980](https://github.com/redis/go-redis/issues/2980))
  Users can use the OpenTelemetry sampler to control the sampling behavior.
  For instance, you can use the `ParentBased(NeverSample())` sampler from `go.opentelemetry.io/otel/sdk/trace` to keep
  a similar behavior (drop orphan spans) of `go-redis` as before.

## [9.0.5](https://github.com/redis/go-redis/compare/v9.0.4...v9.0.5) (2023-05-29)


### Features

* Add ACL LOG ([#2536](https://github.com/redis/go-redis/issues/2536)) ([31ba855](https://github.com/redis/go-redis/commit/31ba855ddebc38fbcc69a75d9d4fb769417cf602))
* add field protocol to setupClusterQueryParams ([#2600](https://github.com/redis/go-redis/issues/2600)) ([840c25c](https://github.com/redis/go-redis/commit/840c25cb6f320501886a82a5e75f47b491e46fbe))
* add protocol option ([#2598](https://github.com/redis/go-redis/issues/2598)) ([3917988](https://github.com/redis/go-redis/commit/391798880cfb915c4660f6c3ba63e0c1a459e2af))



## [9.0.4](https://github.com/redis/go-redis/compare/v9.0.3...v9.0.4) (2023-05-01)


### Bug Fixes

* reader float parser ([#2513](https://github.com/redis/go-redis/issues/2513)) ([46f2450](https://github.com/redis/go-redis/commit/46f245075e6e3a8bd8471f9ca67ea95fd675e241))


### Features

* add client info command ([#2483](https://github.com/redis/go-redis/issues/2483)) ([b8c7317](https://github.com/redis/go-redis/commit/b8c7317cc6af444603731f7017c602347c0ba61e))
* no longer verify HELLO error messages ([#2515](https://github.com/redis/go-redis/issues/2515)) ([7b4f217](https://github.com/redis/go-redis/commit/7b4f2179cb5dba3d3c6b0c6f10db52b837c912c8))
* read the structure to increase the judgment of the omitempty op… ([#2529](https://github.com/redis/go-redis/issues/2529)) ([37c057b](https://github.com/redis/go-redis/commit/37c057b8e597c5e8a0e372337f6a8ad27f6030af))



## [9.0.3](https://github.com/redis/go-redis/compare/v9.0.2...v9.0.3) (2023-04-02)

### New Features

- feat(scan): scan time.Time sets the default decoding (#2413)
- Add support for CLUSTER LINKS command (#2504)
- Add support for acl dryrun command (#2502)
- Add support for COMMAND GETKEYS & COMMAND GETKEYSANDFLAGS (#2500)
- Add support for LCS Command (#2480)
- Add support for BZMPOP (#2456)
- Adding support for ZMPOP command (#2408)
- Add support for LMPOP (#2440)
- feat: remove pool unused fields (#2438)
- Expiretime and PExpireTime (#2426)
- Implement `FUNCTION` group of commands (#2475)
- feat(zadd): add ZAddLT and ZAddGT (#2429)
- Add: Support for COMMAND LIST command (#2491)
- Add support for BLMPOP (#2442)
- feat: check pipeline.Do to prevent confusion with Exec (#2517)
- Function stats, function kill, fcall and fcall_ro (#2486)
- feat: Add support for CLUSTER SHARDS command (#2507)
- feat(cmd): support for adding byte,bit parameters to the bitpos command (#2498)

### Fixed

- fix: eval api cmd.SetFirstKeyPos (#2501)
- fix: limit the number of connections created (#2441)
- fixed #2462  v9 continue support dragonfly,  it's Hello command return "NOAUTH Authentication required" error (#2479)
- Fix for internal/hscan/structmap.go:89:23: undefined: reflect.Pointer (#2458)
- fix: group lag can be null (#2448)

### Maintenance

- Updating to the latest version of redis (#2508)
- Allowing for running tests on a port other than the fixed 6380 (#2466)
- redis 7.0.8 in tests (#2450)
- docs: Update redisotel example for v9 (#2425)
- chore: update go mod, Upgrade golang.org/x/net version to 0.7.0 (#2476)
- chore: add Chinese translation (#2436)
- chore(deps): bump github.com/bsm/gomega from 1.20.0 to 1.26.0 (#2421)
- chore(deps): bump github.com/bsm/ginkgo/v2 from 2.5.0 to 2.7.0 (#2420)
- chore(deps): bump actions/setup-go from 3 to 4 (#2495)
- docs: add instructions for the HSet api (#2503)
- docs: add reading lag field comment (#2451)
- test: update go mod before testing(go mod tidy) (#2423)
- docs: fix comment typo (#2505)
- test: remove testify (#2463)
- refactor: change ListElementCmd to KeyValuesCmd. (#2443)
- fix(appendArg): appendArg case special type (#2489)

## [9.0.2](https://github.com/redis/go-redis/compare/v9.0.1...v9.0.2) (2023-02-01)

### Features

* upgrade OpenTelemetry, use the new metrics API. ([#2410](https://github.com/redis/go-redis/issues/2410)) ([e29e42c](https://github.com/redis/go-redis/commit/e29e42cde2755ab910d04185025dc43ce6f59c65))

## v9 2023-01-30

### Breaking

- Changed Pipelines to not be thread-safe any more.

### Added

- Added support for [RESP3](https://github.com/antirez/RESP3/blob/master/spec.md) protocol. It was
  contributed by @monkey92t who has done the majority of work in this release.
- Added `ContextTimeoutEnabled` option that controls whether the client respects context timeouts
  and deadlines. See
  [Redis Timeouts](https://redis.uptrace.dev/guide/go-redis-debugging.html#timeouts) for details.
- Added `ParseClusterURL` to parse URLs into `ClusterOptions`, for example,
  `redis://user:password@localhost:6789?dial_timeout=3&read_timeout=6s&addr=localhost:6790&addr=localhost:6791`.
- Added metrics instrumentation using `redisotel.IstrumentMetrics`. See
  [documentation](https://redis.uptrace.dev/guide/go-redis-monitoring.html)
- Added `redis.HasErrorPrefix` to help working with errors.

### Changed

- Removed asynchronous cancellation based on the context timeout. It was racy in v8 and is
  completely gone in v9.
- Reworked hook interface and added `DialHook`.
- Replaced `redisotel.NewTracingHook` with `redisotel.InstrumentTracing`. See
  [example](example/otel) and
  [documentation](https://redis.uptrace.dev/guide/go-redis-monitoring.html).
- Replaced `*redis.Z` with `redis.Z` since it is small enough to be passed as value without making
  an allocation.
- Renamed the option `MaxConnAge` to `ConnMaxLifetime`.
- Renamed the option `IdleTimeout` to `ConnMaxIdleTime`.
- Removed connection reaper in favor of `MaxIdleConns`.
- Removed `WithContext` since `context.Context` can be passed directly as an arg.
- Removed `Pipeline.Close` since there is no real need to explicitly manage pipeline resources and
  it can be safely reused via `sync.Pool` etc. `Pipeline.Discard` is still available if you want to
  reset commands for some reason.

### Fixed

- Improved and fixed pipeline retries.
- As usually, added support for more commands and fixed some bugs.
