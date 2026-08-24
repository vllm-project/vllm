# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. This is
the shared, tool-agnostic source of truth: Claude Code loads it through
`CLAUDE.md` (which imports this file), and other agents (Codex, Cursor, Aider,
Zed, …) read `AGENTS.md` directly. Edit repository guidance here, not in
`CLAUDE.md`.

## Repository

go-redis is the official Redis client for Go. Module path:
`github.com/redis/go-redis/v9` (Go 1.24+). The repo is a multi-module workspace
— every directory containing a `go.mod` is built and tested independently:

- root (`github.com/redis/go-redis/v9`) — the client library.
- `extra/redisotel`, `extra/redisotel-native`, `extra/redisprometheus`,
  `extra/rediscensus`, `extra/rediscmd` — instrumentation adapters with their
  own module paths (so they can pin large telemetry deps without forcing them on
  root consumers).
- `internal/customvet` — custom `go vet` analyzers (also its own module).
- `maintnotifications/e2e`, `doctests`, `fuzz`, examples under `example/` —
  separate modules.

The Makefile iterates over every `go.mod` (`GO_MOD_DIRS`) when running
`test.ci`, `go_mod_tidy`, etc. When you add a dependency in one module, you
almost never need to update the others.

## Common commands

Tests run against a Redis stack started via Docker Compose. Profiles in
`docker-compose.yml` control which services come up (`standalone`, `cluster`,
`sentinel`, `all`, `e2e`).

```sh
make docker.start                # bring up the full test stack (profile: all)
make docker.stop
make test                        # docker.start -> test.ci -> docker.stop
make test.ci                     # run tests assuming containers are already up
make test.ci.skip-vectorsets     # when REDIS_VERSION < 8
make bench                       # go test -bench=. (root module only)
make fmt                         # gofumpt + goimports -local github.com/redis/go-redis
make build
make go_mod_tidy                 # go mod tidy across every module
```

E2E (maintenance notifications) needs the extra `cae-resp-proxy` service:

```sh
make test.e2e                    # starts e2e profile, runs ./maintnotifications/e2e/, tears down
make test.e2e.docker             # subset that runs inside docker
make test.e2e.logic              # logic-only tests, no proxy required
```

Run a single test. The root suite is Ginkgo-based (`bsm/ginkgo` + `bsm/gomega`
forks), so `go test -run` matches the Go-level wrapper and you focus a spec with
the Ginkgo flag:

```sh
go test -run TestGinkgoSuite . -ginkgo.focus="ZAdd"
go test -run TestGinkgoSuite . -ginkgo.focus="cluster"
```

Plain `go test` tests (most files outside the Ginkgo suite, e.g. `internal/...`,
`maintnotifications/...`) work the usual way:

```sh
go test -run TestConnStateMachine ./internal/pool/...
go test -race -run TestCircuitBreaker ./maintnotifications/...
```

Env knobs (passed through the Makefile):

- `REDIS_VERSION` — e.g. `8.8`. Drives both the test image tag and
  `main_test.go` version-gating (`SkipBeforeRedisVersion` /
  `SkipAfterRedisVersion`).
- `CLIENT_LIBS_TEST_IMAGE` — full image ref, e.g.
  `redislabs/client-libs-test:8.8-m03`.
- `RE_CLUSTER=true` — run against a Redis Enterprise cluster instead of the
  docker-compose stack (the suite then skips ring/sentinel/TLS-cluster setup).
- `RCE_DOCKER=true` — Redis CE in docker (default for `make test`).
- `REDIS_PORT` — override the default standalone port (`6380`).

CI also runs the custom vet tool:
`go vet -vettool ./internal/customvet/customvet ./...`. The `setval` analyzer
requires every `Cmder` with a `Result()` to also have a `SetVal()`.

## Architecture

### Client types (root package)

All clients are in the root package and share most plumbing:

- `Client` (`redis.go`) — single-node client.
- `ClusterClient` (`osscluster.go`) — Redis Cluster aware. `osscluster_router.go`
  routes commands to the right shard; `internal/routing/` handles cluster-wide
  aggregation policies (e.g. fan-out for `KEYS`, `DBSIZE`).
- `Ring` (`ring.go`) — client-side sharding across independent Redis nodes
  (consistent hashing, no cluster protocol).
- Failover client (`sentinel.go`) — Sentinel-managed failover.
- `UniversalClient` (`universal.go`) — wrapper that picks one of the above based
  on options.

Command surface lives in topical files: `string_commands.go`, `hash_commands.go`,
`stream_commands.go`, `search_commands.go`, `vectorset_commands.go`, etc. Each
file defines methods on the shared `Cmdable` interface so every client type gets
the same API.

### Hooks (`redis.go` `hooksMixin`)

Three hook chains run around every operation: `DialHook`, `ProcessHook`,
`ProcessPipelineHook`. Hooks are registered via `client.AddHook(...)` and chain
in FIFO order; each hook must call `next` to continue. When a hook wraps an
error, it must call `cmd.SetErr(wrappedErr)` so the typed-error helpers
(`redis.IsLoadingError`, `IsMovedError`, etc. in `error.go`) keep working through
`errors.As`. The README has a longer pipeline-hook example.

### Connection pool (`internal/pool`)

Owns dialing, idle/active connection bookkeeping, conn state (`conn_state.go`),
pubsub-conn lifecycle (`pubsub.go`), and the dial-retry/backoff logic that powers
`DialerRetries` / `DialerRetryBackoff` (also exposed at `dial_retry_backoff.go`
in the root). `OnConnect`, `MinIdleConns`, and the buffer-size options
(`ReadBufferSize`/`WriteBufferSize`, default 32 KiB since v9.12) flow through
here.

### Protocol (`internal/proto`)

RESP2/RESP3 reader and writer. Push notifications (RESP3 `>`-prefixed frames) are
peeked here and dispatched via the `push/` package. The `push.Registry` lets
callers register handlers for specific notification names;
`maintnotifications/push_notification_handler.go` is how `maintnotifications`
plugs in.

### Maintenance notifications (`maintnotifications/`)

This is a non-trivial subsystem worth understanding before touching
cluster/handoff code. It listens for RESP3 push notifications about cluster
maintenance (`MOVING`, `MIGRATING`, `MIGRATED`, `FAILING_OVER`, `FAILED_OVER`
for standalone; `SMIGRATING`, `SMIGRATED` for cluster) and performs seamless
connection handoff to new endpoints. Key pieces:

- `manager.go` — coordinates state transitions.
- `handoff_worker.go` — moves in-flight ops to new connections.
- `pool_hook.go` — integrates with `internal/pool` to mark/replace connections.
- `circuit_breaker.go` — backs off when the upstream is unhealthy.
- `state.go` — per-connection state machine.
- E2E coverage lives in `maintnotifications/e2e/` and drives a fault-injector /
  RESP proxy (`cae-resp-proxy`).

Configuration is via `redis.Options.MaintNotificationsConfig`; modes are
`ModeAuto` (default), `ModeEnabled` (require server support), `ModeDisabled`.
RESP3 (`Protocol: 3`) is required.

### Authentication (`auth/`, `internal/auth/streaming`)

Four credential sources, in priority order: streaming provider (e.g. Entra ID
via `go-redis-entraid`), context-based provider, function provider, static
`Username`/`Password`. The streaming provider is what enables token rotation
without reconnecting — the listener in `auth/reauth_credentials_listener.go`
issues `AUTH` on each refresh.

### Internal helpers

- `internal/hscan` — struct scanning for `HGETALL` results (`Scan` interface
  re-exported as `redis.Scanner`).
- `internal/hashtag` — extracts `{tag}` segments for cluster slot routing.
- `internal/routing` — aggregator policies and shard pickers used by
  `ClusterClient` for multi-shard commands.
- `internal/otel` — small OpenTelemetry shim used to keep root free of telemetry
  deps; full instrumentation lives in `extra/redisotel-native`.

## Architectural specs

Read the relevant design doc **before** changing code in that subsystem. They
cover invariants and decisions that aren't obvious from the code, and are plain
markdown any tool or editor can open:

- `.claude/specs/pool.md` — connection pool: `wantConn` queue and FIFO
  discipline, `ConnState` machine, dial retry/backoff, hook integration, the
  re-auth/handoff coexistence contract.
- `.claude/specs/cluster-routing.md` — slot computation, MOVED/ASK redirection,
  request/response policies, aggregators, replica routing, topology reload,
  cross-slot rules.
- `.claude/specs/maintnotifications.md` — RESP3 push notification protocol, mode
  handshake, per-conn state, handoff worker pool, circuit breaker, endpoint-type
  resolution, cluster vs. standalone differences.

## Conventions

- New `Cmder` type → also implement `SetVal` (the custom vet `setval` check
  enforces this; `SetErr` is on the embedded `baseCmd`).
- Wrap errors with custom error types that implement `Unwrap`, or use
  `fmt.Errorf("...: %w", err)`. Always call `cmd.SetErr(...)` after wrapping so
  typed-error checks still pass.
- `gofumpt` + `goimports -local github.com/redis/go-redis` is the formatter
  (`make fmt`); CI runs both.
- Don't log directly — use `internal.Logger` (set via `redis.SetLogger`);
  `logging.Disable()` is called in tests.
- Version-gate Redis-version-specific tests with `SkipBeforeRedisVersion` /
  `SkipAfterRedisVersion` rather than skipping at the suite level.

### Commits and PRs

Conventional Commits, short and exact — `<type>(<scope>): <imperative summary>`.
Subject ≤50 chars (hard cap 72), imperative ("add", not "added"), no trailing
period. Body only when the *why* isn't obvious from the diff; wrap at 72.

- Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore` (also
  `build`, `ci`, `style`, `revert`).
- Scope = the subsystem touched, lowercase: `pool`, `conn`, `pubsub`,
  `sentinel`, `retry`, `command`/`cmd`, `vectorset`, `otel`, `streams`, `push`,
  `deps`, `ci`, `tests`, `docs`. Omit only for genuinely cross-cutting changes.
- Breaking change: `feat(scope)!: ...` plus a `BREAKING CHANGE:` body line.
  Reference issues/PRs at the end — `Closes #42`, `Refs #17`.
- **No AI-attribution trailer.** Do not add `Co-Authored-By: …`, "Generated with
  …", or any AI-attribution line to commits or PR bodies in this repo.

## Repo-specific tooling

`.claude/` holds shared AI config:

- `commands/` — slash commands (e.g. `check-ci`, which summarizes a PR's CI).
- `skills/` — task playbooks: `testing`, `add-command`, `commit-style`,
  `update-ci-image`, `prepare-release`.
- `specs/` — the architecture docs listed above.

For Claude Code, the skills auto-trigger from their descriptions. For other
tools, each `SKILL.md` is plain markdown you can open and follow directly.
