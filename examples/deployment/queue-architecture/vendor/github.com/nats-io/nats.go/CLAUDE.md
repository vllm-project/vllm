# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official Go client library for the NATS messaging system. Provides core pub/sub, request/reply, JetStream (streams, consumers, KV, object store), and a micro services framework. Module path: `github.com/nats-io/nats.go`.

## Build and Test Commands

This project uses a **dual module** setup: `go.mod` for production (minimal deps) and `go_test.mod` for testing (protobuf encoder + jwt + nkeys + nuid). Always use `-modfile=go_test.mod` when running tests.

Integration tests (everything in `./test/`, `./jetstream/test/`, `./micro/test/`) run against a remote **ntf-server** docker service (`synadia/ntf-server:v0.0.5-nats-2.14.3`) instead of an in-process nats-server. Bring it up first via the Makefile, then run tests with `TESTER_NATS_URL` pointing at it.

```bash
# Start the tester (host-side mode publishes the tester's ports so `go test`
# from your terminal can reach the spawned NATS servers via localhost).
make tester-up-host

# Run all tests against the tester (single command, covers both white-box
# tests at the repo root and integration tests in ./test/, ./jetstream/test/,
# ./micro/test/).
TESTER_NATS_URL=nats://localhost:4222 \
  go test -modfile=go_test.mod -race -v -p=1 ./... --failfast -vet=off -tags=internal_testing

# Run NoRace tests (must be run separately, without -race flag)
TESTER_NATS_URL=nats://localhost:4222 \
  go test -modfile=go_test.mod -v -run=TestNoRace -p=1 ./... --failfast -vet=off

# Run a specific test
TESTER_NATS_URL=nats://localhost:4222 \
  go test -modfile=go_test.mod -race -tags=internal_testing -run TestName ./...

# Run tests for a specific package
TESTER_NATS_URL=nats://localhost:4222 go test -modfile=go_test.mod -race ./jetstream/test/... --failfast
TESTER_NATS_URL=nats://localhost:4222 go test -modfile=go_test.mod -race ./micro/test/... --failfast

# Makefile wrappers for the above (TESTER_NATS_URL defaults to nats://localhost:4222)
make test                          # full race-enabled suite with internal_testing tag
make test T=TestName PKG=./test/... # single test, verbose
make test-norace                   # NoRace suite

# Stop the tester
make tester-down

# Alternative: run the full suite inside an alpine sibling container (matches CI).
# Doesn't need TESTER_NATS_URL or tester-up-host — the Makefile target handles it.
make test-tester

# Build
go build ./...

# Formatting
go fmt -modfile=go_test.mod ./...

# Vet
go vet -modfile=go_test.mod ./...

# Static analysis (as CI does it)
staticcheck -modfile=go_test.mod ./...

# Linting (golangci-lint runs only on jetstream/; the modfile GOFLAGS lets it
# typecheck jetstream/test, whose deps live in go_test.mod only)
GOFLAGS="-mod=mod -modfile=go_test.mod" golangci-lint run --timeout 5m0s ./jetstream/...

# Spell check
find . -type f -name "*.go" | xargs misspell -error -locale US

# Update test dependencies (never change go.mod for test deps)
go mod tidy -modfile=go_test.mod
```

If `TESTER_NATS_URL` is unset, the integration tests skip via `t.Skip` rather than fail — so default `go test ./...` on a fresh checkout just runs the white-box unit tests.

## Important Build Tags

- **`internal_testing`** -- Exposes internal test helpers (e.g., `AddMsgFilter`, `CloseTCPConn`) from `testing_internal.go`. Required for some tests in `./test/`.
- **`!race && !skip_no_race_tests`** -- NoRace tests in `test/norace_test.go` only run when the race detector is OFF.
- **`compat`** -- Compatibility tests in `test/compat_test.go` (connect to an external NATS server via `NATS_URL`).
- **`go1.23`** -- Iterator-based tests in `test/nats_iter_test.go` and `nats_iter.go`.

## CI Pipeline (ci.yaml)

1. **lint** -- `go fmt`, `go vet`, `staticcheck`, `misspell` (all packages), `golangci-lint` (jetstream only).
2. **test** -- Matrix of Go 1.25 and 1.26. Runs inside an `alpine` container on the same docker network as the `synadia/ntf-server` service, which is started with `command: serve --advertise nats` (the integration tests dial the spawned NATS servers by service name). Two steps: NoRace tests (without `-race`), then full race-enabled tests with `-tags=internal_testing` (the Go 1.26 row runs `scripts/cov.sh` for coverage instead of the plain race run).

## Project Structure

```
nats.go                 # Core connection, pub/sub, request/reply (~6500 lines)
parser.go               # Client-side protocol parser
ws.go                   # WebSocket transport support
js.go                   # Legacy JetStream API (deprecated, see jetstream/)
jsm.go                  # Legacy JetStream management
kv.go                   # Legacy KeyValue API
object.go               # Legacy Object Store API
enc.go                  # EncodedConn (deprecated)
netchan.go              # Go channel bindings
timer.go                # Internal timer utilities
context.go              # Context-aware request methods
nats_iter.go            # Go 1.23+ iterator support (go:build go1.23)
testing_internal.go     # Internal test hooks (go:build internal_testing)

jetstream/              # New JetStream API (preferred over legacy)
  jetstream.go          #   Top-level JetStream interface
  stream.go             #   Stream management
  stream_config.go      #   Stream configuration types
  consumer.go           #   Consumer management
  consumer_config.go    #   Consumer configuration types
  pull.go               #   Pull consumer implementation
  push.go               #   Push consumer (deprecated)
  ordered.go            #   Ordered consumer
  publish.go            #   JetStream publish methods
  kv.go                 #   KeyValue store
  object.go             #   Object store
  message.go            #   JetStream message types
  errors.go             #   JetStream error types
  test/                 #   Integration tests (package test, uses testservice)

micro/                  # Micro services framework
  service.go            #   Service interface and implementation
  request.go            #   Request handling
  test/                 #   Integration tests

internal/
  parser/               # NATS protocol parser (used by core client)
  syncx/                # Concurrent map utility

encoders/
  builtin/              # Default encoders (JSON, GOB, string)
  protobuf/             # Protocol Buffers encoder

test/                   # Integration tests for core package (package test)
  testservice_helper_test.go # withServer / withServerInstance / newTester / dialInstance helpers
  helper_test.go        #   Shared utility helpers (Wait, checkFor, getStableNumGoroutine, ...)
  norace_test.go        #   Tests that cannot run with -race (build tag guarded)
  js_internal_test.go   #   Tests requiring internal_testing tag
  configs/certs/        #   CA/server/key PEMs loaded by TLS tests (the *.conf files are unused)

bench/                  # Benchmarking utilities
examples/               # Example command-line tools (nats-pub, nats-sub, etc.)
scripts/cov.sh          # Coverage collection script (run by the CI coverage matrix row)
```

The tester client is an external test-only dependency: `github.com/synadia-io/orbit.go/ntf-client`
(package `ntf`, imported under the alias `testservice`). It lives in `go_test.mod` only.

## Test Architecture

- **Root `nats_test.go`** (package `nats`) -- White-box unit tests with access to unexported internals.
- **`test/`** (package `test`) -- Black-box integration tests. Tests bring up a NATS server via the testservice helpers (`withServer`, `withJSServer`, `withJSCluster`, ...) which talk to a remote `synadia/ntf-server:v0.0.5-nats-2.14.3` docker service over NATS. `TESTER_NATS_URL` must point at that service; if unset, tests skip via `t.Skip`.
- **`jetstream/test/`** (package `test`) -- Integration tests for the new JetStream API, same testservice harness.
- **`micro/test/`** (package `micro_test`) -- Integration tests for the micro services framework, same testservice harness.
- **NoRace tests** -- Prefixed `TestNoRace*`, guarded by `//go:build !race && !skip_no_race_tests`. Must be run separately without `-race`.
- Tests always run with `-p=1` (no parallel packages) because the tester serializes some bookkeeping that doesn't tolerate concurrent CreateServer calls from independent test binaries.

## Code Conventions

- **License header** -- Every `.go` file starts with the Apache 2.0 license header (Copyright year range).
- **Error variables** -- Exported errors defined as `var Err... = errors.New("nats: ...")` in `nats.go`. JetStream errors in `jetstream/errors.go` follow the same pattern.
- **Options pattern** -- Connection options use functional options: `nats.Connect(url, nats.Name("myapp"), nats.MaxReconnects(5))`. JetStream and micro use similar patterns.
- **No external dependencies in production** -- Only `klauspost/compress`, `nkeys`, `nuid` in `go.mod`. Test deps (protobuf, jwt, etc.) are isolated in `go_test.mod`. PRs adding dependencies are scrutinized heavily.
- **Commits require sign-off** -- Use `git commit -s` (DCO: `Signed-off-by`).
- **US English spelling** -- Enforced by `misspell -locale US` in CI.
- **Interface-driven design** -- JetStream and micro packages define interfaces (`JetStream`, `Stream`, `Consumer`, `Service`) with concrete unexported implementations.

## Key Types

- `nats.Conn` -- Core connection, handles all NATS protocol operations.
- `nats.Msg` -- Message type for pub/sub and request/reply.
- `nats.Subscription` -- Represents a subscription (sync, async, or channel-based).
- `jetstream.JetStream` -- Entry point for new JetStream API (created via `jetstream.New(nc)`).
- `jetstream.Stream`, `jetstream.Consumer` -- Stream and consumer management.
- `micro.Service` -- Micro service instance (created via `micro.AddService(nc, config)`).
