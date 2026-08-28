# Testing

## TL;DR

```shell
make tester-up-host                  # once: start the test-server manager (requires docker)
make test T=TestName PKG=./test/...  # iterate on a single test
make test                            # full race-enabled suite
make tester-down                     # tear everything down
```

## How the suite is set up

The repo uses two module files: `go.mod` for production (minimal dependencies)
and `go_test.mod` for testing. Always pass `-modfile=go_test.mod` to `go test`;
the make targets below do this for you.

Unit tests live at the repo root (package `nats`, white-box) and run with plain
`go test`. Integration tests live in `./test`, `./jetstream/test`, and
`./micro/test`; they bring up real NATS servers by talking to a **tester**
service (the `synadia/ntf-server` docker image, driven through the
`github.com/synadia-io/orbit.go/ntf-client` package) which spawns `nats-server`
instances, clusters, and super-clusters on demand. Tests find the tester via
the `TESTER_NATS_URL` environment variable.

If `TESTER_NATS_URL` is unset, the integration tests **skip silently** — a
green `go test ./...` on a fresh checkout means only the unit tests ran.

## Host-side mode (iterating on individual tests)

`make tester-up-host` starts the tester with its ports published, so `go test`
run from your terminal can reach the spawned servers via localhost.

```shell
make tester-up-host
make test T=TestSubSubject PKG=./test/...  # one test, verbose
make test PKG=./jetstream/test/...         # one package
make test                                  # everything
make test-norace                           # NoRace tests (race detector off)
```

`make test` wraps the full invocation, which is equivalent to:

```shell
TESTER_NATS_URL=nats://localhost:4222 \
  go test -modfile=go_test.mod -tags=internal_testing -race -p=1 ./... --failfast -vet=off
```

`-p=1` is required: the tester does not tolerate concurrent CreateServer calls
from independent test binaries.

Known caveat: on macOS, docker-proxy races the tester's port handover, so in
heavy suites roughly 5-10% of server creations can fail with
`bind: address already in use`. Rerun the failing test, or use
sibling-container mode for full-suite runs.

## Sibling-container mode (full suite, matches CI)

```shell
make tester-up
make test-tester
make tester-down
```

`make test-tester` runs the whole suite (NoRace pass, then the race-enabled
pass) inside a Go container on the same docker network as the tester — no
published ports, so the docker-proxy race above does not apply. This is the
same shape CI uses, with the tester attached as a service container
(`.github/workflows/ci.yaml`).

## NoRace tests

Tests prefixed `TestNoRace` are guarded by `//go:build !race &&
!skip_no_race_tests` and must run with the race detector off:
`make test-norace`.

## Build tags

- `internal_testing` — exposes internal test hooks from `testing_internal.go`;
  required by some tests in `./test`. The make targets set it.
- `skip_no_race_tests` — excludes the NoRace tests from a non-race run (used
  by `scripts/cov.sh`).
- `compat` — compatibility tests in `test/compat_test.go`, which connect to an
  external NATS server via `NATS_URL`.
- `go1.23` — iterator-based tests in `test/nats_iter_test.go`.

## Coverage

```shell
TESTER_NATS_URL=nats://localhost:4222 ./scripts/cov.sh
```

Merges unit and integration coverage into `acc.out` and opens the HTML report
(CI passes an argument to skip the browser).

## Troubleshooting

- `cannot reach the tester at ...` — the tester is not running (or its ports
  are not published); run `make tester-up-host`.
- Integration tests skip — `TESTER_NATS_URL` is unset.
- `make tester-logs` — follow the tester's logs; server spawn and config
  errors show up there.
- `make tester-restart` — restart the tester but keep its logs;
  `make tester-down` removes the container (and its logs).
- Updating test-only dependencies: `go mod tidy -modfile=go_test.mod` (never
  change the main `go.mod` for test dependencies).
