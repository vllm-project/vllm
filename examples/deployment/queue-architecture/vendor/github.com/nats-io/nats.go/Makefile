# Local convenience targets for running tests against the synadia/ntf-server
# service. The CI `test` job (.github/workflows/ci.yaml) runs the same suite,
# with the tester attached as a docker service container.
#
# Quick start (sibling-container mode — full suite, matches CI):
#   make tester-up        # start the tester container on a dedicated docker network
#   make test-tester      # run the full suite inside a sibling container
#   make tester-down      # tear everything down
#
# Quick start (host-side mode — iterating on individual tests):
#   make tester-up-host   # start the tester with its ports published
#   make test T=TestName PKG=./test/...
#   make tester-down
#
# `make test` wraps the full `go test` incantation (modfile, internal_testing
# tag, -race, -p=1). The equivalent raw command is:
#   TESTER_NATS_URL=nats://localhost:4222 \
#     go test -modfile=go_test.mod -tags=internal_testing -race -p=1 ./... --failfast -vet=off
# `-p=1` is required: the tester does not tolerate concurrent CreateServer
# calls from independent test binaries.

# The tester image is pinned in ci.yaml (the single source of truth); parse it
# from there so server version bumps happen in one place.
TESTER_IMAGE   ?= $(shell sed -n 's|^ *image: *\(synadia/ntf-server:[^ ]*\).*|\1|p' .github/workflows/ci.yaml)
ifeq ($(strip $(TESTER_IMAGE)),)
$(error could not parse the tester image from .github/workflows/ci.yaml)
endif
TESTER_NAME    ?= nats-tester
TESTER_NETWORK ?= nats-tester-net
GO_IMAGE       ?= golang:1.26-alpine

# Host-side test runs (tester started with `make tester-up-host`).
# T limits the run to a single test (-run, verbose); PKG limits the packages.
TESTER_NATS_URL ?= nats://localhost:4222
PKG ?= ./...

.PHONY: tester-net tester-up tester-up-host tester-down tester-restart tester-logs test-tester test test-norace

# test runs the race-enabled suite, like CI. Examples:
#   make test                             # everything
#   make test T=TestSubSubject            # one test, verbose
#   make test PKG=./jetstream/test/...    # one package
test:
	TESTER_NATS_URL=$(TESTER_NATS_URL) go test -modfile=go_test.mod -tags=internal_testing -race -p=1 $(if $(T),-v -run '$(T)') $(PKG) --failfast -vet=off

# test-norace runs the TestNoRace* tests, which must run with the race
# detector off. T overrides the -run pattern within the NoRace suite.
test-norace:
	TESTER_NATS_URL=$(TESTER_NATS_URL) go test -modfile=go_test.mod -p=1 $(if $(T),-v) -run '$(or $(T),TestNoRace)' $(PKG) --failfast -vet=off

tester-net:
	@docker network inspect $(TESTER_NETWORK) >/dev/null 2>&1 || \
		docker network create $(TESTER_NETWORK)

# tester-up runs the tester WITHOUT host port publishing. Use this when running
# tests via `make test-tester` (sibling-container mode — the test container is
# on the same docker network as the tester, so host port publishing is not
# only unnecessary, it actively breaks server bring-up: docker-proxy holds
# 0.0.0.0:<port> inside the container's net namespace, racing the tester's
# localhost:0 port-reservation handover and causing intermittent
# "bind: address already in use" failures.
#
# No sysctl-constrained ephemeral port range either: the spawned servers can
# use the full kernel default (typically 32768-60999), so churn-driven TIME_WAIT
# accumulation never crowds the pool. The constrained range is only meaningful
# when ports must match a host-published range (see tester-up-host).
tester-up: tester-net
	docker run -d \
		--name $(TESTER_NAME) \
		--network $(TESTER_NETWORK) \
		--restart unless-stopped \
		-e NATS_ADVERTISE=$(TESTER_NAME) \
		$(TESTER_IMAGE) serve
	@echo "Tester running on docker network $(TESTER_NETWORK) as host '$(TESTER_NAME)'"
	@echo "Sibling-container mode: use 'make test-tester' to run the suite."
	@echo "For host-side dev (running 'go test' directly), use 'make tester-up-host' instead."

# tester-up-host runs the tester WITH host port publishing for host-side dev
# workflows (running 'go test' directly from your terminal). The
# ip_local_port_range sysctl is intentionally narrowed to match the published
# port range so the tester's net.Listen(":0") picks ports the host can reach.
# Note: in this mode, docker-proxy on macOS races the tester's port handover
# and causes intermittent server-creation failures (~5-10% of runs in heavy
# suites); rerun the failing test or restart the tester if it happens.
# Sibling-container mode (tester-up + make test-tester) does not have this issue.
tester-up-host: tester-net
	docker run -d \
		--name $(TESTER_NAME) \
		--network $(TESTER_NETWORK) \
		--restart unless-stopped \
		--sysctl net.ipv4.ip_local_port_range="30000 31000" \
		-p 4222:4222 \
		-p 30000-31000:30000-31000 \
		-e NATS_ADVERTISE=localhost \
		$(TESTER_IMAGE) serve
	@echo "Tester running on docker network $(TESTER_NETWORK) as host '$(TESTER_NAME)'"
	@echo "Run the full suite:  make test"
	@echo "Run a single test:   make test T=TestName PKG=./test/..."
	@echo "Or directly:         TESTER_NATS_URL=nats://localhost:4222 go test -modfile=go_test.mod -tags=internal_testing -race -p=1 -v -run TestName ./test/..."

# tester-down stops AND removes the container; logs are lost. Use tester-restart
# instead to keep the container (and its logs) around for debugging.
tester-down:
	-docker rm -f $(TESTER_NAME)
	-docker network rm $(TESTER_NETWORK)

tester-restart:
	-docker restart $(TESTER_NAME)
	@echo "Tester restarted; logs preserved (use 'make tester-logs')"

tester-logs:
	docker logs -f $(TESTER_NAME)

# test-tester runs the same suite as the CI `test` job: the NoRace pass first
# (needs the race detector off), then the full race-enabled suite with
# -tags=internal_testing.
test-tester: tester-net
	docker run --rm \
		--network $(TESTER_NETWORK) \
		-v $(CURDIR):/src \
		-w /src \
		-e TESTER_NATS_URL=nats://$(TESTER_NAME):4222 \
		-e CGO_ENABLED=1 \
		$(GO_IMAGE) sh -c '\
			apk add --no-cache gcc libc-dev git make >/dev/null && \
			go test -modfile=go_test.mod -v -run=TestNoRace -p=1 ./... --failfast -vet=off && \
			go test -modfile=go_test.mod -tags=internal_testing -race -v -p=1 ./... --failfast -vet=off'
