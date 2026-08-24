GO_MOD_DIRS := $(shell find . -type f -name 'go.mod' -exec dirname {} \; | sort)
REDIS_VERSION ?= 8.10
RE_CLUSTER ?= false
RCE_DOCKER ?= true
CLIENT_LIBS_TEST_IMAGE ?= redislabs/client-libs-test:8.10.0

docker.start:
	export RE_CLUSTER=$(RE_CLUSTER) && \
	export RCE_DOCKER=$(RCE_DOCKER) && \
	export REDIS_VERSION=$(REDIS_VERSION) && \
	export CLIENT_LIBS_TEST_IMAGE=$(CLIENT_LIBS_TEST_IMAGE) && \
	docker compose --profile all up -d --quiet-pull

docker.stop:
	docker compose --profile all down

docker.e2e.start:
	@echo "Starting Redis and cae-resp-proxy for E2E tests..."
	docker compose --profile e2e up -d --quiet-pull
	@echo "Waiting for services to be ready..."
	@sleep 3
	@echo "Services ready!"

docker.e2e.stop:
	@echo "Stopping E2E services..."
	docker compose --profile e2e down

test:
	$(MAKE) docker.start
	@if [ -z "$(REDIS_VERSION)" ]; then \
		echo "REDIS_VERSION not set, running all tests"; \
		$(MAKE) test.ci; \
	else \
		MAJOR_VERSION=$$(echo "$(REDIS_VERSION)" | cut -d. -f1); \
		if [ "$$MAJOR_VERSION" -ge 8 ]; then \
			echo "REDIS_VERSION $(REDIS_VERSION) >= 8, running all tests"; \
			$(MAKE) test.ci; \
		else \
			echo "REDIS_VERSION $(REDIS_VERSION) < 8, skipping vector_sets tests"; \
			$(MAKE) test.ci.skip-vectorsets; \
		fi; \
	fi
	$(MAKE) docker.stop

test.ci:
	set -e; for dir in $(GO_MOD_DIRS); do \
	  echo "go test in $${dir}"; \
	  (cd "$${dir}" && \
	    export RE_CLUSTER=$(RE_CLUSTER) && \
	    export RCE_DOCKER=$(RCE_DOCKER) && \
	    export REDIS_VERSION=$(REDIS_VERSION) && \
	    go mod tidy && \
	    go vet && \
	    go test -v -coverprofile=coverage.txt -covermode=atomic ./... -race -skip Example); \
	done
	cd internal/customvet && go build .
	go vet -vettool ./internal/customvet/customvet

test.ci.skip-vectorsets:
	set -e; for dir in $(GO_MOD_DIRS); do \
	  echo "go test in $${dir} (skipping vector sets)"; \
	  (cd "$${dir}" && \
	    export RE_CLUSTER=$(RE_CLUSTER) && \
	    export RCE_DOCKER=$(RCE_DOCKER) && \
	    export REDIS_VERSION=$(REDIS_VERSION) && \
	    go mod tidy && \
	    go vet && \
	    go test -v -coverprofile=coverage.txt -covermode=atomic ./... -race \
	      -run '^(?!.*(?:VectorSet|vectorset|ExampleClient_vectorset)).*$$' -skip Example); \
	done
	cd internal/customvet && go build .
	go vet -vettool ./internal/customvet/customvet

# Replay the parametrized command integration suites through the AutoPipeliner
# Cmdable faces (blocking + async) to prove the commands behave the same
# batched as on a plain client. Selected via GOREDIS_TEST_SUBJECT (see
# newUniversalSubject in main_test.go). Requires the docker env
# (make docker.start).
#
# The focus covers every parametrized suite (some via the Commands/BitCount
# substrings — Describe names differ in case). Skipped on purpose: DDL Commands (needs the cluster topology the
# lightweight BeforeSuite doesn't build), HotKeys Commands (same), and
# AutoPipeline Blocking Commands (the AP suite itself — running it through an
# AP subject would nest engines).
test.autopipeline-subjects:
	# RE_CLUSTER=true forces the lightweight BeforeSuite (no sentinel/ring/cluster
	# setup): the command suite only needs the standalone Redis, and running the
	# full stateful BeforeSuite twice (once per subject) against the same server
	# corrupts replication state and fails the second run. Both faces then run
	# cleanly against the same env. The explicit -timeout keeps a hang from
	# eating go test's 10m default per subject.
	set -e; for subj in ap-blocking ap-async; do \
	  echo "=== command suite via GOREDIS_TEST_SUBJECT=$$subj ==="; \
	  (export RE_CLUSTER=true && \
	   export RCE_DOCKER=$(RCE_DOCKER) && \
	   export REDIS_VERSION=$(REDIS_VERSION) && \
	   export GOREDIS_TEST_SUBJECT=$$subj && \
	   go test -v . -race -skip Example -run TestGinkgoSuite -timeout 8m \
	     -ginkgo.focus='Commands|RediSearch commands|Probabilistic commands|RedisTimeseries commands|Redis VectorSet commands|BitCount|ScanIterator|Advanced JSON' \
	     -ginkgo.skip='AutoPipeline Blocking Commands|DDL Commands|HotKeys Commands'); \
	done

bench:
	export RE_CLUSTER=$(RE_CLUSTER) && \
	export RCE_DOCKER=$(RCE_DOCKER) && \
	export REDIS_VERSION=$(REDIS_VERSION) && \
	go test ./... -test.run=NONE -test.bench=. -test.benchmem -skip Example -timeout 11m

test.e2e:
	@echo "Running E2E tests with auto-start proxy..."
	$(MAKE) docker.e2e.start
	@echo "Running tests..."
	@E2E_SCENARIO_TESTS=true go test -v ./maintnotifications/e2e/ -timeout 30m || ($(MAKE) docker.e2e.stop && exit 1)
	$(MAKE) docker.e2e.stop
	@echo "E2E tests completed!"

test.e2e.docker:
	@echo "Running Docker-compatible E2E tests..."
	$(MAKE) docker.e2e.start
	@echo "Running unified injector tests..."
	@E2E_SCENARIO_TESTS=true go test -v -run "TestUnifiedInjector|TestCreateTestFaultInjectorLogic|TestFaultInjectorClientCreation" ./maintnotifications/e2e/ -timeout 10m || ($(MAKE) docker.e2e.stop && exit 1)
	$(MAKE) docker.e2e.stop
	@echo "Docker E2E tests completed!"

test.e2e.logic:
	@echo "Running E2E logic tests (no proxy required)..."
	@E2E_SCENARIO_TESTS=true \
		REDIS_ENDPOINTS_CONFIG_PATH=/tmp/test_endpoints_verify.json \
		FAULT_INJECTION_API_URL=http://localhost:8080 \
		go test -v -run "TestCreateTestFaultInjectorLogic|TestFaultInjectorClientCreation" ./maintnotifications/e2e/
	@echo "Logic tests completed!"

.PHONY: all test test.ci test.ci.skip-vectorsets test.autopipeline-subjects bench fmt test.e2e test.e2e.logic docker.e2e.start docker.e2e.stop

build:
	export RE_CLUSTER=$(RE_CLUSTER) && \
	export RCE_DOCKER=$(RCE_DOCKER) && \
	export REDIS_VERSION=$(REDIS_VERSION) && \
	go build .

fmt:
	gofumpt -w ./
	goimports -w  -local github.com/redis/go-redis ./

go_mod_tidy:
	set -e; for dir in $(GO_MOD_DIRS); do \
	  echo "go mod tidy in $${dir}"; \
	  (cd "$${dir}" && \
	    go get -u ./... && \
	    go mod tidy); \
	done
