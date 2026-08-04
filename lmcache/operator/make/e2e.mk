##@ E2E (no-GPU)

# Both *-cluster targets need a registry-reachable IMG and refuse the
# scaffold default (controller:latest) which is meaningless to a remote
# cluster's image puller. Used as a prereq by every cluster-targeting
# recipe in both e2e.mk and e2e-gpu.mk.
.PHONY: _require-img
_require-img:
	@if [ -z "$(IMG)" ] || [ "$(IMG)" = "controller:latest" ]; then \
		echo "ERROR: pass IMG=<registry/image:tag> pointing at an image the cluster can pull"; \
		exit 1; \
	fi

# Unique Kind cluster name per `make test-e2e-kind` invocation so
# concurrent runs never collide and cleanup only touches THIS run's
# cluster. Set KIND_CLUSTER explicitly to target an existing cluster
# (you're responsible for cleanup in that case).
ifndef KIND_CLUSTER
KIND_CLUSTER := operator-test-e2e-$(shell date +%s%N | tail -c 9)
endif

.PHONY: setup-test-e2e-kind
setup-test-e2e-kind: ## Create a fresh, uniquely-named Kind cluster for e2e tests
	@command -v $(KIND) >/dev/null 2>&1 || { \
		echo "Kind is not installed. Please install Kind manually."; \
		exit 1; \
	}
	@echo "Creating Kind cluster '$(KIND_CLUSTER)'..."
	@$(KIND) create cluster --name $(KIND_CLUSTER)

# setup-test-e2e-kind is invoked from inside the trapped shell rather
# than as a Make prerequisite, so a setup failure still triggers
# cleanup (prerequisites run BEFORE the recipe's shell installs the
# trap). KIND_CLUSTER is passed to both sub-makes so they target THIS
# run's cluster, not a newly-computed unique name. SIGKILL is
# uncatchable; on that rare path the cluster leaks under its unique
# name (`kind delete cluster --name ...` to clean up manually).
.PHONY: test-e2e-kind
test-e2e-kind: manifests generate fmt vet ## Run the e2e tests. Expected an isolated environment using Kind.
	@echo "==> Using Kind cluster: $(KIND_CLUSTER) (auto-deleted after the run)"
	@trap '$(MAKE) --no-print-directory cleanup-test-e2e-kind KIND_CLUSTER=$(KIND_CLUSTER)' EXIT INT TERM; \
	 $(MAKE) --no-print-directory setup-test-e2e-kind KIND_CLUSTER=$(KIND_CLUSTER) && \
	 KIND=$(KIND) KIND_CLUSTER=$(KIND_CLUSTER) go test -tags=e2e ./test/e2e/ -v -ginkgo.v -timeout 30m

.PHONY: cleanup-test-e2e-kind
cleanup-test-e2e-kind: ## Delete the Kind cluster used for e2e tests (idempotent)
	@if $(KIND) get clusters 2>/dev/null | grep -qx "$(KIND_CLUSTER)"; then \
		echo "Deleting Kind cluster '$(KIND_CLUSTER)'..."; \
		$(KIND) delete cluster --name $(KIND_CLUSTER); \
	else \
		echo "Kind cluster '$(KIND_CLUSTER)' does not exist; nothing to delete."; \
	fi

# Run the smoke suite against an already-configured cluster (no Kind setup).
# Caller is responsible for: KUBECONFIG pointing at the cluster, the
# operator image being pushed to a registry the cluster can pull from,
# and any cluster cleanup. The suite will install CRDs, deploy the
# controller, run every spec, then undeploy + uninstall; the cluster
# is left intact.
.PHONY: test-e2e-cluster
test-e2e-cluster: _require-img manifests generate fmt vet ## Run smoke against the currently-configured cluster (no Kind).
	@echo "Running smoke suite against $$(kubectl config current-context) using IMG=$(IMG)"
	IMG=$(IMG) SMOKE_SKIP_IMAGE_LOAD=true \
		go test -tags=e2e ./test/e2e/ -v -ginkgo.v -timeout 30m
