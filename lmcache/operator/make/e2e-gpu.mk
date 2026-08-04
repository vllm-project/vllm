##@ E2E (GPU)

# Creates a 2-node Kind cluster (control-plane + GPU worker) and
# installs the NVIDIA GPU Operator. The Operator handles toolkit
# install + containerd reconfig inside the worker so pods with
# `runtimeClassName: nvidia` get NVML / libcuda injected.
#
# Full design notes (why GPU Operator vs bare device plugin, host
# prereqs, single-node intent) live in operator/AGENTS.md under the
# "GPU tier (M2)" section. Keep this header lean.
ifndef GPU_KIND_CLUSTER
GPU_KIND_CLUSTER := operator-test-e2e-gpu-$(shell date +%s%N | tail -c 9)
endif
HELM ?= helm

# GPU Operator helm flags worth a note (the rest are defaults):
#   driver.enabled=false   — host docker's nvidia runtime already
#                            injects /dev/nvidia* + driver libs into
#                            the Kind worker; we don't want a kernel
#                            driver installed inside the cluster.
#   toolkit.enabled=true   — toolkit daemonset writes containerd's
#                            config.toml + registers a `nvidia`
#                            runtime handler. Pods with
#                            runtimeClassName=nvidia then get NVML.
#   cdi.enabled=true       — modern CDI annotation path; needs
#                            containerd 1.7+ (kindest/node has it).
#   CONTAINERD_CONFIG /    — Kind's containerd config + socket are at
#   CONTAINERD_SOCKET        non-default paths; without overriding,
#                            the toolkit daemonset edits a phantom
#                            config and ClusterPolicy never goes Ready.
#   validator.driver.env   — DISABLE_DEV_CHAR_SYMLINK_CREATION; Kind
#                            nodes can't mknod /dev/char/* and the
#                            driver validator pod would loop without
#                            this. Documented gpu-operator-on-kind
#                            workaround.
.PHONY: setup-test-e2e-gpu-kind
setup-test-e2e-gpu-kind: ## Create a Kind GPU cluster (inline config) + install the NVIDIA GPU Operator.
	@command -v $(KIND)    >/dev/null 2>&1 || { echo "kind not found. Install Kind."; exit 1; }
	@command -v $(HELM)    >/dev/null 2>&1 || { echo "helm not found. Install helm v3."; exit 1; }
	@command -v $(KUBECTL) >/dev/null 2>&1 || { echo "kubectl not found."; exit 1; }
	@docker info 2>/dev/null | grep -q "Default Runtime: nvidia" || { \
		echo "ERROR: Docker default-runtime is not 'nvidia'."; \
		echo "  Fix: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled && sudo systemctl restart docker"; \
		exit 1; \
	}
	@grep -E '^\s*accept-nvidia-visible-devices-as-volume-mounts\s*=\s*true' /etc/nvidia-container-runtime/config.toml >/dev/null 2>&1 || { \
		echo "ERROR: 'accept-nvidia-visible-devices-as-volume-mounts' is not true in /etc/nvidia-container-runtime/config.toml."; \
		echo "  Fix: sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place && sudo systemctl restart docker"; \
		echo "  Without this, the volume-mount marker won't propagate into the Kind worker."; \
		exit 1; \
	}
	@echo "==> Creating Kind GPU cluster '$(GPU_KIND_CLUSTER)' (1 control-plane + 1 worker, all GPUs)..."
	@printf '%s\n' \
		'kind: Cluster' \
		'apiVersion: kind.x-k8s.io/v1alpha4' \
		'name: $(GPU_KIND_CLUSTER)' \
		'nodes:' \
		'- role: control-plane' \
		'- role: worker' \
		'  labels:' \
		'    nvidia.com/gpu.present: "true"' \
		'  extraMounts:' \
		'    - hostPath: /dev/null' \
		'      containerPath: /var/run/nvidia-container-devices/all' \
		| $(KIND) create cluster --config=-
	@echo "==> Installing the NVIDIA GPU Operator (this takes 5-10 minutes)..."
	$(HELM) repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
	$(HELM) repo update nvidia >/dev/null
	@$(HELM) upgrade -i gpu-operator nvidia/gpu-operator \
		--kube-context=kind-$(GPU_KIND_CLUSTER) \
		--namespace gpu-operator --create-namespace \
		--set driver.enabled=false \
		--set toolkit.enabled=true \
		--set cdi.enabled=true \
		--set toolkit.env[0].name=CONTAINERD_CONFIG \
		--set-string toolkit.env[0].value=/etc/containerd/config.toml \
		--set toolkit.env[1].name=CONTAINERD_SOCKET \
		--set-string toolkit.env[1].value=/run/containerd/containerd.sock \
		--set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
		--set-string toolkit.env[2].value=nvidia \
		--set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
		--set-string toolkit.env[3].value=true \
		--set validator.driver.env[0].name=DISABLE_DEV_CHAR_SYMLINK_CREATION \
		--set-string validator.driver.env[0].value=true \
		--wait --timeout=15m \
		|| { $(MAKE) --no-print-directory diagnose-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER); exit 1; }
	@echo "==> Waiting for the node to advertise nvidia.com/gpu (up to 5 min)..."
	@gpus=""; \
	for i in $$(seq 1 150); do \
		gpus=$$($(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) get nodes \
			-o jsonpath='{.items[?(@.metadata.labels.nvidia\.com/gpu\.present=="true")].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true); \
		case "$$gpus" in [1-9]*) echo "==> Node advertises $$gpus GPU(s)"; exit 0 ;; esac; \
		sleep 2; \
	done; \
	echo ""; \
	echo "ERROR: node never advertised nvidia.com/gpu after 5 min."; \
	$(MAKE) --no-print-directory diagnose-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER); \
	exit 1

.PHONY: diagnose-test-e2e-gpu-kind
diagnose-test-e2e-gpu-kind: ## Dump gpu-operator state for the named cluster (for setup failures).
	@echo ""
	@echo "=========================================================================="
	@echo "GPU Operator diagnostic dump for cluster: $(GPU_KIND_CLUSTER)"
	@echo "=========================================================================="
	@echo ""
	@echo "--- kubectl get nodes (allocatable / capacity) ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\n  allocatable: "}{.status.allocatable}{"\n  capacity: "}{.status.capacity}{"\n"}{end}' || true
	@echo ""
	@echo "--- ClusterPolicy status ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) get clusterpolicy -o jsonpath='{.items[0].status}' 2>/dev/null | head -c 2000 || echo "(no ClusterPolicy yet)"
	@echo ""
	@echo ""
	@echo "--- gpu-operator pods ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) get pods -n gpu-operator -o wide || true
	@echo ""
	@echo "--- gpu-operator events (last 40) ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) get events -n gpu-operator --sort-by=.lastTimestamp 2>/dev/null | tail -40 || true
	@echo ""
	@echo "--- toolkit daemonset logs ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) logs -n gpu-operator -l app=nvidia-container-toolkit-daemonset --tail=60 --all-containers 2>&1 | head -120 || true
	@echo ""
	@echo "--- device-plugin daemonset logs ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) logs -n gpu-operator -l app=nvidia-device-plugin-daemonset --tail=60 --all-containers 2>&1 | head -120 || true
	@echo ""
	@echo "--- driver-validator pod logs (if running) ---"
	@$(KUBECTL) --context=kind-$(GPU_KIND_CLUSTER) logs -n gpu-operator -l app=nvidia-operator-validator --tail=60 --all-containers 2>&1 | head -120 || true
	@echo ""
	@echo "--- /dev/nvidia* inside the Kind worker container ---"
	@(docker exec $(GPU_KIND_CLUSTER)-worker  sh -c 'ls /dev/nvidia* 2>&1' || \
	  docker exec $(GPU_KIND_CLUSTER)-control-plane sh -c 'ls /dev/nvidia* 2>&1' || \
	  echo "(no nvidia devices visible)") | head -30 || true
	@echo ""
	@echo "--- /etc/containerd/config.toml nvidia stanza (inside Kind worker) ---"
	@docker exec $(GPU_KIND_CLUSTER)-worker sh -c 'grep -A8 "nvidia" /etc/containerd/config.toml 2>&1 || echo "(no nvidia stanza)"' || true
	@echo "=========================================================================="

.PHONY: test-e2e-gpu-kind
test-e2e-gpu-kind: manifests generate fmt vet ## Self-contained GPU smoke on a Kind cluster (build + load + run + cleanup).
	@echo "==> Using Kind GPU cluster: $(GPU_KIND_CLUSTER) (auto-deleted after the run unless KEEP_CLUSTER_ON_FAILURE=1)"
	@trap 'rc=$$?; \
		if [ "$$rc" -ne 0 ] && [ "$(KEEP_CLUSTER_ON_FAILURE)" = "1" ]; then \
			echo ""; \
			echo "==> KEEP_CLUSTER_ON_FAILURE=1 set; leaving cluster '$(GPU_KIND_CLUSTER)' alive."; \
			echo "    To diagnose:  make diagnose-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER)"; \
			echo "    To delete:    make cleanup-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER)"; \
		else \
			$(MAKE) --no-print-directory cleanup-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER); \
		fi; \
		exit $$rc' EXIT INT TERM; \
	 $(MAKE) --no-print-directory setup-test-e2e-gpu-kind GPU_KIND_CLUSTER=$(GPU_KIND_CLUSTER) && \
	 KIND=$(KIND) KIND_CLUSTER=$(GPU_KIND_CLUSTER) \
		go test -tags=e2e,e2e_gpu ./test/e2e/ -v -ginkgo.v -timeout 60m

.PHONY: cleanup-test-e2e-gpu-kind
cleanup-test-e2e-gpu-kind: ## Delete the Kind GPU cluster (idempotent).
	@if $(KIND) get clusters 2>/dev/null | grep -qx "$(GPU_KIND_CLUSTER)"; then \
		echo "Deleting Kind GPU cluster '$(GPU_KIND_CLUSTER)'..."; \
		$(KIND) delete cluster --name $(GPU_KIND_CLUSTER); \
	else \
		echo "Kind GPU cluster '$(GPU_KIND_CLUSTER)' does not exist; nothing to delete."; \
	fi

# Run the GPU smoke tier against an already-configured GPU cluster.
# Runs the no-GPU specs PLUS the e2e_gpu specs. Caller is responsible
# for: a GPU node labeled nvidia.com/gpu.present=true with the nvidia
# RuntimeClass; KUBECONFIG pointing at it; pushing the operator image
# (pass IMG=); and any cluster cleanup. Knobs: VLLM_MODEL,
# VLLM_IMAGE, SKIP_VLLM_INTEGRATION (see vllm_integration_smoke_test.go).
# 60m timeout absorbs cold pulls of the ~10GB vllm-openai image.
.PHONY: test-e2e-gpu-cluster
test-e2e-gpu-cluster: _require-img manifests generate fmt vet ## Run GPU smoke against an existing GPU cluster.
	@echo "Running GPU smoke against $$(kubectl config current-context) using IMG=$(IMG)"
	IMG=$(IMG) SMOKE_SKIP_IMAGE_LOAD=true \
		go test -tags=e2e,e2e_gpu ./test/e2e/ -v -ginkgo.v -timeout 60m
