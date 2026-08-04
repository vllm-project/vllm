# Top-level Makefile orchestrator. Targets and their recipes live in
# make/*.mk by concern (dev / e2e / e2e-gpu / lint / build / deploy /
# tools). Each sub-file carries its own `##@ Category` header so the
# `make help` awk command groups them correctly.

# Image URL to use all building/pushing image targets
IMG ?= controller:latest

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# CONTAINER_TOOL defines the container tool to be used for building images.
# Be aware that the target commands are only tested with Docker which is
# scaffolded by default. However, you might want to replace it to use other
# tools. (i.e. podman)
CONTAINER_TOOL ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. Categories come from `##@ <name>` lines, targets
# from `<name>: ... ## <description>`. $(MAKEFILE_LIST) expands to this
# Makefile + every included sub-file, so all sub-files are scanned.
.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# Order matters only when one file uses a macro `define`d in another:
#   - tools.mk defines `gomodver` + `go-install-tool` (used by every
#     tool-fetcher target it ships).
#   - dev/e2e/e2e-gpu/lint reference $(KIND), $(KUBECTL), etc. defined
#     in tools.mk — but those are simple variable references and Make
#     resolves them lazily, so include order doesn't matter for that.
# Listing tools.mk first is just stylistic so readers see the
# variables before the targets that consume them.
include make/tools.mk
include make/dev.mk
include make/unit.mk
include make/build.mk
include make/deploy.mk
include make/lint.mk
include make/e2e.mk
include make/e2e-gpu.mk
