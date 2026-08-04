##@ Lint

.PHONY: install-hooks
install-hooks: golangci-lint ## Install git pre-commit hooks (requires pre-commit: pip install pre-commit).
	@command -v pre-commit >/dev/null 2>&1 || { \
		echo "pre-commit not found. Install with: pip install pre-commit"; \
		exit 1; \
	}
	cd "$(shell git rev-parse --show-toplevel)" && pre-commit install -c operator/.pre-commit-config.yaml
	@echo "Pre-commit hooks installed (operator)."

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter
	"$(GOLANGCI_LINT)" run

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes
	"$(GOLANGCI_LINT)" run --fix

.PHONY: lint-config
lint-config: golangci-lint ## Verify golangci-lint linter configuration
	"$(GOLANGCI_LINT)" config verify
