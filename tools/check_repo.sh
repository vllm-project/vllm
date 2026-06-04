#!/bin/bash
# Checks whether the repo is clean and whether tags are available (necessary to correctly produce vllm version at build time)

# Some Docker builds intentionally omit tracked, non-build files from the
# context. Restore only those paths from the mounted .git object database before
# checking cleanliness so release builds still see a coherent worktree.
if [ -f /.dockerenv ]; then
	git ls-files -z -- docs .github .pre-commit-config.yaml format.sh \
		| git checkout-index -f -z --stdin
fi

if ! git diff --quiet; then
	echo "Repo is dirty" >&2

	exit 1
fi

if ! git describe --tags; then
	echo "No tags are present. Is this a shallow clone? git fetch --unshallow --tags" >&2

	exit 1
fi
