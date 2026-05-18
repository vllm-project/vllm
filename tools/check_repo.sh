#!/bin/bash
# Checks whether the repo is clean and whether tags are available (necessary to correctly produce vllm version at build time)

if ! git diff --quiet; then
	echo "Repo is dirty" >&2

	exit 1
fi

if ! git describe --tags; then
	echo "No tags are present. Is this a shallow clone? git fetch --unshallow --tags" >&2

	exit 1
fi
