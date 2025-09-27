# Fork extras helpers

This directory keeps helper scripts that are specific to the `Zhuul/vllm` fork.

- `sync_with_upstream.sh`: fetches the latest upstream changes and merges them
  into the current branch while backing up and restoring the `extras/` directory
  so local patches remain intact.

Run scripts from the repository root. Adjust the remote/branch by passing
arguments, e.g. `./.github/extras/sync_with_upstream.sh upstream main`.
