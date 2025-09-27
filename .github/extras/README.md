# Fork extras helpers

This directory keeps helper scripts that are specific to the `Zhuul/vllm` fork.

- `sync_with_upstream.sh`: fetches the latest upstream changes and merges them
  into the current branch while backing up and restoring fork-specific paths
  (`extras/`, `.github/extras`, `.github/workflows/fork-sync.yml` by default)
  so local patches and automation remain intact. Override the protected list by
  exporting `PROTECTED_PATHS="path1 path2"` before running the script.

Run scripts from the repository root. Adjust the remote/branch by passing
arguments, e.g. `./.github/extras/sync_with_upstream.sh upstream main`.
