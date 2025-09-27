# Fork extras helpers

This directory keeps helper scripts that are specific to the `Zhuul/vllm` fork.

- `sync_with_upstream.sh`: fetches the latest upstream changes and merges them
  into the current branch while backing up and restoring fork-specific paths
  (`extras/`, `.github/extras`, `.github/workflows/fork-sync.yml` by default)
  so local patches and automation remain intact. Override the protected list by
  exporting `PROTECTED_PATHS="path1 path2"` before running the script.
- `sync_with_upstream.ps1`: Windows helper that locates Git Bash and executes
  `sync_with_upstream.sh` without launching an editor. Usage:

  ```powershell
  pwsh -File .github/extras/sync_with_upstream.ps1
  # or
  .\.github\extras\sync_with_upstream.ps1 -UpstreamRemote upstream -UpstreamBranch main
  ```

  On macOS/Linux invoke the shell script directly: `./.github/extras/sync_with_upstream.sh`.

Running the sync script requires a clean working tree. It will abort if a merge
or rebase is in progress; finish or cancel those operations first.

Run scripts from the repository root. Adjust the remote/branch by passing
arguments, e.g. `./.github/extras/sync_with_upstream.sh upstream main`.
