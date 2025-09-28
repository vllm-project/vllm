# Fork sync tooling

Scripts in this directory keep the `Zhuul/vllm` fork aligned with the upstream
`vllm-project/vllm` repository while preserving fork-specific customisations in
`extras/` and the GitHub Actions workflow.

- `sync_with_upstream.sh` – bash helper that performs the upstream fetch,
  fast-forward (with merge fallback), and protected path backup/restore.
- `sync_with_upstream.ps1` – PowerShell wrapper that locates Git Bash on
  Windows and executes the shell script with the same arguments.

## Usage

From the repository root run one of the following:

```bash
./extras/tools/fork-sync/sync_with_upstream.sh
```

```powershell
pwsh -File extras/tools/fork-sync/sync_with_upstream.ps1
```

Both scripts accept optional parameters for the remote and branch. For example:

```bash
./extras/tools/fork-sync/sync_with_upstream.sh upstream main
```

To override the protected paths, set the `PROTECTED_PATHS` environment variable
with a space-separated list before running the script.

Ensure your working tree is clean and no merge/rebase is in progress prior to
invoking the tooling.
