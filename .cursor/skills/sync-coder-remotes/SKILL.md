---
name: sync-coder-remotes
description: Set up and use remote Coder workspace access for command execution and result checks, including syncing the remote repo to an exact commit SHA when repo state matters. Use when the user wants to run a command on another Coder VM/workspace, inspect remote output, verify auth, or connect to a shared workspace.
---

# Sync Coder Remotes

Use this skill when the user wants help reaching another Coder workspace from
the current one, especially to run a command remotely and inspect the result.

Default to the `coder` CLI access method.

## What To Gather

Before running anything remotely, collect or confirm:

1. Target workspace name.
2. Target owner, if not obvious.
3. Command to run.
4. What result to check: exit code, stdout/stderr, generated files, service
   health, benchmark values, or another artifact.

Useful optional details:

- Working directory on the remote workspace.
- Remote repo root, if the command depends on repository contents.
- Exact commit SHA to sync, if the command should run against a specific repo
  state.
- Whether the command needs multiline shell logic; if so, prefer piping a
  script to `bash -s` instead of nesting heavy quoting inside `bash -lc`.
- Required env vars.
- Whether the command is read-only or may modify state.
- Whether the command is long-running.

If key details are missing, ask targeted follow-up questions instead of
guessing.

## Default Access Method

Use `coder` by default.

Only switch to SSH or another method if:

- the user explicitly requests it, or
- the `coder` path is unavailable and the user provides another supported
  access method.

## Workflow

### 1. Check local prerequisites

First verify:

1. `coder` CLI is installed with `command -v coder`.
2. The current session is authenticated with `coder whoami`.

If `coder` is missing, stop and tell the user how to install it.

If auth is missing, do not proceed to remote execution yet.

### 2. Handle auth

If `coder whoami` fails:

1. Prompt the user for their Coder base URL if they have not already provided
   it.
2. Instruct them to log in with:

```bash
coder login https://<their-coder-host>
```

3. After they confirm login is complete, re-run `coder whoami`.

If the user asks how to find the base URL, explain that it is the scheme and
host from their normal Coder web UI, for example
`https://coder.example.com`.

### 3. Verify the target workspace

Confirm the workspace is visible before trying to run commands:

```bash
coder list --search "owner:<owner> name:<workspace>"
```

If the workspace is not visible, ask the user to confirm:

- the exact workspace name,
- the owner, and
- whether the workspace has been shared with the current identity when needed.

### 4. Sync the remote repo to the exact commit when needed

If the remote command depends on repository contents, prefer syncing the remote
repo to the exact commit SHA rather than assuming the remote branch is current.

1. Confirm the remote `repo_root` and target `commit_sha`.
2. If the SHA should match the local workspace, get it with:

```bash
git rev-parse HEAD
```

3. Ensure the remote can fetch that SHA.
4. Prefer a clean detached worktree instead of modifying the main remote
   checkout in place.
5. Verify the synced path resolves to the expected SHA with `git rev-parse
   HEAD`.

Example:

```bash
cat <<'EOF' | coder ssh <owner>/<workspace> -- bash -s
set -e
REPO_ROOT=/root/repos/vllm-cohere
SYNC_ROOT=/root/repos/vllm-cohere-sync/<commit_sha>
COMMIT_SHA=<commit_sha>

test -d "$REPO_ROOT/.git"
mkdir -p "$(dirname "$SYNC_ROOT")"
git -C "$REPO_ROOT" fetch --all --tags
git -C "$REPO_ROOT" worktree remove --force "$SYNC_ROOT" >/dev/null 2>&1 || true
rm -rf "$SYNC_ROOT"
git -C "$REPO_ROOT" worktree add --detach "$SYNC_ROOT" "$COMMIT_SHA"
git -C "$SYNC_ROOT" rev-parse HEAD
EOF
```

If the SHA is not fetchable on the remote, stop and ask the user to push the
commit or choose a snapshot-based sync instead of guessing.

### 5. Run the remote command

Use non-interactive `coder ssh` by default.

Prefer one of these two patterns:

1. Short single-line probes:

```bash
coder ssh <owner>/<workspace> -- <command>
coder ssh <owner>/<workspace> -- bash -lc '<short command>'
```

2. Complex, multiline, or env-heavy commands:

```bash
cat <<'EOF' | coder ssh <owner>/<workspace> -- env HF_TOKEN="$HF_TOKEN" bash -s
set -euo pipefail
cd /path/on/remote
<command>
EOF
```

Why this matters:

- `bash -lc` is fine for small probes.
- For long scripts, nested quotes become fragile and secrets are easier to
  mishandle.
- `env VAR=value bash -s` is the safest default when you must inject env vars
  like `HF_TOKEN` into a remote script.

Examples:

```bash
coder ssh dongluw/donglu-dev-8xmi300 -- df
coder ssh dongluw/donglu-dev-8xmi300 -- bash -lc 'cd /root && df -h'
cat <<'EOF' | coder ssh dongluw/donglu-dev-8xmi300 -- bash -s
set -e
cd /root
df -h
EOF
```

When the command depends on the synced repo state, run it from the synced path
rather than the base checkout.

### 6. Report results

Summarize:

1. Whether the command succeeded.
2. The important output, not just raw logs.
3. Any follow-up observations the user asked for.

If the user asked for command output, relay the meaningful lines in the
response instead of saying only that it was run.

## Prompting Style

When inputs are missing, ask for the minimum needed to proceed:

- workspace name
- owner
- access method if not `coder`
- command
- what to verify

Good defaults:

- access method: `coder`
- remote form: `coder ssh <owner>/<workspace> -- <command>`

## Safety Notes

- Avoid modifying the remote workspace unless the user asked for it.
- Prefer read-only sanity checks first when the request is ambiguous.
- Do not assume authentication, visibility, or workspace sharing is already set
  up.
- If a command may run for a long time, tell the user before starting and
  clarify whether they want a one-shot result or ongoing monitoring.

## Example Triggers

Use this skill for requests like:

```text
Run `df` on another Coder VM
```

```text
Can you check results from a command on my other workspace?
```

```text
Help me connect to a shared Coder workspace and inspect it
```
