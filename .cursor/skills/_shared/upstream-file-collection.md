# Upstream File Collection

Shared workflow for determining the upstream reference point, collecting
modified files, and classifying them. Used by `check-cohere-markers`,
`minimize-upstream-diff`, and `rebase-assistant`.

## Two Remotes

This repo has two remotes:

| Remote | Repo | Purpose |
|--------|------|---------|
| `origin` | Your fork / custom branch | Where you push your work |
| `upstream` | `git@github.com:vllm-project/vllm.git` | The canonical vllm project |

Whether a file is in scope is always determined by checking if it exists in
**upstream**, not `origin`.

## Step A: Ensure Upstream Remote

```bash
if ! git remote get-url upstream &>/dev/null; then
  git remote add upstream git@github.com:vllm-project/vllm.git
fi
git fetch upstream --tags
```

## Step B: Determine and Confirm Upstream Ref

Suggest a candidate `UPSTREAM_REF` using the first successful method:

1. Explicit ref given by the user or a calling skill (e.g., `v0.15.1`).
2. Most recent upstream tag reachable from the merge-base:
   ```bash
   MERGE_BASE=$(git merge-base HEAD upstream/main)
   git describe --tags --abbrev=0 --match="v*" "$MERGE_BASE" 2>/dev/null
   ```
3. Merge-base SHA itself: `git merge-base HEAD upstream/main`

**Always confirm with the user before proceeding.** Present the candidate and
ask the user to confirm or override:

> The candidate upstream ref is **v0.15.1** (most recent tag reachable from
> merge-base). Is this correct, or would you like to use a different tag or
> commit?

Wait for explicit confirmation. Store the confirmed value as `UPSTREAM_REF`.

## Step C: Collect and Classify Changed Files

Use a **content-difference approach**: compare the current working tree
directly against `UPSTREAM_REF`, without relying on commit history.

**C1. Get all files that differ:**

```bash
git diff --name-status "$UPSTREAM_REF"
```

**C2. For each file, check if it exists in upstream:**

```bash
git cat-file -e "$UPSTREAM_REF:<file_path>" 2>/dev/null
```

**C3. Classify:**

| Exists in upstream? | Change type | In scope? |
|---------------------|-------------|-----------|
| YES | Modified (M) | YES — modification to an upstream file |
| NO | Added (A) | NO — new fork-only file |
| YES | Deleted (D) | NO — deletion is intentional |
| YES | Renamed (R) | CHECK — in scope if content also changed |

**The key rule: a file is in scope if and only if it exists at
`UPSTREAM_REF` and has content changes.**

To identify entire new directories (fork-only additions), check whether the
directory exists at `UPSTREAM_REF`:

```bash
git ls-tree --name-only "$UPSTREAM_REF" -- path/to/directory/
```

If it returns nothing, the entire directory is new and no files inside are in
scope.

**C4. Output:** the list of in-scope files (modified upstream files) and a
count. Optionally display the skipped files (added, deleted) for
transparency.
