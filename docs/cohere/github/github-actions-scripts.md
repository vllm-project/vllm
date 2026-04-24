# GitHub Actions Helper Scripts

These helper scripts live in `.github/scripts/` and are used by GitHub Actions workflows.
This page primarily documents the upstream-sync/rerere flow in detail, and also lists other helper scripts used by CI.

## Automated Upstream Sync Workflow

The complete automated workflow for keeping the fork in sync with upstream vLLM:

```text
┌─────────────┐
│   cohere    │  ← Your main development branch
└──────┬──────┘
       │ (push)
       ↓
  ┌────────────────────┐
  │ GitHub Actions:    │
  │ auto-squash-branch │  ← Squashes all commits
  └────────┬───────────┘
           ↓
  ┌────────────────┐
  │ cohere-squashed│  ← Single squashed commit
  └────────┬───────┘
           │ (weekly or manual)
           ↓
  ┌─────────────────────┐
  │ GitHub Actions:     │
  │ auto-rebase-upstream│  ← Rebases with rerere
  └────────┬────────────┘
           ↓
    ┌─────────────────────┐
    │  Success?           │
    └──────┬──────┬───────┘
           │      │
     Yes   │      │ No
           ↓      ↓
  ┌───────────┐  ┌──────────────────────┐
  │  cohere-  │  │ cohere-synced-       │
  │  synced   │  │ conflicts + issue    │
  └─────┬─────┘  └──────────────────────┘
        │                   │
        │ (triggers)        ↓
        ↓          ┌─────────────────────┐
  ┌────────────┐   │ Manual resolution   │
  │  nightly-  │   │ (trains rerere)     │
  └────────────┘   └─────────────────────┘
```

## Scripts

### General workflow helpers

- `cleanup_pr_body.sh` - trims generated PR body sections in automation flows.
- `extract-wheels-from-registry.sh` - pulls wheel artifacts from registry layers.
- `upload-wheels-dir-to-gar.sh` - uploads wheel directories to GAR Python repos.

### resolve-rebase-conflicts.sh

Helper script for manually resolving upstream rebase conflicts and training git rerere. **Automatically downloads existing cache first** so you don't re-resolve conflicts.

### download-rerere-cache.sh

Downloads existing rerere cache from GitHub to your local machine. **Always run this before resolving conflicts** to avoid re-resolving already-handled conflicts.

**Usage:**

```bash
.github/scripts/download-rerere-cache.sh
```

Note: The `resolve-rebase-conflicts.sh` script automatically runs this for you.

### upload-rerere-cache.sh

Uploads your local rerere cache to GitHub so that GitHub Actions can use your conflict resolutions.

**Usage:**

```bash
.github/scripts/upload-rerere-cache.sh
```

Run this after you've resolved conflicts locally to sync your resolutions to GitHub Actions.

---

## resolve-rebase-conflicts.sh (detailed)

### Usage

```bash
# Basic usage (rebase onto vllm-upstream/main)
.github/scripts/resolve-rebase-conflicts.sh

# Rebase onto specific upstream ref
.github/scripts/resolve-rebase-conflicts.sh vllm-upstream/v0.6.0

# Use custom target branch
.github/scripts/resolve-rebase-conflicts.sh vllm-upstream/main my-custom-sync
```

### When to use this script

1. **First-time conflict resolution**: When the auto-rebase workflow encounters new conflicts
2. **Training rerere**: To teach rerere how to resolve conflicts automatically in the future
3. **Testing locally**: Before triggering the GitHub Actions workflow

### How it works

1. **Downloads existing rerere cache** from GitHub (if available)
2. Enables git rerere (reuse recorded resolution)
3. Fetches latest upstream vLLM and cohere-squashed
4. Creates a new branch and starts rebase
5. If conflicts occur, pauses for you to resolve them
6. As you resolve and continue, rerere records your resolutions
7. Future rebases will automatically apply the same resolutions

### Example workflow

```bash
# 1. Run the script
.github/scripts/resolve-rebase-conflicts.sh

# 2. If conflicts occur, resolve them in your editor
vim path/to/conflicted/file.py

# 3. Stage resolved files
git add path/to/conflicted/file.py

# 4. Continue rebase
git rebase --continue

# 5. Repeat steps 2-4 until rebase completes

# 6. Push to GitHub
git push -f origin cohere-synced

# 7. Upload rerere cache so GitHub Actions can use it
.github/scripts/upload-rerere-cache.sh
```

### Rerere cache and synchronization

**Important**: Local and GitHub Actions rerere caches are separate!

After you resolve conflicts locally:

```bash
# Upload your resolutions to GitHub
.github/scripts/upload-rerere-cache.sh
```

This creates a `rerere-cache-upload` branch with your resolutions. The GitHub Actions workflow will automatically:

- Download this branch
- Merge it with its cache
- Use your resolutions in future rebases

**How it works:**

1. **Download first**: `download-rerere-cache.sh` fetches existing resolutions from `rerere-cache-upload` branch
2. **Local resolution**: Rerere saves to `.git/rr-cache/` on your machine (merges with downloaded cache)
3. **Upload**: `upload-rerere-cache.sh` pushes your cache to `rerere-cache-upload` branch
4. **GitHub Actions**: Downloads and merges this cache before rebasing
5. **Future rebases**: Automatically apply all resolutions!

### Troubleshooting

#### Cannot rebase: You have unstaged changes

```bash
git stash
.github/scripts/resolve-rebase-conflicts.sh
git stash pop
```

#### Want to abort the rebase

```bash
git rebase --abort
```

#### Rerere applied wrong resolution

```bash
# Clear rerere cache for a specific conflict
git rerere forget path/to/file.py

# Or clear entire rerere cache
rm -rf .git/rr-cache
```
