# Patch workflow for vLLM development

This document summarizes the intended workflow for keeping the fork in sync, preparing the
container environment, and layering patches for experimentation. It covers both the automation
running on GitHub and the local steps you take before hacking on the codebase.

## 1. Keep the fork aligned with upstream

- The scheduled workflow `.github/workflows/fork-sync.yml` copies the upstream repository
  (`vllm-project/vllm`) into the fork (`Zhuul/vllm`) every day.
- Files that only exist in the fork—`extras/`, the fork-sync workflow itself, and other
  additive automation—are marked as *protected*. The sync script (`extras/tools/fork-sync/`
  `sync_with_upstream.{sh,ps1}`) backs them up before the merge and restores them afterward so
  an upstream update never removes local tooling.
- When developing locally, run the same helper script to merge upstream changes. That guarantees
  the protected paths stay identical between CI and your machine.

## 2. Container setup and pre-build overlays

- Launching `extras/podman/run.ps1 -Setup` (or `run.sh` on Linux) builds the dev container if
  needed and starts it with the entrypoint `extras/podman/entrypoint/apply-patches-then-exec.sh`.
- The entrypoint normalizes Windows line endings, configures `git` to trust the mounted
  workspace, and invokes `extras/patches/apply_patches_overlay.sh` before any build commands run.
- Overlay definitions live in `extras/patches/python-overrides.txt`. Each line copies a file from
  the repository into an overlay directory inside the container (defaults to
  `/opt/work/python-overrides`) and can apply transforms (for example, adapting
  `vllm/device_allocator/cumem.py` for CUDA 13).
- Overlay mode must leave the repository clean. The helper now fails early if a patch no longer
  applies or if any tracked file stays modified, signalling that the overlay definitions need an
  update instead of silently mutating the tree.

## 3. Editable install

- `extras/podman/dev-setup.sh` performs the editable install. It exports
  `SETUPTOOLS_SCM_ROOT=/workspace` and `SETUPTOOLS_SCM_IGNORE_VCS_ERRORS=1` so
  `setuptools_scm` resolves the version without probing the temporary build directory.
- After publishing overlays it re-checks `git status`. If any tracked file is dirty, the script
  stops immediately, ensuring the pre-build stage remains reproducible.

## 4. Post-build experimentation

- Once the setup script finishes, drop into an interactive shell with `run.ps1 -Interactive`.
  Use this phase for manual edits, runtime experiments, or additional scripted patches.
- Future customizations can be organized under `extras/patches/post-setup.d/` (or a similar
  directory) and invoked from the interactive helper so that experimental work is clearly
  separated from the deterministic pre-build overlay stage.

By enforcing clean working trees after each automated step, the workflow mirrors what CI expects
and keeps the Windows-mounted repository free of unexpected modifications.
