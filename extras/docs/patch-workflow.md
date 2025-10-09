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

## 5. Handling secrets for local testing

- Real API credentials live in `extras/secrets/*.env`. Copy the provided `*.env.example`
  templates, fill in your personal tokens (for example, Hugging Face access keys), and keep them
  next to the examples with a `.env` suffix.
- The helpers now auto-discover every `extras/secrets/*.env` file (excluding `*.env.example`) and
  forward them to `podman run` through `--env-file`, so everything inside the dev container can use
  the same credentials without baking them into the image.
- All `extras/secrets/*.env` files are ignored by Git; only the example templates belong in the
  repository. Verify with `git status` before committing changes.

## 6. GPU passthrough on Windows + WSL2 Podman

1. Make sure your Windows host has the latest NVIDIA driver with WSL2 support and that `wsl --update`
   has run recently.
1. If you prefer to prepare the machine manually, download the
  [Rocky Linux 10 WSL Base image](https://dl.rockylinux.org/pub/rocky/10/images/x86_64/Rocky-10-WSL-Base.latest.x86_64.wsl)
  and run `podman machine init --image <local-file>`. The helper downloads and caches this archive automatically,
  so you can skip this step unless you want to supply a different build or an offline mirror.
    > **Heads-up:** Podman on Windows currently rejects `template://` URIs, so always point it at an actual
    > local file or an HTTP(S) download.
1. From the repository root, run the helper script:

  ```powershell
  pwsh extras/tools/enable-podman-wsl-gpu.ps1
  ```

   Add `-MachineName <name>` if you use a non-default Podman machine or `-SkipReboot` when you prefer
  to restart it manually later. Use `-ImagePath <file-or-url>` to override the default image; HTTP(S) URLs are
  downloaded into `%LOCALAPPDATA%\vllm-podman-images` on first use. Pass `-Rootful` to enable rootful
  mode automatically. Add `-Reset` to wipe and reinitialize the Podman machine (the helper removes the
  existing VM and re-runs `podman machine init`) when you want to start from a clean slate.

1. After the script restarts the machine, launch `extras/podman/run.ps1 -GPUCheck` (or `run.sh --gpu-check`)
   to confirm that `/dev/dxg` and the CUDA libraries are visible from inside the dev container. If the helper
   reports `Image missing. Use --build.`, rebuild the development container first via `extras/podman/run.ps1 --build`.

If the helper still reports missing `/dev/dxg`, open Podman Desktop, ensure GPU sharing is enabled for
the selected machine, and rerun the script (include `-Rootful` if you skipped it the first time, since
rootless containers cannot mount GPU device nodes). When running on other distributions, replicate the
script’s steps manually: install `nvidia-container-toolkit`, and generate a CDI spec via
`nvidia-ctk cdi generate --mode wsl`.
