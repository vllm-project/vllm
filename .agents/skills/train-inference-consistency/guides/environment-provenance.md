# Environment, image and runtime version provenance

Once the environment is set up (the image pins the native runtime, uv manages training dependencies, source arrives as a traceable snapshot or a minimal overlay), prove that it is what is actually running. Code existing does not mean it is executing.

## Find by symptom

| Symptom / question | Where to go |
| --- | --- |
| Not sure which version of source / wheel / .so / image is actually running | [Run manifest](#use-one-run-manifest-to-answer-what-exactly-is-it-right-now) |
| How many commits "latest" actually differs from the target branch | ["Latest" must come with a target branch](#latest-must-come-with-a-target-branch) |
| Overlay, or change the source and rebuild | [Boundary between overlay and native wheel](#boundary-between-overlay-and-native-wheel) |
| Recompiles on every start / not sure where the cache should go | [Compile cache](#compile-cache-where-it-lives-decides-whether-it-can-load) |
| Not sure which path DeepEP actually runs / how it was built | [Four things about the communication implementation](#communication-implementation-do-not-use-one-name-for-four-things) |

## Key rules

1. Pin the image digest, CPU architecture and Python version first, then resolve training dependencies. Image tag, PyPI version and upstream PR merge status do not replace one minimal inference run in the image.
2. `--system-site-packages` does not stop uv from installing a same-named package that then imports first; check the install plan, the actual import path and the GPU operators together.
3. Each of the six manifest layers (source, installed packages, native libraries, JIT kernels, container, run) has evidence nothing else replaces: `latest` is not the digest, `pip show` in another venv is not the actual import path, the Python source HEAD is not the loaded .so.
4. Audit inside the worker, interpreter and mount namespace that actually run; the login container shell and the Ray worker may differ.
5. One image tag may point to different digests on different dates/nodes; ARM64 and AMD64 tags may be independent builds.
6. Modify in the host worktree, execute in the container: same-commit worktree plus per-file hash check, never an overlay stack detached from the formal source.
7. Write version pins such as "the Megatron fork is only compatible with Megatron-Bridge v0.6.2" into the run manifest.
8. How the image was produced (container commit exported to sqsh, or Dockerfile rebuild), where it was pushed, and the three-repo SHAs in the tag are part of the run identity.

## Environment identity: code existing does not mean it is executing

### Use one run manifest to answer "what exactly is it right now"

Record recipe, trainer, Megatron/MCore, vLLM, DeepEP, DeepGEMM, FlashAttention/FlashInfer and NVSHMEM separately, each with at least:

| Layer | Evidence to save | What cannot substitute for it |
| --- | --- | --- |
| Source | remote, target branch, HEAD, dirty patch hash, submodule gitlink/actual HEAD | `latest`, the repo directory name |
| Installed package | Actual interpreter, import path, package version, wheel/build revision | `pip show` of another venv |
| Native library | Path/hash of the `.so` actually loaded, build inputs, target architecture, dynamic dependencies | Python source HEAD, extension file name |
| JIT kernel | DSL/source hash, compiler version, specialization/config, cache origin | `.so` unchanged |
| Container | OCI manifest/platform digest or `.sqsh` SHA256, parent image, build record | A mutable Docker tag, image creation time |
| Run | Resolved config, worker environment, model/weight digest, the real kernel | An export in the head node shell |

Write `unknown` for unknown items; do not infer the whole source tree from a wheel version string. Audit inside **the worker, interpreter and mount namespace that actually run**. If dynamic linking is suspect, check the process's actual mappings/loader result, not the static prediction of `ldd`.

ARM64 and AMD64 tags may be independent builds or members of one multi-arch manifest. Compare the target platform's manifest/config/layers; a similar tag or update time does not show it is the same image. Record generic scheduling switches such as `CUDA_DEVICE_MAX_CONNECTIONS=1` per the task convention; do not add them quietly and then credit BI.

A cross-repo compatibility pin is a manifest row too: state the pin (the Megatron fork works only with Megatron-Bridge v0.6.2) and its basis, and re-verify when either side is upgraded. Do not assemble incompatible releases just so every item reads "latest".

#### Communication implementation: do not use one name for four things

See [debugging mismatch](debugging-mismatch.md#do-not-use-one-name-for-four-things).

### "Latest" must come with a target branch

Pin each repo's target branch per the [delivery target](../practices/code-and-delivery.md#8-three-repo-delivery-from-experimental-implementation-to-three-prs-and-one-image). The reference implementation's PR is a design reference, not automatically the PR base. If the user names another fork/branch, record the override. fetch is not checkout; submodule update follows the parent repo's gitlink, not each submodule's own remote.

```bash
git -C /explicit/repo status --short --branch
git -C /explicit/repo remote -v
git -C /explicit/repo fetch origin
git -C /explicit/repo rev-list --left-right --count A...B
git -C /explicit/repo log --left-right --oneline A...B
git -C /explicit/repo merge-base A B
git -C /explicit/repo diff --stat A B
git -C /explicit/repo submodule status --recursive
```

The left/right counts of `A...B` say how many commits each side has; the merge base and diff say what differs. Diverged branches have no single before/after relation. Commit time and short-hash magnitude do not decide ancestry. If a shallow clone lacks the ancestor, fetch history before claiming the lines are unrelated.

Keep the recipe's declared compatible combination by default. If the user explicitly asks to update a development branch, record the deviation from the pin and check the actual changes and the ABI.

### Boundary between overlay and native wheel

During development, fix the container code-sync and build/test method first and record the tool versions in the run manifest. Modify in the host worktree, execute in the container; do not maintain an overlay stack detached from the formal source:

1. Confirm the config, model registration and import identity of the native image/checkpoint; pin the image digest, the full source commit and the real install directory. Do not guess a missing implementation from the model name or tag.
2. The sync worktree's `HEAD` equals the image's full commit, and the changes under test are a diff on that baseline. They may be ported from the development branch, but a different HEAD cannot bypass the check.
3. In a container not yet running the workload, sync each changed in-package Python file individually: before copying, check the original file's hash against `HEAD`; after copying, check the hash of the file under test; save the sync manifest. For new files, check no original exists on either side. Never overwrite the whole package or modify a running container.
4. Keep the image's version, generated files and unchanged extensions. Recompile native/binding changes and verify the artifacts; for JIT changes check compile inputs and cache; if the ABI/toolchain changes, rebuild the image.
5. After startup, re-check import/library paths in the actual worker and test. Finally build the unified image from the formal PR's clean commit and rerun acceptance; the user does not redo the development sync. Details: [Implementation and delivery](../practices/code-and-delivery.md#8-three-repo-delivery-from-experimental-implementation-to-three-prs-and-one-image).

Overlays are for development/diagnosis only, not the delivery format.

- An overlay covers specific files in the container with external copies; a wheel is an installed package that may carry compiled machine code.
- Cover only the necessary files at their exact target paths; do not mount the whole repo over the installed package and shadow its extensions. Verify the target's original hash first, apply once at startup, stop on failure, and save the overlay manifest, final hashes and import paths.
- A Python-only change can still change the binding/JIT/shape contract; it is not automatically exempt from recompiling or testing.
- Changes to CUDA/C++, bindings, compile macros, submodule kernels or the CUDA/Torch ABI need checking and recompiling. Before acting on a "missing extension", rule out the wrong venv, shadowing by a whole-repo mount, search paths and dynamic dependencies.
- If the image already has the correct artifact, verify it and use it. Do not recompile a DeepEP already proven to work, and do not change only the source pin in the recipe expecting the old `.so` to change.
- Do not fix a FlashAttention extension load failure by `unset`ting a variable. Accept a config fix only once it is shown the variable selected an unneeded/incompatible path and the run passes with the user's target path restored.
- Keep observability overlays separate from numerical/performance changes and off by default; when cleaning up the main PR, back up still-useful diagnostic code before deleting it.

### Delivery gate for immutable images

1. Commit production changes to their PRs; freeze the clean source commit and all native/JIT build inputs and dependencies; record the base image digest, not just the tag. A dirty patch may identify a diagnostic candidate but cannot be an uncommitted dependency of the final run.
2. Build the final image and check import/dependencies/hashes in a fresh container **with no development source mounts and no inherited JIT cache**, keeping the device, model and data mounts production needs.
3. Run the agreed correctness, Graph/cache and performance gates in that image; if observability was on, accept again with the formal config and no probes.
4. Publish exactly the accepted artifact and verify the registry digest/architecture; if conversion or repackaging changes the content, verify the equivalent closure and run a final smoke.
5. Sync recipe, PR description and image receipt; "image push succeeded" does not cover items that failed. Follow this order when the user asks to push after testing.

The receipt states how the image was produced: Dockerfile rebuild from a clean commit, or changes installed in a container then committed and exported as `.sqsh` (the only option when the coordinator has no docker permission; fine for development, but formal delivery returns to the Dockerfile); which registry it went to (Nemotron: public Docker Hub); a tag carrying the vLLM/Megatron/verl SHAs, not just the date. Record both the `.sqsh` SHA256 and the registry digest under the same tag.

Adding a new project source layer to an old OCI base does not bring in another experiment image's vLLM/FA4 changes. A deleted file missing from the top-level tar may still exist in lower layers; deleting a base file needs a correct whiteout or a check in the final rootfs.

Upgrading FlashInfer/NVSHMEM must not silently upgrade Torch/CUDA. Pin the target interpreter and constraints, compare packages/dynamic dependencies before and after, and confirm the APIs used and build compatibility. A version number is not evidence of an HCA fix or of model correctness.

## Compile cache: where it lives decides whether it can load

Put the Triton/Inductor/vLLM compile caches on node-local disk mounted explicitly into the container; a `noexec` mount (such as `/dev/shm` in some containers) cannot load the compiled `.so`, and a per-container tmpfs recompiles on every start. Isolate caches by image, source commit and GPU architecture, and give each rank on a node its own copy, since an existing cache directory does not mean a cache hit.
