---
name: startup-cache-image
description: Prepare and verify a vLLM container image with prewarmed compilation caches. Use when baking caches into an image or moving a warmed cache to another deployment.
---

# Build an image with a warm compile cache

Use vLLM's existing cache support. For repeated starts on one host, a
[persistent volume](../../../docs/deployment/docker.md#persist-the-compile-cache-across-containers)
may be enough. Bake a cache into an image when it needs to travel with the image.

## Pin the serving configuration

Start from the user's working serve command or the
[model-recipes skill](../model-recipes/SKILL.md). Record the base image digest,
model revision, GPU model, parallelism, serving arguments and environment.
Keep these fixed across generation and verification. Changing optimization
levels or using eager mode would measure a different serving configuration.
If no working recipe is available, establish the uncached serving baseline
first; label an inferred command as unverified until it runs successfully.

Read [Faster Startup](../../../docs/configuration/optimization.md#faster-startup)
and [Compilation Cache](../../../docs/design/torch_compile.md#compilation-cache)
for supported behavior and invalidation. Check the selected image's version;
checkout documentation may describe features that image does not contain.

## Generate the cache

Run the pinned server on the target GPU model with a new, private host directory
bind-mounted at its `VLLM_CACHE_ROOT`. Use the same absolute cache path and
container user in the final image. Adapt the existing Docker serve command's
cache mount; retain its model, networking, shared-memory and recipe settings.

Wait for readiness, run a representative inference check, save its result and
startup log, then stop the server cleanly before copying the directory. Warming
requires a GPU runtime; an ordinary `docker build` does not supply one.
Generation and image building can run on different hosts. Transfer the cache
with its permissions and timestamps preserved, for example with `cp -a` or tar.

Copy the complete vLLM compile-cache tree, including AOT artifacts if enabled.
If `TRITON_CACHE_DIR` is explicitly set outside that tree, carry that directory
too, at the same path. Inventory backend-specific caches separately; do not
assume a vLLM cache hit covers FlashInfer autotuning or every kernel cache.

## Bake the image

Create a minimal build context containing the cache and a Dockerfile. Keep
weights, credentials and unrelated home-directory contents out of it. Caches
contain executable artifacts: follow the
[cache security guidance](../../../docs/usage/security.md).

For the official image's default root user, with the host cache copied to
`cache/` in the build context:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV VLLM_CACHE_ROOT=/root/.cache/vllm
COPY cache/ /root/.cache/vllm/
```

```bash
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t vllm-warm:local .
```

Set `BASE_IMAGE` to the same digest used for generation. For a non-root image,
use its writable cache path and ownership as described in
[Run as a non-root user](../../../docs/deployment/docker.md#run-as-a-non-root-user).
Keep runtime writes possible; do not make the cache read-only to force a hit.

## Verify the packaged cache

Start a new container from the built image with the original serve command.
Do not mount a volume at or above the baked cache path: it would hide the
image's cache. Preserve the other mounts needed by the recipe.

For an AOT-enabled configuration, set `VLLM_FORCE_AOT_LOAD=1` for this check.
It is ignored when `VLLM_USE_AOT_COMPILE` is disabled. Confirm AOT is active and
record its resolved setting in the selected image; defaults depend on the
installed PyTorch version and compile-cache settings.
Look for `Directly load AOT compilation from path` or, for the non-AOT path,
`Directly load the compiled graph(s)` in the relevant worker logs. Check the
selected version's log messages. `Using cache directory` alone is not a hit.
Record compilation or AOT-load failures rather than reporting cache reuse as
verified. Readiness alone does not prove a hit.
A forced-load failure is a failed verification, not a reason to drop the flag.

Use a bounded readiness wait that fails if the server exits. While the packaged
server is still running, repeat the inference check and save the comparison
against the uncached reference before stopping it. Report success only after
both cache-loading and output checks pass. If cross-node reuse is required,
verify on a second matching node using only the image and its declared inputs.

Measure base-image and warm-image startup with the same readiness boundary,
configuration and model-download state. Report repetitions and separate image
pull time from server initialization. Compilation-cache reuse does not preserve
live weights or CUDA graphs across processes.

Return the image digest, pinned command/configuration, cache inventory,
verification logs, correctness result and any measured startup comparison.
Label unrun checks as pending; a built image is not a verified warm start.
