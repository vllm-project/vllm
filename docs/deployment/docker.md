---
toc_depth: 2
---

# Using Docker

## Pre-built images

--8<-- "docs/getting_started/installation/gpu.md:pre-built-images"

## Run as a non-root user

The CUDA `vllm/vllm-openai` image runs as root by default for backward
compatibility. It is also prepared to run as the built-in `vllm` user
(UID 2000, GID 0):

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

When mounting model or cache volumes for a non-root container, mount writable
paths under `/home/vllm` instead of `/root`. For example, mount the Hugging
Face cache at `/home/vllm/.cache/huggingface` and make the mounted directory
writable by group 0.

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

To build an image that defaults to the non-root `vllm` user, use the opt-in
`vllm-openai-nonroot` target:

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

The `vllm-openai-nonroot` target also supports OpenShift-style arbitrary UIDs
when the runtime UID is a member of group 0. In Kubernetes manifests, set the
container security context accordingly and keep mounted cache/model paths
writable by group 0:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```

Runtime UIDs outside group 0 are not part of the documented support matrix
because they may be unable to write to `/home/vllm` or `/opt/uv/cache`.

## Fast restarts (imports snapshot)

The `vllm/vllm-openai` image snapshots the server's Python import state with
CRIU and restores it on later starts instead of re-paying the import cost.
This is enabled by default when the container has the privileges CRIU needs:
`docker run --privileged` is the verified way to grant them
(`CAP_CHECKPOINT_RESTORE` and `CAP_SYS_PTRACE` for dump/restore, plus
`CAP_SYS_ADMIN` because CRIU's kernel-feature probe mounts a tmpfs, which
container seccomp/AppArmor policy also has to allow). Without those
capabilities every start is a normal cold start with zero snapshot overhead.
The capability check is necessary but not sufficient. A container that holds
the capabilities but still confines CRIU (default seccomp or AppArmor) fails
CRIU's kernel-feature probe. The priming attempt runs the import work before
that probe, and it leaves no marker behind, so this cost repeats on every
start rather than only the first one. Serving is unaffected, because the
failure falls back to a normal cold start. Use `--privileged`, or set
`VLLM_SNAPSHOT=0` to skip priming entirely on such hosts.

The first privileged start primes the snapshot at normal speed (~546MB under
`VLLM_SNAPSHOT_ROOT`, default `/root/.cache/vllm/snapshots`); later starts
restore the import state instead. Restores survive same-container restarts
automatically; a recreated container (a fresh `docker run`) only restores if
`VLLM_SNAPSHOT_ROOT` points at a persistent volume, for example
`-v vllm-snapshots:/root/.cache/vllm/snapshots`. A fresh container without a
volume simply cold-starts and re-primes.

Fleets pinned to one image can bake the snapshot instead by running
`vllm snapshot create` during their own image build, on a builder matching
the runtime kernel and CPU (compatibility, not same-host identity). Set
`VLLM_SNAPSHOT=0` to opt out.

Treat `VLLM_SNAPSHOT_ROOT` as trusted input, on the same footing as the
compiled-kernel cache. A restore executes the process images it finds there, so
anyone who can write to that path can run code in the next server that reads
it.

Give each trust domain its own snapshot volume. The default image runs as
root, so containers that share one volume also share write access to it, and a
compromised container can replace the images that every later container
restores. That outlives the container which wrote it, and container
replacement would otherwise clear it. The volume has to stay writable, because
a restore takes a lock file inside it, so read-only mounting is not an
alternative.

vLLM refuses to create or restore a snapshot under a world-writable directory
and falls back to a cold start. That catches a volume mounted or pre-created
at 0777, including a world-writable parent directory; sticky-bit parents like
/tmp are allowed. It is not a general integrity check, and it deliberately allows
group-writable and root-owned directories, because arbitrary-UID pods need
those. Anyone inside the trust domain of the volume can still replace the
images, which is why the separation above is the real control.

## Build image from source

--8<-- "docs/getting_started/installation/gpu.md:build-image-from-source"
