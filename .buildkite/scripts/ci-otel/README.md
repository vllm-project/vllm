# CI timing and GPU samples

The pipeline generator activates these helpers for trusted, traced CI commands.
`ci_otel.sh` records command spans and `ci_pytest.sh` injects the pytest timing
plugin. `ci_gpu.py` runs once per command, outside pytest workers, when
`nvidia-smi` is available. It queries up to 16 container-visible NVIDIA devices
once per second without importing PyTorch or creating a CUDA context.
Set `CI_INFRA_GPU_SAMPLING=0` in the job environment to disable sampling.

Samples contain device UUID, index, name, utilization percent, and used/total
device memory in bytes. They are `ci.gpu.sample` OTLP events inside
`ci.gpu.samples` spans, parented to the command span and authenticated using the
existing job-scoped Buildkite OIDC identity. Batches upload every 30 seconds;
the final batch is spooled for the shell's normal job-end upload. No host agent
or long-lived dashboard token is needed. The dashboard companion change adds
GPU controls alongside jobs, commands, and pytest rows in the Builds timeline.

Device queries and periodic uploads each have a two-second timeout. Three
consecutive query failures stop sampling. Failed periodic uploads are dropped;
memory stays bounded, and the dashboard shows the missing intervals as gaps.
Uploads can delay the next poll. The shell stops and joins the sampler before
flushing traces, allows at most three seconds for shutdown, and preserves the
original command exit status. Samples already uploaded survive an interrupted
job; the last unflushed batch may be lost on forced termination.

These are device readings during each test's time interval, not process-level
kernel attribution. Other processes and overlapping tests can contribute to
the readings. Subsecond tests may have no sample. Missing readings never become
zero utilization. MIG parent-device memory and utilization are omitted because
they cannot be attributed to the job's partition. The initial implementation
covers NVIDIA commands in the existing tracing rollout; CPU jobs and AMD
mirrors do not collect GPU samples.

Run the helper tests without loading vLLM or requiring a GPU:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest \
  .buildkite/scripts/ci-otel/tests/test_ci_otel.py -q
```

The suite uses a fake `nvidia-smi` to exercise the shell lifecycle, device-query
failure handling, disable switch, event encoding, and bounded uploads. Real
hardware sampling overhead still needs a canary run before broad deployment.
