# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.snapshot.types import Oracle

_CANARY_PROMPT = "The capital of France is"


class SnapshotCanaryError(RuntimeError):
    """The initialized engine did not produce a valid snapshot oracle."""


async def _release_reloadable_state(engine: Any) -> None:
    """Discard model and KV state before the process image is captured."""
    await engine.sleep(level=2)


async def _restore_reloadable_state(engine: Any) -> None:
    """Rebuild state discarded by ``_release_reloadable_state``."""
    await engine.wake_up(tags=["weights"])
    await engine.collective_rpc("reload_weights")
    await engine.wake_up(tags=["kv_cache"])


def oracle_from_request_output(request_output: Any) -> Oracle:
    try:
        candidate = request_output.outputs[0]
        token_ids = tuple(candidate.token_ids)
        (sampled_token_id,) = token_ids
        return Oracle(
            token_ids=token_ids,
            text=candidate.text,
            sampled_token_logprob=candidate.logprobs[0][sampled_token_id].logprob,
        )
    except ValueError as error:
        raise SnapshotCanaryError(
            "snapshot canary must produce exactly one finite token logprob"
        ) from error
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise SnapshotCanaryError(
            "snapshot canary did not return sampled token logprob"
        ) from error


async def run_engine_canary(engine: Any) -> Oracle:
    from vllm import SamplingParams

    final_output = None
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=1,
        max_tokens=1,
        seed=0,
        logprobs=0,
    )
    async for output in engine.generate(
        _CANARY_PROMPT,
        sampling_params,
        request_id="vllm-snapshot-canary",
    ):
        final_output = output
    if final_output is None:
        raise SnapshotCanaryError("canary generation returned no output")
    return oracle_from_request_output(final_output)
