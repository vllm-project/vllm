# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


@pytest.mark.parametrize("stop,reason", [("xxx", "stop"), ("!", "length")])
def test_spec_decode_metrics_cover_text_stop_and_core_finish(stop, reason):
    scheduler = create_scheduler(
        num_speculative_tokens=2, per_request_spec_decode_metrics="detailed"
    )
    params = SamplingParams(stop=stop, max_tokens=4)
    (req,) = create_requests(num_requests=1, num_tokens=1, sampling_params=params)
    scheduler.add_request(req)

    tokenizer = Tokenizer(models.WordLevel({"prompt": 0, "x": 1}))
    tokenizer.decoder = decoders.Fuse()
    frontend = OutputProcessor(
        PreTrainedTokenizerFast(tokenizer_object=tokenizer), collect_stats=False
    )
    frontend.add_request(
        EngineCoreRequest(
            request_id=req.request_id,
            external_req_id=req.request_id,
            prompt_token_ids=[0],
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )

    # Prefill, then accept one of two drafts on the first verification step.
    # A second verification is needed only when the frontend didn't stop.
    for drafted, sampled in (([], [1]), ([1, 0], [1, 1]), ([0], [1])):
        if drafted:
            scheduler.update_draft_token_ids(DraftTokenIds([req.request_id], [drafted]))
        step = scheduler.schedule()
        result = ModelRunnerOutput(
            req_ids=[req.request_id],
            sampled_token_ids=[sampled],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        core_outputs = scheduler.update_from_output(step, result)[0].outputs
        (output,) = frontend.process_outputs(core_outputs).request_outputs
        if output.finished:
            break
    (completion,) = output.outputs
    assert completion.finish_reason == reason
    assert completion.spec_decode_metrics is not None
    metrics = completion.spec_decode_metrics.to_dict()
    assert metrics["per_step_accepted"] == ([1] if reason == "stop" else [1, 0])
    assert metrics["num_draft_tokens"] == (2 if reason == "stop" else 3)
