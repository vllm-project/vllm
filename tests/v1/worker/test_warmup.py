# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


class _KVConnector:
    def __init__(self):
        self.disabled_states: list[bool] = []

    def set_disabled(self, disabled: bool) -> None:
        self.disabled_states.append(disabled)


def test_mixed_prefill_decode_warmup_runs_scheduler_steps():
    kv_connector = _KVConnector()
    model_runner = SimpleNamespace(
        is_pooling_model=False,
        kv_connector=kv_connector,
        kv_cache_config=SimpleNamespace(
            num_blocks=16,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8)),
            ],
        ),
    )
    execute_outputs = []
    sample_outputs = []

    def execute_model(scheduler_output):
        execute_outputs.append(scheduler_output)

    def sample_tokens(grammar_output):
        sample_outputs.append(grammar_output)

    ran = run_mixed_prefill_decode_warmup(
        model_runner,
        execute_model,
        sample_tokens,
        7,
        req_id_prefix="_test",
    )

    assert ran
    assert kv_connector.disabled_states == [True, False]
    assert len(execute_outputs) == 3
    assert len(sample_outputs) == 2

    decode_prefill, mixed, cleanup = execute_outputs
    assert decode_prefill.total_num_scheduled_tokens == 2
    assert decode_prefill.scheduled_new_reqs[0].req_id == "_test_decode_"

    assert mixed.total_num_scheduled_tokens == 7
    assert mixed.scheduled_cached_reqs.req_ids == ["_test_decode_"]
    assert mixed.scheduled_new_reqs[0].req_id == "_test_prefill_"
    assert mixed.num_scheduled_tokens == {
        "_test_decode_": 1,
        "_test_prefill_": 6,
    }

    assert cleanup.finished_req_ids == {"_test_decode_", "_test_prefill_"}
