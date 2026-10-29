# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connectors that ask for hidden states receive, through
save_hidden_states, each step's rows matching the generation."""

import os
from pathlib import Path

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    HiddenStatesRequest,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    StepHiddenStates,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)

MODEL = "Qwen/Qwen3-0.6B"
MODULE = "tests.v1.kv_connector.unit.test_save_hidden_states"


class RecordingConnector(KVConnectorBase_V1):
    """Asks for the hidden states in ``extra_config["layers"]`` and saves each
    step's rows to ``extra_config["dump_dir"]``."""

    @classmethod
    def get_required_hidden_states(cls, vllm_config):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        layers = extra.get("layers")
        return HiddenStatesRequest(None if layers is None else tuple(layers))

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.dump_dir = Path(extra.get("dump_dir", "."))
        self.step = 0

    def on_new_request(self, request):
        # Every position of a request is computed, so every one is received.
        request.skip_reading_prefix_cache = True

    def save_hidden_states(self, hidden_states: StepHiddenStates) -> None:
        n = hidden_states.num_tokens
        torch.save(
            {
                "req_ids": list(hidden_states.req_ids),
                "query_start_loc": hidden_states.query_start_loc.tolist(),
                "layers": hidden_states.layers,
                "input_ids": hidden_states.input_ids[:n].cpu(),
                "positions": hidden_states.positions[:n].cpu(),
                "aux": [h[:n].cpu() for h in hidden_states.aux_hidden_states],
                "final": hidden_states.hidden_states[:n].cpu(),
                "sampled": hidden_states.sampled_token_ids.cpu(),
                "num_sampled": hidden_states.num_sampled.cpu(),
                "num_rejected": hidden_states.num_rejected.cpu(),
            },
            self.dump_dir / f"{os.getpid()}-{self.step}.pt",
        )
        self.step += 1

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()


class FinalOnlyConnector(RecordingConnector):
    @classmethod
    def get_required_hidden_states(cls, vllm_config):
        return HiddenStatesRequest(layers=())


KVConnectorFactory.register_connector(
    "RecordingConnector", MODULE, RecordingConnector.__name__
)
KVConnectorFactory.register_connector(
    "FinalOnlyConnector", MODULE, FinalOnlyConnector.__name__
)


def test_connectors_ask_for_no_hidden_states_by_default():
    config = create_vllm_config(model=MODEL, kv_connector="ExampleConnector")
    connector_cls = KVConnectorFactory.get_connector_class(config.kv_transfer_config)
    assert connector_cls.get_required_hidden_states(config) is None


def test_multi_connector_merges_one_request_and_refuses_two(monkeypatch):

    def multi(*children):
        return create_vllm_config(
            model=MODEL,
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": c, "kv_role": "kv_both"} for c in children
                ]
            },
        )

    assert MultiConnector.get_required_hidden_states(
        multi("ExampleConnector", "RecordingConnector")
    ) == HiddenStatesRequest(None)
    with pytest.raises(ValueError, match="different hidden states"):
        MultiConnector.get_required_hidden_states(
            multi("RecordingConnector", "FinalOnlyConnector")
        )


def _lm_head():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(MODEL)
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        tied = "lm_head.weight" not in set(f.keys())
        return f.get_tensor(
            "model.embed_tokens.weight" if tied else "lm_head.weight"
        ).float()


@pytest.mark.parametrize("speculative", [False, True])
def test_save_hidden_states_matches_generation(monkeypatch, tmp_path, speculative):
    from vllm import LLM, SamplingParams

    kwargs = {}
    if speculative:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }
    llm = LLM(
        model=MODEL,
        max_model_len=1024,
        gpu_memory_utilization=0.3,
        kv_transfer_config={
            "kv_connector": "RecordingConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": MODULE,
            "kv_connector_extra_config": {"dump_dir": str(tmp_path)},
        },
        **kwargs,
    )
    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 6 + "Repeat it:",
        "Write the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9 and continue:",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=32)
    # Twice: the second round would read the prefix cache if it were allowed.
    outputs = llm.generate(prompts, params) + llm.generate(prompts, params)
    del llm

    rows: dict[str, list[tuple]] = {}
    for path in sorted(tmp_path.glob("*.pt")):
        step = torch.load(path)
        loc = step["query_start_loc"]
        rejected = step["num_rejected"].tolist()
        for i, req_id in enumerate(step["req_ids"]):
            # The request's last num_rejected rows are rejected drafts.
            s = slice(loc[i], loc[i + 1] - rejected[i])
            rows.setdefault(req_id, []).append(
                (step["positions"][s], step["input_ids"][s], step["final"][s])
            )
        assert step["layers"] == (2, 14, 25)  # Qwen3-0.6B's EAGLE-3 defaults
        assert len(step["aux"]) == 3

    lm_head = _lm_head()
    for output in outputs:
        assert output.num_cached_tokens == 0
        (req_id,) = [r for r in rows if r.startswith(output.request_id + "-")]
        positions, tokens, final = (torch.cat(p) for p in zip(*rows[req_id]))
        assert output.prompt_token_ids is not None
        full = torch.tensor(output.prompt_token_ids + list(output.outputs[0].token_ids))
        # Under speculative decoding a request can run a step past its end.
        keep = positions < len(full) - 1
        order = torch.argsort(positions[keep])
        assert torch.equal(positions[keep][order], torch.arange(len(full) - 1))
        assert torch.equal(tokens[keep][order].long(), full[:-1])
        # Greedy decoding: the final hidden states produce the generated tokens.
        generated = slice(len(output.prompt_token_ids) - 1, None)
        logits = final[keep][order][generated].float() @ lm_head.T
        assert torch.equal(logits.argmax(-1), full[1:][generated])
