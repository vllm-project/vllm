"""Requests for the offline benchmark runners.

Workloads come from ``vllm.benchmarks.datasets``; ``from_sample_requests``
converts their samples.  Import those datasets after the simulator hooks are
installed (i.e. after ``vllm_simulator.simulation.vllm.vllm_worker``).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from vllm.benchmarks.datasets import SampleRequest


@dataclass(slots=True)
class GenericRequest:
    prompt: Optional[str] = None
    token_ids: Optional[list[int]] = None
    input_length: int = -1
    output_length: int = -1
    custom_params: dict = field(default_factory=dict)
    # Extra args used as API request parameters when sending the request
    extra_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt is None and self.token_ids is None:
            raise ValueError("Invalid Request")


def from_sample_requests(
    samples: Iterable["SampleRequest"],
) -> list[GenericRequest]:
    """Convert ``vllm.benchmarks.datasets`` samples for the offline runners.

    A sample timestamp becomes ``created_time``.  Only single text or
    token-id prompts are supported.
    """
    requests = []
    for sample in samples:
        if sample.multi_modal_data is not None or sample.lora_request is not None:
            raise ValueError("Multimodal and LoRA samples are not supported")
        prompt = sample.prompt
        if isinstance(prompt, str):
            req = GenericRequest(prompt=prompt)
        elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            req = GenericRequest(token_ids=list(prompt))
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt).__name__}")
        req.input_length = sample.prompt_len
        req.output_length = sample.expected_output_len
        if sample.timestamp is not None:
            req.custom_params["created_time"] = sample.timestamp
        requests.append(req)
    return requests
