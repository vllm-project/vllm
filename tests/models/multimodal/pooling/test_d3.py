# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""d3-edge against the checkpoint's own runtime (`trust_remote_code`).

The runtime reads a question's options from the 255 code logits of its readout
at the last prompt token. vLLM's classify head must give the same code logits
for the same prompt tokens, and for an image prompt that vLLM processes itself.
"""

import pytest
import torch
from transformers import AutoModel

from vllm.assets.image import ImageAsset
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "vllm-sr/d3-edge"
STATE = (
    "The order arrived damaged yesterday. The customer has a receipt and asks "
    "for a replacement today."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"],
    },
    "department": {
        "type": "choice",
        "criteria": {f"department {i}": None for i in range(40)},
    },
}
IMAGE_QUESTIONS = {"photo": {"type": "noul", "instructions": "Is this a photograph?"}}


def option_probs(logits: torch.Tensor, count: int) -> torch.Tensor:
    return torch.softmax(logits[:count].float(), dim=-1)


@pytest.mark.parametrize("model", [MODEL])
def test_d3_matches_the_checkpoint_runtime(vllm_runner, model: str) -> None:
    reference = AutoModel.from_pretrained(model, trust_remote_code=True)
    runtime = reference.runtime
    text = runtime.prepare(STATE, QUESTIONS)
    keys = text.runnable
    counts = [len(text.questions[key].keys) for key in keys]
    sequences = [text.sequences[key] for key in keys]
    ref_logits = list(runtime.logits(sequences, counts).cpu())

    image = ImageAsset("stop_sign").pil_image
    visual = runtime.prepare("Describe the attached image.", IMAGE_QUESTIONS, [image])
    (image_key,) = visual.runnable
    image_prompt = visual.texts[image_key]
    ref_logits += list(
        runtime.image_logits(
            [image_prompt], visual.images, [2], visual.lengths[image_key]
        ).cpu()
    )
    counts.append(2)
    del reference, runtime
    cleanup_dist_env_and_memory()

    with vllm_runner(model, runner="pooling", max_model_len=4096) as vllm_model:
        outputs = vllm_model.llm.classify(
            [{"prompt_token_ids": ids} for ids in sequences]
            + [{"prompt": image_prompt, "multi_modal_data": {"image": image}}],
            use_tqdm=False,
        )
    # The head returns raw code logits by default.
    vllm_logits = [torch.tensor(output.outputs.probs) for output in outputs]
    assert all(len(logits) == 255 for logits in vllm_logits)

    for ref, ours, count in zip(ref_logits, vllm_logits, counts):
        ref_probs, our_probs = option_probs(ref, count), option_probs(ours, count)
        torch.testing.assert_close(our_probs, ref_probs, atol=0.02, rtol=0)
