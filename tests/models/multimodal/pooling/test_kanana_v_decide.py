# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kanana-2-v-decide against the checkpoint's own runtime (`trust_remote_code`).

The runtime embeds each question's query and its candidate answers and answers
with the softmax of their cosine similarities. vLLM must embed the runtime's
prompts, text and image, to the same vectors and so the same probabilities.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from tests.models.utils import check_embeddings_close
from tests.utils import ROCM_ENGINE_KWARGS
from vllm.assets.image import ImageAsset
from vllm.entrypoints.openai.decisions.strategies import kanana_fit_image

MODEL = "kakaocorp/kanana-2-v-decide-3b-2611"
REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
EMBEDDING_TOL = 1e-3
PROBABILITY_ATOL = 0.02
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
}
IMAGE_STATE = "Describe the attached image."
IMAGE_QUESTIONS = {
    "sign": {
        "type": "choice",
        "instructions": "Which traffic sign is in the image?",
        "criteria": {"stop": None, "yield": None, "speed limit": None},
    },
    "has_sign": {"type": "noul", "instructions": "Is there a traffic sign?"},
}


def remote(name: str):
    return get_class_from_dynamic_module(name, MODEL, revision=REVISION)


def test_kanana_v_decide_matches_the_checkpoint_runtime(hf_runner, vllm_runner):
    build_pairs = remote("decision_kanana_v_decide.build_pairs")
    query_messages = remote("decision_kanana_v_decide.query_messages")
    validate_candidates = remote("decision_kanana_v_decide.validate_candidates")
    message_ids = remote("decision_kanana_v_decide.message_ids")
    fit_image = remote("processing_kanana_v_decide.fit_image")
    processor = remote(
        "processing_kanana_v_decide.KananaVDecideProcessor"
    ).from_pretrained(MODEL, revision=REVISION)
    tokenizer = processor.tokenizer

    raw_image = ImageAsset("stop_sign").pil_image
    image = fit_image(raw_image)
    assert kanana_fit_image(raw_image).tobytes() == image.tobytes()

    cases = [(STATE, QUESTIONS, None), (IMAGE_STATE, IMAGE_QUESTIONS, [raw_image])]
    messages: list[list[dict]] = []
    reads = []
    for state, questions, images in cases:
        for payload, keys, candidates in build_pairs(
            state, questions, images=images
        ).values():
            query = len(messages)
            messages.append(query_messages(payload, tokenizer, processor))
            messages += [
                m for m, _ in validate_candidates(candidates, tokenizer, processor)
            ]
            reads.append((query, list(range(query + 1, len(messages))), keys))
    ref_ids = [message_ids(m, tokenizer, processor) for m in messages]

    prompts = []
    for m, ids in zip(messages, ref_ids):
        if any(p["type"] == "image" for p in m[1]["content"]):
            prompt = processor.apply_chat_template(m)
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": [image]}})
        else:
            prompts.append({"prompt_token_ids": ids})
    with vllm_runner(
        MODEL,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=4096,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        revision=REVISION,
        trust_remote_code=True,
        **ROCM_ENGINE_KWARGS,
    ) as vllm_model:
        outputs = vllm_model.llm.embed(prompts, use_tqdm=False)
    assert [o.prompt_token_ids for o in outputs] == ref_ids
    vllm_embeddings = [o.outputs.embedding for o in outputs]

    with hf_runner(
        MODEL,
        dtype="bfloat16",
        auto_cls=AutoModel,
        trust_remote_code=True,
        revision=REVISION,
    ) as hf_model:
        hf_embeddings, _ = hf_model.model.embed_messages(messages)
        answers = {}
        for state, questions, images in cases:
            result = hf_model.model.system_one(state, questions, images=images)
            answers.update(result["answers"])
        temperature = hf_model.model.config.decision_temperature

    check_embeddings_close(
        embeddings_0_lst=hf_embeddings.tolist(),
        embeddings_1_lst=vllm_embeddings,
        name_0="hf",
        name_1="vllm",
        tol=EMBEDDING_TOL,
    )
    vectors = F.normalize(torch.tensor(vllm_embeddings), dim=-1)
    for (query, candidates, keys), answer in zip(reads, answers.values()):
        probs = torch.softmax(vectors[candidates] @ vectors[query] / temperature, -1)
        if answer["type"] == "noul":
            expected = torch.tensor([1 - answer["noul"], answer["noul"]])
        else:
            expected = torch.tensor([answer["probabilities"][k] for k in keys])
        torch.testing.assert_close(probs, expected, atol=PROBABILITY_ATOL, rtol=0)
        if answer["type"] == "choice":
            assert keys[int(probs.argmax())] == answer["choice"]
