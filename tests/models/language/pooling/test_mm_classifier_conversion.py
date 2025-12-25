# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.pooler import PoolerConfig


def test_idefics_multimodal(
    vllm_runner,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    with vllm_runner(
        model_name="HuggingFaceM4/Idefics3-8B-Llama3",
        runner="pooling",
        convert="classify",
        load_format="dummy",
        max_model_len=512,
        enforce_eager=True,
        tensor_parallel_size=1,
        disable_log_stats=True,
        dtype="bfloat16",
    ) as vllm_model:
        llm = vllm_model.get_llm()
        outputs = llm.classify(prompts)
        for output in outputs:
            assert len(output.outputs.probs) == 2


def update_config(config):
    config.text_config.update(
        {
            "architectures": ["Gemma3ForSequenceClassification"],
            "classifier_from_token": ["A", "B", "C", "D", "E"],
            "method": "no_post_processing",
            "id2label": {
                "A": "Chair",
                "B": "Couch",
                "C": "Table",
                "D": "Bed",
                "E": "Cupboard",
            },
        }
    )
    return config


def test_gemma_multimodal(
    vllm_runner,
) -> None:
    messages = [
        {
            "role": "system",
            "content": """
    You are a helpful assistant. You will be given a product description
    which may also include an image. Classify the following product into
    one of the categories:

    A = chair
    B = couch
    C = table
    D = bed
    E = cupboard

    You'll answer with exactly one letter (A, B, C, D, or E).""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/red_chair.jpg"
                    },
                },
                {"type": "text", "text": "A fine 19th century piece of furniture."},
            ],
        },
    ]

    with vllm_runner(
        model_name="google/gemma-3-4b-it",
        runner="pooling",
        convert="classify",
        load_format="auto",
        hf_overrides=update_config,
        pooler_config=PoolerConfig(pooling_type="LAST"),
        max_model_len=512,
        enforce_eager=True,
        tensor_parallel_size=1,
        disable_log_stats=True,
        dtype="bfloat16",
    ) as vllm_model:
        llm = vllm_model.get_llm()
        prompts = llm.preprocess_chat(messages)

        result = llm.classify(prompts)
        assert result[0].outputs.probs[0] > 0.95
        assert all(c < 0.05 for c in result[0].outputs.probs[1:])
