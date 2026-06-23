from pathlib import Path


def test_v2_gpu_model_runner_has_internal_profile_scopes():
    source = Path("vllm/v1/worker/gpu/model_runner.py").read_text()

    expected_scopes = [
        "v2_gpu_model_runner: state_update",
        "v2_gpu_model_runner: dispatch",
        "v2_gpu_model_runner: prepare_inputs",
        "v2_gpu_model_runner: prepare_attn",
        "v2_gpu_model_runner: mm_embeddings",
        "v2_gpu_model_runner: forward",
        "v2_gpu_model_runner: sample_logits",
        "v2_gpu_model_runner: prompt_logprobs",
        "v2_gpu_model_runner: async_output",
        "v2_gpu_model_runner: sample_postprocess",
        "v2_gpu_model_runner: draft",
    ]

    for scope in expected_scopes:
        assert scope in source
