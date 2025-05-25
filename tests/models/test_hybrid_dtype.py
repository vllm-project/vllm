import pytest

from tests.models.language.pooling.mteb_utils import mteb_test_embed_models
from tests.models.utils import Dtype, EmbedModelInfo, check_logprobs_close

high_precision_data_types = [
    Dtype(dtype="float32"),
    Dtype(dtype="hybrid"),
    Dtype(dtype="float32", attn_dtype="float16"),
    Dtype(dtype="float32", attn_dtype="bfloat16")
]
low_precision_data_types = [
    Dtype(dtype="auto"),
    Dtype(dtype="float16"),
    Dtype(dtype="bfloat16")
]
data_types = high_precision_data_types + low_precision_data_types
embed_model = "intfloat/e5-small"
generate_model = "EleutherAI/pythia-70m"


@pytest.mark.parametrize("dtype", data_types)
def test_embed_model(hf_runner, vllm_runner, dtype: Dtype):
    model_info = EmbedModelInfo(embed_model,
                                architecture="BertModel",
                                dtype=dtype)

    if model_info.dtype in high_precision_data_types:
        mteb_test_embed_models(hf_runner, vllm_runner, model_info)
    else:
        with pytest.raises(AssertionError):
            mteb_test_embed_models(hf_runner, vllm_runner, model_info)


@pytest.mark.parametrize("model", [generate_model])
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [4])
def test_generate_models(hf_runner, vllm_runner, example_prompts, model: str,
                         dtype: Dtype, max_tokens: int,
                         num_logprobs: int) -> None:
    if dtype in [Dtype(dtype="float32", attn_dtype="bfloat16")]:
        pytest.skip("This combination can't pass the test.")

    def run_test():
        with hf_runner(model, dtype="float32") as hf_model:
            hf_outputs = hf_model.generate_greedy_logprobs_limit(
                example_prompts, max_tokens, num_logprobs)

        with vllm_runner(model, dtype=dtype.dtype,
                         attn_dtype=dtype.attn_dtype) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, num_logprobs)

        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )

    if dtype in high_precision_data_types:
        run_test()
    else:
        with pytest.raises(AssertionError):
            run_test()
