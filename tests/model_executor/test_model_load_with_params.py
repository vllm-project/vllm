import os

MAX_MODEL_LEN = 128
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-base-en-v1.5")
REVISION = os.environ.get("REVISION", "main")


def test_model_loading_with_params(vllm_runner):
    """
    Test parameter weight loading with tp>1.
    """
    with vllm_runner(model_name=MODEL_NAME,
                     revision=REVISION,
                     dtype="float16",
                     max_model_len=MAX_MODEL_LEN) as model:
        output = model.encode("Write a short story about a robot that"
                              " dreams for the first time.\n")

        model_config = model.model.llm_engine.model_config

        assert model_config.max_model_len == 512
        assert model_config.do_lower_case
        assert model_config.pooling_config.pooling_type == 2
        assert model_config.pooling_config.normalize
        assert output
