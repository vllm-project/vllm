import os

from vllm.model_executor.layers.pooler import PoolingType
from vllm.model_executor.models.bert import BertEmbeddingModel

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

        model_tokenizer = model.model.llm_engine.tokenizer

        # asserts on the bert model config file
        assert model_config.bert_config["max_seq_length"] == 512
        assert model_config.bert_config["do_lower_case"]

        # asserts on the pooling config files
        assert model_config.pooler_config.pooling_type == PoolingType.CLS.name
        assert model_config.pooler_config.pooling_norm

        # asserts on the tokenizer loaded
        assert model_tokenizer.tokenizer_id == "BAAI/bge-base-en-v1.5"
        assert model_tokenizer.tokenizer_config["do_lower_case"]
        assert model_tokenizer.tokenizer.model_max_length == 512

        model = model.model.llm_engine.model_executor\
                     .driver_worker.model_runner.model
        assert isinstance(model, BertEmbeddingModel)
        assert model._pooler.pooling_type == PoolingType.CLS
        assert model._pooler.normalize
        # assert output
        assert output
