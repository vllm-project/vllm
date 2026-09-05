from unittest.mock import Mock, patch


def test_phi_embedding_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.phi import PhiModel

    quant_config = Mock()

    config = Mock()
    config.vocab_size = 128
    config.hidden_size = 64
    config.num_hidden_layers = 0
    config.layer_norm_eps = 1e-5

    vllm_config = Mock()
    vllm_config.model_config.hf_config = config
    vllm_config.cache_config = Mock()
    vllm_config.quant_config = quant_config
    vllm_config.compilation_config.mode = CompilationMode.NONE

    with (
        patch("vllm.model_executor.models.phi.VocabParallelEmbedding") as embedding,
        patch(
            "vllm.model_executor.models.phi.make_layers",
            return_value=(0, 0, []),
        ),
        patch(
            "vllm.model_executor.models.phi.make_empty_intermediate_tensors_factory",
            return_value=Mock(),
        ),
    ):
        PhiModel(vllm_config=vllm_config, prefix="model")

    embedding.assert_called_once_with(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix="model.embed_tokens",
    )
