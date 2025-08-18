import pytest
from vllm.config import SpeculativeConfig, ParallelConfig


def test_spec_config_valid_and_invalid_ranges(dummy_target_config):
    # Valid
    cfg = SpeculativeConfig(
        method="layer_skip",
        layer_skip=6,
        num_speculative_tokens=5,
        target_model_config=dummy_target_config,
        target_parallel_config=ParallelConfig(),
        enable_chunked_prefill=False,
        disable_log_stats=True,
    )
    assert cfg.method == "layer_skip"
    assert cfg.layer_skip == 6
    assert cfg.num_speculative_tokens == 5
    assert cfg.model == dummy_target_config.model

    # Invalid: too large
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="layer_skip",
            layer_skip=999,
            num_speculative_tokens=3,
            target_model_config=dummy_target_config,
            target_parallel_config=ParallelConfig(),
            enable_chunked_prefill=False,
            disable_log_stats=True,
        )

    # Invalid: negative
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="layer_skip",
            layer_skip=-1,
            num_speculative_tokens=3,
            target_model_config=dummy_target_config,
            target_parallel_config=ParallelConfig(),
            enable_chunked_prefill=False,
            disable_log_stats=True,
        )


def test_spec_config_lsq_path_validation(dummy_target_config, tmp_path):
    bad = tmp_path / "does_not_exist"
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="layer_skip",
            layer_skip=4,
            lsq_head_path=str(bad),
            num_speculative_tokens=3,
            target_model_config=dummy_target_config,
            target_parallel_config=ParallelConfig(),
            enable_chunked_prefill=False,
            disable_log_stats=True,
        )