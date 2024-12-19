import pytest

from vllm.inputs import InputContext

from ....utils import build_model_context


@pytest.fixture()
def get_max_llava_next_image_tokens():
    from vllm.model_executor.models.llava_next import (
        get_max_llava_next_image_tokens)
    return get_max_llava_next_image_tokens


@pytest.fixture()
def dummy_data_for_llava_next():
    from vllm.model_executor.models.llava_next import dummy_data_for_llava_next
    return dummy_data_for_llava_next


@pytest.mark.parametrize("gridpoints,expected_max_tokens", [
    ([[336, 336]], 1176),
    ([[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]], 2928),
])
def test_get_max_llava_next_image_tokens(gridpoints, expected_max_tokens,
                                         get_max_llava_next_image_tokens):
    ctx = build_model_context(model_name="llava-hf/llava-v1.6-mistral-7b-hf")

    # Update the config image_grid_pinpoints
    # and calculate the resulting max tokens
    ctx.model_config.hf_config.image_grid_pinpoints = gridpoints

    actual_max_tokens = get_max_llava_next_image_tokens(
        InputContext(ctx.model_config))

    assert expected_max_tokens == actual_max_tokens


@pytest.mark.parametrize(
    "gridpoints,expected_size",
    [
        # One point; it has to be the largest
        ([[336, 336]], (336, 336)),
        # Default for most llava next models; the 2x2 tile is the largest
        ([[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
         (672, 672)),
        # If two rectangular gridpoints are the same, the more vertical
        # one has the higher feature count due to newline features
        ([[336, 672], [672, 336]], (672, 336))
    ])
def test_dummy_data_for_llava_next_feature_size(dummy_data_for_llava_next,
                                                gridpoints, expected_size):
    ctx = build_model_context(model_name="llava-hf/llava-v1.6-mistral-7b-hf")

    # Update the config image_grid_pinpoints
    ctx.model_config.hf_config.image_grid_pinpoints = gridpoints
    seq_len = 5000  # bigger than the max feature size for any image

    dummy_data = dummy_data_for_llava_next(
        ctx,
        seq_len=seq_len,
        mm_counts={"image": 1},
    )
    seq_data = dummy_data.seq_data
    mm_data = dummy_data.multi_modal_data

    # The dummy data dims should match the gridpoint with the biggest feat size
    assert mm_data["image"].height == expected_size[0]
    assert mm_data["image"].width == expected_size[1]
    assert len(seq_data.get_token_ids()) >= seq_len
