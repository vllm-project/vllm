import torch
from vllm.model_executor.parameter import BasevLLMParameter


def test_assert_and_load_shape_mismatch_raises():
    # parameter expects shape (2,)
    param_tensor = torch.empty(2)

    # create BasevLLMParameter with a dummy loader
    p = BasevLLMParameter(data=param_tensor, weight_loader=lambda: None)

    # loaded_weight has incompatible shape (3,)
    loaded = torch.empty(3)

    try:
        p._assert_and_load(loaded)
        assert False, "Expected ValueError on shape mismatch"
    except ValueError as e:
        assert "weight shape mismatch" in str(e)
        assert "(2,)" in str(e)
        assert "(3,)" in str(e)
