import ray
import torch

import vllm.envs as envs
from vllm.utils import (DeferredTensor, cuda_device_count_stateless, is_hip,
                        update_environment_variables)


@ray.remote
class _CUDADeviceCountStatelessTestActor:

    def get_count(self):
        return cuda_device_count_stateless()

    def set_cuda_visible_devices(self, cuda_visible_devices: str):
        update_environment_variables(
            {"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    def get_cuda_visible_devices(self):
        return envs.CUDA_VISIBLE_DEVICES


def test_cuda_device_count_stateless():
    """Test that cuda_device_count_stateless changes return value if
    CUDA_VISIBLE_DEVICES is changed."""
    if is_hip():
        # Set HIP_VISIBLE_DEVICES == CUDA_VISIBLE_DEVICES. Conversion
        # is handled by `update_environment_variables`
        update_environment_variables(
            {"CUDA_VISIBLE_DEVICES": envs.CUDA_VISIBLE_DEVICES})
    actor = _CUDADeviceCountStatelessTestActor.options(  # type: ignore
        num_gpus=2).remote()
    assert sorted(ray.get(
        actor.get_cuda_visible_devices.remote()).split(",")) == ["0", "1"]
    assert ray.get(actor.get_count.remote()) == 2
    ray.get(actor.set_cuda_visible_devices.remote("0"))
    assert ray.get(actor.get_count.remote()) == 1
    ray.get(actor.set_cuda_visible_devices.remote(""))
    assert ray.get(actor.get_count.remote()) == 0


def test_deferred_tensor():
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors = {
        "scalar": torch.ones(tuple()),
        "vector": torch.ones(2),
        "matrix": torch.ones((2, 3)),
        "tensor": torch.ones((2, 3, 4)),
    }
    save_file(tensors, "model.safetensors")

    with safe_open("model.safetensors", framework="pt") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            tensor_slice = f.get_slice(k)
            dt = DeferredTensor(tensor_slice)
            real_tensor = dt.materialize()
            real_tensor.copy_(dt)  # test we can use `copy_`
            stacked = torch.stack([real_tensor, real_tensor])
            stacked[0] = dt  # test we can use `__setitem__` to assign
            if k != "scalar":
                real_tensor[1:] = dt[1:]  # test we can assign slices
            if k in ["matrix", "tensor"]:
                real_norm = torch.nn.functional.normalize(real_tensor)
                dt_norm = torch.nn.functional.normalize(dt)
                assert torch.allclose(real_norm, dt_norm) # test we can use `normalize`
            assert torch.allclose(real_tensor.cpu(),
                                  dt.cpu())  # test we can move to device
            assert torch.allclose(
                real_tensor.to(dtype=torch.float64),
                dt.to(dtype=torch.float64))  # test we can change dtype

            assert torch.allclose(real_tensor + 1, dt + 1)
            assert torch.allclose(real_tensor.view(-1), dt.view(-1)) # test we can use `view`
            assert torch.allclose(torch.reshape(real_tensor, (-1,)), torch.reshape(dt, (-1,))) # test we can use `reshape`
