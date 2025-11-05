# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE: if your PR has broken one of the tests here (sorry),
# kindly patch the corresponding integration in
# /vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py
# or reach out to @aposataC for assistance

# Assumption vs. Correctness Tests:
# these unit tests do *not* test correctness of LMCache-side or vLLM-side logic
# it is to ensure that assumptions LMCache makes about vLLM's interface are stable
def assumes(obj, attr, is_callable=False, is_instance_of=None):
    import inspect
    from dataclasses import is_dataclass

    assumption_msg = (
        f"LMCache connector currently assumes that {obj} has a(n) {attr} attribute"
    )
    if hasattr(obj, attr):
        attr_value = getattr(obj, attr)
    elif is_dataclass(obj) and attr in getattr(obj, "__dataclass_fields__", {}):
        field = obj.__dataclass_fields__[attr]
        field_type = field.type
        origin = getattr(field_type, "__origin__", None)
        if origin is not None:
            field_type = origin
        attr_value = field_type
    else:
        raise AssertionError(assumption_msg)
    if is_callable:
        assumption_msg += f" and that {obj}.{attr} is a callable"
        assert callable(attr_value), assumption_msg
    if is_instance_of:
        assumption_msg += f" and that {obj}.{attr} is an instance of {is_instance_of}"
        if isinstance(attr_value, property):
            fget = attr_value.fget
            assert fget is not None, f"Property {obj}.{attr} has no fget"
            sig = inspect.signature(fget)
            ret_anno = sig.return_annotation
            assert ret_anno is not inspect._empty, (
                f"Property {obj}.{attr} has no return annotation"
            )
            assert ret_anno == is_instance_of, assumption_msg
        else:
            if isinstance(attr_value, type):
                assert attr_value is is_instance_of, assumption_msg
            else:
                assert isinstance(attr_value, is_instance_of), assumption_msg


def test_multimodal_interface():
    # protect against interface changes
    from vllm.multimodal.inputs import PlaceholderRange

    assumes(PlaceholderRange, "offset")
    assumes(PlaceholderRange, "length")

    # test a minimal case
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (
        apply_mm_hashes_to_token_ids,
    )

    token_ids = torch.arange(10, dtype=torch.long)
    mm_hashes = ["0000", "1111"]  # hex repr of 0 and 4369
    mm_positions = [
        PlaceholderRange(offset=0, length=4),
        PlaceholderRange(offset=5, length=4),
    ]
    apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
    assert token_ids.tolist() == [0, 0, 0, 0, 4, 4369, 4369, 4369, 4369, 9]


def test_config_interface():
    # protect against interface changes
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

    assumes(VllmConfig, "model_config")
    assumes(VllmConfig, "cache_config")
    assumes(VllmConfig, "parallel_config")
    assumes(VllmConfig, "kv_transfer_config")

    assumes(KVTransferConfig, "kv_role")
    assumes(KVTransferConfig, "kv_connector_extra_config")

    assumes(ModelConfig, "use_mla", is_instance_of=bool)
    assumes(ModelConfig, "dtype")
    assumes(ModelConfig, "max_model_len")
    assumes(ModelConfig, "get_vocab_size", is_callable=True)
    assumes(ModelConfig, "get_num_attention_heads", is_callable=True)
    assumes(ModelConfig, "get_num_kv_heads", is_callable=True)
    assumes(ModelConfig, "get_head_size", is_callable=True)
    assumes(ModelConfig, "get_num_layers", is_callable=True)
    assumes(ModelConfig, "get_num_kv_heads", is_callable=True)
    assumes(ModelConfig, "model")

    assumes(ParallelConfig, "world_size")
    assumes(ParallelConfig, "rank")
    assumes(ParallelConfig, "tensor_parallel_size")
    assumes(ParallelConfig, "pipeline_parallel_size")
    assumes(ParallelConfig, "data_parallel_size_local")
    assumes(ParallelConfig, "data_parallel_rank_local")

    assumes(CacheConfig, "cache_dtype")
    assumes(CacheConfig, "block_size")
    assumes(CacheConfig, "gpu_memory_utilization")

    # mla metadata minimal cases
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (
        mla_enabled,
    )

    model_config = ModelConfig(model="deepseek-ai/DeepSeek-R1")
    assert mla_enabled(model_config)
    model_config = ModelConfig(model="Qwen/Qwen3-0.6B")
    assert not mla_enabled(model_config)

    # kv metadata minimal case
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype

    model_config = ModelConfig(dtype="bfloat16")
    parallel_config = ParallelConfig()
    cache_config = CacheConfig(cache_dtype="bfloat16")
    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)
    use_mla = mla_enabled(model_config)
    chunk_size = 256
    num_layer = model_config.get_num_layers(parallel_config)
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    # dummy lmcache metadata creation example
    _ = (
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
    )


def test_request_interface():
    # protect against interface changes
    from types import NoneType

    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    req = Request(
        request_id="test_request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
    )
    assumes(req, "mm_features", is_instance_of=(list, NoneType))
    assumes(req, "request_id")
    assumes(req, "priority")
    assumes(req, "prompt_token_ids")
    assumes(req, "sampling_params")
    assumes(req, "num_tokens")
    assumes(req, "kv_transfer_params", is_instance_of=(dict, NoneType))

    from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem

    assumes(MultiModalFeatureSpec, "identifier")
    assumes(MultiModalFeatureSpec, "mm_position")

    # minimal case:
    from vllm.multimodal.inputs import PlaceholderRange

    request = Request(
        request_id="test_request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
        mm_features=[
            MultiModalFeatureSpec(
                modality="image",
                identifier="0000",
                data=MultiModalKwargsItem.dummy("dummy_m"),
                mm_position=PlaceholderRange(offset=0, length=10),
            )
        ],
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (
        extract_mm_features,
    )

    mm_hashes, mm_positions = extract_mm_features(request)
    assert isinstance(mm_hashes, list)
    assert len(mm_hashes) == 1
    assert isinstance(mm_positions, list)
    assert len(mm_positions) == 1
    assert mm_positions[0].offset == 0
    assert mm_positions[0].length == 10


def test_new_request_interface():
    # protect against interface changes
    from vllm.v1.core.sched.output import NewRequestData

    assumes(NewRequestData, "req_id")
    assumes(NewRequestData, "block_ids")
    assumes(NewRequestData, "prompt_token_ids")
    assumes(NewRequestData, "sampling_params")


def test_sampling_params_interface():
    # protect against interface changes
    from vllm.sampling_params import SamplingParams

    assumes(SamplingParams, "extra_args")

    # dumb example use case in LMCache
    kv_transfer_params = {
        "lmcache.tag.user": "example_user_1",
        "lmcache.ttl": 60,
    }
    sampling_params = SamplingParams(
        extra_args={"kv_transfer_params": kv_transfer_params}
    )
    assert sampling_params.extra_args["kv_transfer_params"] == kv_transfer_params


def test_tp_interface():
    # protect against interface changes
    import inspect

    from vllm.distributed.parallel_state import get_tp_group

    sig = inspect.signature(get_tp_group)
    GroupCoordinator = sig.return_annotation

    assumes(GroupCoordinator, "broadcast", is_callable=True)
    assumes(GroupCoordinator, "broadcast_object", is_callable=True)


def test_forward_context_interface():
    # protect against interface changes
    from vllm.forward_context import ForwardContext

    assumes(ForwardContext, "no_compile_layers", is_instance_of=dict)
    assumes(ForwardContext, "virtual_engine")
    assumes(ForwardContext, "attn_metadata")


def test_scheduler_output_interface():
    # protect against interface changes
    from vllm.v1.core.sched.output import SchedulerOutput

    assumes(SchedulerOutput, "finished_req_ids")
    assumes(SchedulerOutput, "scheduled_new_reqs", is_instance_of=list)
    assumes(SchedulerOutput, "num_scheduled_tokens", is_instance_of=dict)
    assumes(SchedulerOutput, "scheduled_cached_reqs")

    from vllm.v1.core.sched.output import CachedRequestData

    assumes(CachedRequestData, "req_ids", is_instance_of=list)
    assumes(CachedRequestData, "new_block_ids", is_instance_of=list)
