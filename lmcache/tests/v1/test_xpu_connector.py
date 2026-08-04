# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from unittest.mock import patch
import random
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.gpu_connector.utils import get_dtype
from lmcache.v1.gpu_connector.xpu_connectors import (
    VLLMBufferLayerwiseXPUConnector,
    VLLMPagedMemLayerwiseXPUConnector,
    VLLMPagedMemXPUConnectorV2,
    VLLMPagedMemXPUConnectorV3,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

pytestmark = pytest.mark.skipif(
    torch_device_type != "xpu",
    reason="XPU-only tests",
)

if torch_device_type == "xpu":
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        lmc_ops = None
else:
    lmc_ops = None

# Mock c_ops when not available
if lmc_ops is None:

    class MockEngineKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2

    class MockCOps:
        EngineKVFormat = MockEngineKVFormat
        GPUKVFormat = MockEngineKVFormat

    lmc_ops = MockCOps()

# Local
from .utils import (  # noqa: E402
    check_paged_kv_cache_equal,
    check_paged_kv_cache_equal_with_mla,
    generate_kv_cache_paged_list_tensors,
    recover_gpu_connector_states,
)


@pytest.fixture(autouse=True, scope="module")
def patch_pin_allocator():
    def fake_pin_init(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def fake_pin_close(self):
        if not self._unregistered:
            torch.xpu.synchronize()
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_allocators.pin_memory_allocator.PinMemoryAllocator.__init__",
            fake_pin_init,
        ),
        patch(
            "lmcache.v1.memory_allocators.pin_memory_allocator.PinMemoryAllocator.close",
            fake_pin_close,
        ),
    ):
        yield


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
def test_vllm_paged_connector_v2_with_gpu_and_mla(use_gpu, engine_kv_format):
    use_mla = engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=torch_device_type,
        block_size=block_size,
        engine_kv_format=engine_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=torch_device_type,
        block_size=block_size,
        engine_kv_format=engine_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, engine_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(
        slot_mapping, device=torch_device_type, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
            )
        else:
            check_paged_kv_cache_equal(
                gpu_kv_src,
                gpu_kv_dst,
                slot_mapping,
                num_heads,
                head_size,
                engine_kv_format,
            )

    connector = VLLMPagedMemXPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=torch_device_type,
        use_mla=use_mla,
    )
    connector2 = VLLMPagedMemXPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=torch_device_type,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        recover_gpu_connector_states(connector)
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
        )
    else:
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, engine_kv_format
        )
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("num_groups", [1, 2, 3])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
def test_vllm_paged_connector_v3_with_gpu_and_mla(
    use_gpu, num_groups, engine_kv_format
):
    use_mla = engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    head_sizes = [64, 66, 66]
    dtypes = [torch.uint8, torch.bfloat16, torch.uint8]
    num_blocks = 100
    block_size = 16
    num_heads = 1 if use_mla else 8
    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    # generate kv cache tensors
    src_kv_groups: list[list] = []
    dst_kv_groups: list[list] = []
    src_kv_caches: dict[str, torch.Tensor] = {}
    dst_kv_caches: dict[str, torch.Tensor] = {}
    for i in range(num_groups):
        for groups, kv_caches in [
            (src_kv_groups, src_kv_caches),
            (dst_kv_groups, dst_kv_caches),
        ]:
            kv_group = generate_kv_cache_paged_list_tensors(
                num_blocks=num_blocks,
                device=torch_device_type,
                block_size=block_size,
                dtype=dtypes[i],
                num_layers=8,
                head_size=head_sizes[i],
                engine_kv_format=engine_kv_format,
            )
            groups.append(kv_group)
            for j, layer_tensor in enumerate(kv_group):
                kv_caches[f"{i}-{j}"] = layer_tensor

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(
        slot_mapping, device=torch_device_type, dtype=torch.int64
    )

    # Check the kv group is not the same before copying
    with pytest.raises(AssertionError):
        for i in range(num_groups):
            if use_mla:
                check_paged_kv_cache_equal_with_mla(
                    src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
                )
            else:
                check_paged_kv_cache_equal(
                    src_kv_groups[i],
                    dst_kv_groups[i],
                    slot_mapping,
                    num_heads,
                    head_sizes[i],
                    engine_kv_format,
                )

    # create metadata and init kv layer groups
    metadata = _create_metadata(use_mla, src_kv_caches, engine_kv_format)
    metadata2 = _create_metadata(use_mla, dst_kv_caches, engine_kv_format)

    # connector will copy with src_kv_groups
    connector = VLLMPagedMemXPUConnectorV3(
        metadata=metadata,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    # connector2 will copy with dst_kv_groups
    connector2 = VLLMPagedMemXPUConnectorV3(
        metadata=metadata2,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla

    # copy from src_kv_groups to memory_obj,
    # and then copy from memory_obj to dst_kv_groups
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        memory_obj = allocator.allocate(
            metadata.get_shapes(end - start), metadata.get_dtypes()
        )
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(dst_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check the kv group is same after copying
    for i in range(num_groups):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
            )
        else:
            check_paged_kv_cache_equal(
                src_kv_groups[i],
                dst_kv_groups[i],
                slot_mapping,
                num_heads,
                head_sizes[i],
                engine_kv_format,
            )
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
    ],
)
def test_layerwise_vllm_paged_connector_with_gpu(use_gpu, engine_kv_format):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=torch_device_type,
        block_size=block_size,
        engine_kv_format=engine_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=torch_device_type,
        block_size=block_size,
        engine_kv_format=engine_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, engine_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(
        slot_mapping, device=torch_device_type, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, engine_kv_format
        )

    connector = VLLMPagedMemLayerwiseXPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=torch_device_type,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, engine_kv_format
    )

    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
def test_batched_layerwise_vllm_paged_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size

    num_tokens_1 = 800
    num_tokens_2 = 500
    num_tokens_total = num_tokens_1 + num_tokens_2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping_total = random.sample(
        range(0, num_blocks * block_size), num_tokens_total
    )
    slot_mapping_total = torch.tensor(
        slot_mapping_total, device=torch_device_type, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(gpu_kv_src, gpu_kv_dst, slot_mapping_total)

    connector = VLLMPagedMemLayerwiseXPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=torch_device_type,
    )

    # from gpu to cpu
    starts_1 = []
    ends_1 = []
    memory_objs_1 = []

    for start in range(0, num_tokens_1, chunk_size):
        end = min(start + chunk_size, num_tokens_1)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_1.append(start)
        ends_1.append(end)
        memory_objs_1.append(memory_objs_multi_layer)

    memory_objs_1 = [list(row) for row in zip(*memory_objs_1, strict=False)]

    starts_2 = []
    ends_2 = []
    memory_objs_2 = []
    for start in range(num_tokens_1, num_tokens_total, chunk_size):
        end = min(start + chunk_size, num_tokens_total)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_2.append(start)
        ends_2.append(end)
        memory_objs_2.append(memory_objs_multi_layer)

    memory_objs_2 = [list(row) for row in zip(*memory_objs_2, strict=False)]

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_2 = connector.batched_from_gpu(
        memory_objs_2,
        starts_2,
        ends_2,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=False,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator_1)
        next(mem_obj_generator_2)

    # from cpu to gpu
    mem_obj_consumer_1 = connector.batched_to_gpu(
        starts_1,
        ends_1,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=False,
    )
    mem_obj_consumer_2 = connector.batched_to_gpu(
        starts_2,
        ends_2,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)
    for layer_id in range(num_layers):
        mem_obj_consumer_1.send(memory_objs_1[layer_id])
        mem_obj_consumer_2.send(memory_objs_2[layer_id])
    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs_1:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    for mem_obj_multi_layer in memory_objs_2:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src,
        gpu_kv_dst,
        slot_mapping_total,
        num_heads,
        head_size,
    )

    allocator.close()


@pytest.mark.skip(
    reason=(
        "Requires vLLM blending runtime with a registered blender instance "
        "for 'vllm-instance' (LMCBlenderBuilder)."
    )
)
@pytest.mark.parametrize("use_gpu", [True])
def test_layerwise_vllm_buffer_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(
        slot_mapping, device=torch_device_type, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    connector = VLLMBufferLayerwiseXPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        dtype=dtype,
        device=torch_device_type,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_2TD
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    allocator.close()


def test_vllm_paged_connector_v2_to_gpu_bench(benchmark):
    """
    VLLMPagedMemXPUConnectorV2.to_gpu() micro-benchmark.

    This test is to measure the performance of
    VLLMPagedMemXPUConnectorV2.to_gpu() when both KV caches and
    memobject are on GPU.

    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size

    chunk_size = 256

    allocator = GPUMemoryAllocator(1024 * 1024 * 1024, torch_device_type)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks, torch_device_type, block_size
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), chunk_size)
    slot_mapping = torch.tensor(
        slot_mapping, device=torch_device_type, dtype=torch.int64
    )

    connector = VLLMPagedMemXPUConnectorV2(hidden_dim, num_layers)
    shape = connector.get_shape(chunk_size)
    memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
    connector.from_gpu(
        memory_obj,
        0,
        chunk_size,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        offset=0,
    )
    recover_gpu_connector_states(connector)
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
    benchmark.pedantic(
        connector.to_gpu,
        args=(memory_obj, 0, chunk_size),
        kwargs={
            "kvcaches": gpu_kv_dst,
            "slot_mapping": slot_mapping,
            "offset": 0,
        },
        rounds=100,
        iterations=1000,
        warmup_rounds=10,
    )
    allocator.free(memory_obj)
    assert allocator.memcheck()

    allocator.close()


def _create_metadata(use_mla, kv_caches, engine_kv_format):
    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    num_heads = 1 if use_mla else 8
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, num_heads, 128),
        use_mla=use_mla,
    )
    kv_list = list(kv_caches.values())
    metadata.kv_layer_groups_manager = KVLayerGroupsManager(
        kv_list,
        engine_kv_formats=[engine_kv_format] * len(kv_list),
    )
    return metadata
