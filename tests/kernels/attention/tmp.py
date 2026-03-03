import torch
from vllm.utils.deep_gemm import _lazy_init

with torch.no_grad():
    data = torch.load("/root/eagle/vllm-dev-agentic/fp8_paged_mqa_logits_inputs.pt")
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            print(f"Data[{i}]: shape={data[i].shape}, dtype={data[i].dtype}, device={data[i].device}, stride={data[i].stride()}")
        else:
            print(f"Data[{i}]: type={type(data[i])}, value={data[i]}")
    args = data[:-1]
    clean_logits = data[-1]
    (
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    ) = args

    print(f"context_lens: shape={context_lens.shape}, dtype={context_lens.dtype}, device={context_lens.device}, stride={context_lens.stride()}, values={context_lens}")
    print(f"block_tables: shape={block_tables.shape}, dtype={block_tables.dtype}, device={block_tables.device}, stride={block_tables.stride()}, values={block_tables.tolist()}")

    _lazy_init()
    from vllm.utils.deep_gemm import _fp8_paged_mqa_logits_impl, _get_paged_mqa_logits_metadata_impl
    props = torch.cuda.get_device_properties(context_lens.device)
    sm_count = props.multi_processor_count
    num_sms = sm_count
    num_sms = 148
    new_metadata = _get_paged_mqa_logits_metadata_impl(context_lens, 64, num_sms)
    print(f"New metadata: shape={new_metadata.shape}, dtype={new_metadata.dtype}, device={new_metadata.device}, stride={new_metadata.stride()}, values={new_metadata.tolist()}")
    print(f"Schedule metadata: shape={schedule_metadata.shape}, dtype={schedule_metadata.dtype}, device={schedule_metadata.device}, stride={schedule_metadata.stride()}, values={schedule_metadata.tolist()}")
    assert _fp8_paged_mqa_logits_impl is not None
    print("Number of SMs:", num_sms)
    print("Running _fp8_paged_mqa_logits_impl with the loaded data...")
    result = _fp8_paged_mqa_logits_impl(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            new_metadata,
            max_model_len, 
            clean_logits=clean_logits)
    print("Result:", result)