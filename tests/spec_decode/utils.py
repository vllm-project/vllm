
from typing import List
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
import time


def get_outputs(generator, prompts, sampling_params):
    for llm in generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        del llm
    return outputs


def get_tokens_and_text(outputs):
    all_text = []
    all_token_ids = []
    for request_output in outputs:
        for completion in request_output.outputs:
            all_text.append(completion.text)
            all_token_ids.append(completion.token_ids)
    return all_text, all_token_ids


def wait_for_gpu_memory_to_clear(devices: List[int],
                                 threshold_bytes: int,
                                 timeout_s: float = 120) -> None:
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    nvmlInit()
    start_time = time.time()
    while True:
        output = {}
        output_raw = {}
        for device in devices:
            dev_handle = nvmlDeviceGetHandleByIndex(device)
            mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
            gb_used = mem_info.used / 2**30
            output_raw[device] = gb_used
            output[device] = f'{gb_used:.02f}'

        print('gpu memory used (GB): ', end='')
        for k, v in output.items():
            print(f'{k}={v}; ', end='')
        print('')
        if all(v <= (threshold_bytes / 2**30) for v in output_raw.values()):
            break

        if time.time() - start_time >= timeout_s:
            raise ValueError(f'Memory of devices {devices=} not free after '
                             f'{timeout_s=} ({threshold_bytes/2**30=})')

        time.sleep(5)
