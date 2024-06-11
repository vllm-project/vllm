.. _debugging:

Debugging Tips
===============

Debugging hang/crash issues
---------------------------

When an vLLM instance hangs or crashes, it is very difficult to debug the issue. But wait a minute, it is also possible that vLLM is doing something that indeed takes a long time:

- Downloading a model: do you have the model already downloaded in your disk? If not, vLLM will download the model from the internet, which can take a long time. Be sure to check the internet connection. It would be better to download the model first using `huggingface cli <https://huggingface.co/docs/huggingface_hub/en/guides/cli>`_ and then use the local path to the model. This way, you can isolate the issue.
- Loading the model from disk: if the model is large, it can take a long time to load the model from disk. Please take care of the location you store the model. Some clusters have shared filesystems across nodes, e.g. distributed filesystem or network filesystem, which can be slow. It would be better to store the model in a local disk. In addition, please also watch the CPU memory usage. When the model is too large, it might take much CPU memory, which can slow down the operating system because it needs to frequently swap memory between the disk and the memory.
- Tensor parallel inference: if the model is too large to fit in a single GPU, you might want to use tensor parallelism to split the model across multiple GPUs. In that case, every process will read the whole model and split it into chunks, which makes the disk reading time even longer (proportional to the size of tensor parallelism). You can convert the model checkpoint to a sharded checkpoint using `the provided script <https://docs.vllm.ai/en/latest/getting_started/examples/save_sharded_state.html>`_ . The conversion process might take some time, but later you can load the sharded checkpoint much faster. The model loading time should remain constant regardless of the size of tensor parallelism.

If you already take care of the above issues, and the vLLM instance still hangs, with CPU and GPU utilization at near zero, it is likely that the vLLM instance is stuck somewhere. Here are some tips to help debug the issue:

- Set the environment variable ``export VLLM_LOGGING_LEVEL=DEBUG`` to turn on more logging.
- Set the environment variable ``export CUDA_LAUNCH_BLOCKING=1`` to know exactly which CUDA kernel is causing the trouble.
- Set the environment variable ``export NCCL_DEBUG=TRACE`` to turn on more logging for NCCL.
- Set the environment variable ``export VLLM_TRACE_FUNCTION=1`` . All the function calls in vLLM will be recorded. Inspect these log files, and tell which function crashes or hangs. **Note: it will generate a lot of logs and slow down the system. Only use it for debugging purposes.**

With more logging, hopefully you can find the root cause of the issue.

Here are some common issues that can cause hangs:

- The network setup is incorrect. The vLLM instance cannot get the correct IP address. You can find the log such as ``DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl``. The IP address should be the correct one. If not, override the IP address by setting the environment variable ``export VLLM_HOST_IP=your_ip_address``.
- Hardware/driver setup is incorrect. GPU communication cannot be established. You can run a sanity check script below to see if the GPU communication is working correctly.

.. code-block:: python

    # save it as `test.py`` , and run it with `NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test.py`
    # adjust `--nproc-per-node` to the number of GPUs you want to use.
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    data = torch.FloatTensor([1,] * 128).to(f"cuda:{dist.get_rank()}")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    value = data.mean().item()
    assert value == dist.get_world_size()

If the problem persists, feel free to open an `issue <https://github.com/vllm-project/vllm/issues/new/choose>`_ on GitHub, with a detailed description of the issue, your environment, and the logs.
