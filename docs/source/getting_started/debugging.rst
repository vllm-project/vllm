.. _debugging:

Debugging Tips
===============

Debugging hang/crash issues
---------------------------

When an vLLM instance hangs or crashes, it is very difficult to debug the issue. But wait a minute, it is also possible that vLLM is doing something that indeed takes a long time:

- **Downloading a model**: Do you have the model already downloaded in your disk? If not, vLLM will download the model from the internet, which can take a long time. Be sure to check the internet connection. It would be better to download the model first using `huggingface-cli <https://huggingface.co/docs/huggingface_hub/en/guides/cli>`_ and then use the local path to the model. This way, you can isolate the issue.
- **Loading the model from disk**: If the model is large, it can take a long time to load the model from disk. Please take care of the location you store the model. Some clusters have shared filesystems across nodes, e.g. distributed filesystem or network filesystem, which can be slow. It would be better to store the model in a local disk. In addition, please also watch the CPU memory usage. When the model is too large, it might take much CPU memory, which can slow down the operating system because it needs to frequently swap memory between the disk and the memory.
- **Tensor parallel inference**: If the model is too large to fit in a single GPU, you might want to use tensor parallelism to split the model across multiple GPUs. In that case, every process will read the whole model and split it into chunks, which makes the disk reading time even longer (proportional to the size of tensor parallelism). You can convert the model checkpoint to a sharded checkpoint using `the provided script <https://docs.vllm.ai/en/latest/getting_started/examples/save_sharded_state.html>`_ . The conversion process might take some time, but later you can load the sharded checkpoint much faster. The model loading time should remain constant regardless of the size of tensor parallelism.

If you have already taken care of the above issues, but the vLLM instance still hangs, with CPU and GPU utilization at near zero, it is likely that the vLLM instance is stuck somewhere. Here are some tips to help debug the issue:

- Set the environment variable ``export VLLM_LOGGING_LEVEL=DEBUG`` to turn on more logging.
- Set the environment variable ``export CUDA_LAUNCH_BLOCKING=1`` to know exactly which CUDA kernel is causing the trouble.
- Set the environment variable ``export NCCL_DEBUG=TRACE`` to turn on more logging for NCCL.
- Set the environment variable ``export VLLM_TRACE_FUNCTION=1``. All the function calls in vLLM will be recorded. Inspect these log files, and tell which function crashes or hangs.

With more logging, hopefully you can find the root cause of the issue.

If it crashes, and the error trace shows somewhere around ``self.graph.replay()`` in ``vllm/worker/model_runner.py``, it is a cuda error inside cudagraph. To know the particular cuda operation that causes the error, you can add ``--enforce-eager`` to the command line, or ``enforce_eager=True`` to the :class:`~vllm.LLM` class, to disable the cudagraph optimization. This way, you can locate the exact cuda operation that causes the error.

Here are some common issues that can cause hangs:

- **Incorrect network setup**: The vLLM instance cannot get the correct IP address if you have complicated network config. You can find the log such as ``DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl``. The IP address should be the correct one. If not, override the IP address by setting the environment variable ``export VLLM_HOST_IP=your_ip_address``. You might also need to set ``export NCCL_SOCKET_IFNAME=your_network_interface`` and ``export GLOO_SOCKET_IFNAME=your_network_interface`` to specify the network interface for the IP address.
- **Incorrect hardware/driver**: GPU/CPU communication cannot be established. You can run the following sanity check script to see if the GPU/CPU communication is working correctly.

.. code-block:: python

    # Test PyTorch NCCL
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    data = torch.FloatTensor([1,] * 128).to("cuda")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    value = data.mean().item()
    world_size = dist.get_world_size()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch NCCL is successful!")

    # Test PyTorch GLOO
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    cpu_data = torch.FloatTensor([1,] * 128)
    dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
    value = cpu_data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch GLOO is successful!")

    # Test vLLM NCCL, with cuda graph
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
    pynccl.disabled = False

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        data.fill_(1)
        pynccl.all_reduce(data, stream=s)
        value = data.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL is successful!")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=g, stream=s):
        pynccl.all_reduce(data, stream=torch.cuda.current_stream())

    data.fill_(1)
    g.replay()
    torch.cuda.current_stream().synchronize()
    value = data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL with cuda graph is successful!")

    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()

.. tip::

    Save the script as ``test.py``.
    
    If you are testing in a single-node, run it with ``NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test.py``, adjust ``--nproc-per-node`` to the number of GPUs you want to use.
    
    If you are testing with multi-nodes, run it with ``NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test.py``. Adjust ``--nproc-per-node`` and ``--nnodes`` according to your setup. Make sure ``MASTER_ADDR``:
  
    - is the correct IP address of the master node
    - is reachable from all nodes
    - is set before running the script.

    If the script runs successfully, you should see the message ``sanity check is successful!``.

If the problem persists, feel free to `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_, with a detailed description of the issue, your environment, and the logs.

Some known issues:

- In ``v0.5.2``, ``v0.5.3``, and ``v0.5.3.post1``, there is a bug caused by `zmq <https://github.com/zeromq/pyzmq/issues/2000>`_ , which can cause hangs at a low probability (once in about 20 times, depending on the machine configuration). The solution is to upgrade to the latest version of ``vllm`` to include the `fix <https://github.com/vllm-project/vllm/pull/6759>`_ .

.. warning::

    After you find the root cause and solve the issue, remember to turn off all the debugging environment variables defined above, or simply start a new shell to avoid being affected by the debugging settings. If you don't do this, the system might be slow because many debugging functionalities are turned on.
