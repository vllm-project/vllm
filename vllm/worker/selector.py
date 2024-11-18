from vllm.platforms import current_platform

if current_platform.is_neuron():
    from vllm.worker.neuron_worker import NeuronWorker as WorkerCls
elif current_platform.is_hpu():
    from vllm.worker.hpu_worker import HPUWorker as WorkerCls  # type: ignore
elif current_platform.is_cpu():
    from vllm.worker.cpu_worker import CPUWorker as WorkerCls  # type: ignore
elif current_platform.is_tpu():
    from vllm.worker.tpu_worker import TPUWorker as WorkerCls  # type: ignore
elif current_platform.is_xpu():
    from vllm.worker.xpu_worker import XPUWorker as WorkerCls  # type: ignore
else:
    from vllm.worker.worker import Worker as WorkerCls  # type: ignore


def init_worker(*args, **kwargs):
    return WorkerCls(*args, **kwargs)
