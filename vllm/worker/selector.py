from vllm.platforms import current_platform


def init_worker(*args, **kwargs):
    if current_platform.is_neuron():
        from vllm.worker.neuron_worker import NeuronWorker
        return NeuronWorker(*args, **kwargs)
    elif current_platform.is_tpu():
        from vllm.worker.tpu_worker import TPUWorker
        return TPUWorker(*args, **kwargs)
    elif current_platform.is_cpu():
        from vllm.worker.cpu_worker import CPUWorker
        return CPUWorker(*args, **kwargs)
    elif current_platform.is_hpu():
        from vllm.worker.hpu_worker import HPUWorker
        return HPUWorker(*args, **kwargs)
    elif current_platform.is_openvino():
        from vllm.worker.openvino_worker import OpenVINOWorker
        return OpenVINOWorker(*args, **kwargs)
    elif current_platform.is_xpu():
        from vllm.worker.xpu_worker import XPUWorker
        return XPUWorker(*args, **kwargs)
    else:
        from vllm.worker.worker import Worker
        return Worker(*args, **kwargs)
