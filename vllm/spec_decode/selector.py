from vllm.platforms import current_platform

if current_platform.is_neuron():
    from vllm.worker.neuron_model_runner import (  # noqa: F401
        ModelInputForNeuron as ModelInputCls)
    from vllm.worker.neuron_model_runner import (  # noqa: F401
        NeuronModelRunner as ModelRunnerCls)
    from vllm.worker.neuron_worker import (  # noqa: F401
        NeuronWorker as WorkerCls)
    DEVICE_TYPE = "neuron"
elif current_platform.is_hpu():
    from vllm.worker.hpu_model_runner import (  # noqa: F401
        HPUModelRunner as ModelRunnerCls)
    from vllm.worker.hpu_model_runner import (  # noqa: F401
        ModelInputForHPUWithSamplingMetadata as ModelInputCls)
    from vllm.worker.hpu_worker import HPUWorker as WorkerCls  # noqa: F401
    DEVICE_TYPE = "hpu"
elif current_platform.is_openvino():
    from vllm.worker.openvino_model_runner import (  # noqa: F401
        ModelInput as ModelInputCls)
    from vllm.worker.openvino_model_runner import (  # noqa: F401
        OpenVINOModelRunner as ModelRunnerCls)
    from vllm.worker.openvino_worker import (  # noqa: F401
        OpenVINOWorker as WorkerCls)
    DEVICE_TYPE = "openvino"
elif current_platform.is_cpu():
    from vllm.worker.cpu_model_runner import (  # noqa: F401
        CPUModelRunner as ModelRunnerCls)
    from vllm.worker.cpu_model_runner import (  # noqa: F401
        ModelInputForCPUWithSamplingMetadata as ModelInputCls)
    from vllm.worker.cpu_worker import CPUWorker as WorkerCls  # noqa: F401
    DEVICE_TYPE = "cpu"
elif current_platform.is_tpu():
    from vllm.worker.tpu_model_runner import (  # noqa: F401
        ModelInputForTPU as ModelInputCls)
    from vllm.worker.tpu_model_runner import (  # noqa: F401
        TPUModelRunner as ModelRunnerCls)
    from vllm.worker.tpu_worker import TPUWorker as WorkerCls  # noqa: F401
    DEVICE_TYPE = "tpu"
elif current_platform.is_xpu():
    from vllm.worker.xpu_model_runner import (  # noqa: F401
        ModelInputForXPUWithSamplingMetadata as ModelInputCls)
    from vllm.worker.xpu_model_runner import (  # noqa: F401
        XPUModelRunner as ModelRunnerCls)
    from vllm.worker.xpu_worker import XPUWorker as WorkerCls  # noqa: F401
    DEVICE_TYPE = "xpu"
else:
    from vllm.worker.model_runner import (  # noqa: F401
        ModelInputForGPUWithSamplingMetadata as ModelInputCls)
    from vllm.worker.model_runner import (  # noqa: F401
        ModelRunner as ModelRunnerCls)
    from vllm.worker.worker import Worker as WorkerCls  # noqa: F401
    DEVICE_TYPE = "cuda"
