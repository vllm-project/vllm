.. _supported_hardware_for_quantization:

Supported Hardware for Quantization Kernels
===========================================

The table below shows the compatibility of various quantization implementations with different hardware platforms in vLLM:

=====================  ======  =======  =======  =====  ======  =======  =========  =======  ==============  ==========
Implementation         Volta   Turing   Ampere   Ada    Hopper  AMD GPU  Intel GPU  x86 CPU  AWS Inferentia  Google TPU
=====================  ======  =======  =======  =====  ======  =======  =========  =======  ==============  ==========
AWQ                    ❌      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
GPTQ                   ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
Marlin (GPTQ/AWQ/FP8)  ❌      ❌       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
INT8 (W8A8)            ❌      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
FP8 (W8A8)             ❌      ❌       ❌       ✅     ✅      ❌        ❌         ❌       ❌              ❌
AQLM                   ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
bitsandbytes           ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
DeepSpeedFP            ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
GGUF                   ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
SqueezeLLM             ✅      ✅       ✅       ✅     ✅      ❌        ❌         ❌       ❌              ❌
=====================  ======  =======  =======  =====  ======  =======  =========  =======  ==============  ==========

Notes:
^^^^^^

- Volta refers to SM 7.0, Turing to SM 7.5, Ampere to SM 8.0/8.6, Ada to SM 8.9, and Hopper to SM 9.0.
- "✅" indicates that the quantization method is supported on the specified hardware.
- "❌" indicates that the quantization method is not supported on the specified hardware.

Please note that this compatibility chart may be subject to change as vLLM continues to evolve and expand its support for different hardware platforms and quantization methods.

For the most up-to-date information on hardware support and quantization methods, please check the `quantization directory <https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization>`_ or consult with the vLLM development team.
