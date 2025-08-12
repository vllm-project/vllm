from vllm.envs import VLLM_USE_MODELSCOPE

if VLLM_USE_MODELSCOPE:
    # Patch here, before each import happens
    import modelscope
    from packaging import version

    # patch_hub begins from modelscope>=1.18.1
    if version.parse(modelscope.__version__) <= version.parse('1.18.0'):
        raise ImportError(
            'Using vLLM with ModelScope needs modelscope>=1.18.1, please '
            'install by `pip install modelscope -U`')

    from modelscope.utils.hf_util import patch_hub

    # Patch hub to download models from modelscope to speed up.
    patch_hub()
