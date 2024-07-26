'''
Worker-related helper functions.
'''

from vllm.worker.model_runner import GPUModelRunnerBase


def assert_enc_dec_mr_supported_scenario(
    enc_dec_mr: GPUModelRunnerBase, ) -> None:
    '''
    Asserted that the provided encoder/decoder model runner instance reflects
    a supported scenario.
    '''

    if enc_dec_mr.cache_config.enable_prefix_caching:
        raise NotImplementedError("Prefix caching is currently not "
                                  "supported with encoder/decoder "
                                  "models.")

    if enc_dec_mr.sliding_window is not None:
        raise NotImplementedError("Sliding-window attention is currently "
                                  "not supported with encoder/decoder "
                                  "models.")

    if enc_dec_mr.scheduler_config.chunked_prefill_enabled:
        raise NotImplementedError("Chunked prefill is currently not "
                                  "supported with encoder/decoder "
                                  "models.")

    if not enc_dec_mr.model_config.enforce_eager:
        raise NotImplementedError("CUDAGraph is currently not "
                                  "supported with encoder/decoder "
                                  "models.")

    if getattr(enc_dec_mr.model_config.hf_config, 'attn_logit_softcapping',
               None) is not None:
        raise NotImplementedError(
            "Models with logits_soft_cap "
            "require FlashInfer backend, which is "
            "currently not supported for encoder/decoder "
            "models.")

    if enc_dec_mr.lora_config is not None:
        raise NotImplementedError("LoRA is currently not currently "
                                  "supported with encoder/decoder "
                                  "models.")

    if enc_dec_mr.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Pipeline parallelism is not "
                                  "currently supported with "
                                  "encoder/decoder models.")

    if enc_dec_mr.multimodal_config is not None:
        raise NotImplementedError("Multimodal is not currently "
                                  "supported with encoder/decoder "
                                  "models.")

    if enc_dec_mr.scheduler_config.num_lookahead_slots > 0:
        raise NotImplementedError("Speculative decoding is not "
                                  "currently supported with encoder/"
                                  "decoder models.")
