"""Compare the outputs of HF and vLLM for BART models using greedy sampling.

Run `pytest tests/models/test_bart.py`.
"""
from typing import List, Optional, Tuple

from vllm.utils import is_cpu

if not is_cpu():
    # CPU backend is not currently supported with encoder/decoder models
    # skip test definitions entirely to avoid importing GPU kernel libs
    # (xFormers, etc.)

    import pytest

    from vllm.sequence import SampleLogprobs

    from ..conftest import DecoderPromptType
    from .utils import check_logprobs_close

    MODELS = ["facebook/bart-base", "facebook/bart-large-cnn"]

    def vllm_to_hf_output(
        vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
        decoder_prompt_type: DecoderPromptType,
    ):
        """Sanitize vllm output to be comparable with hf output."""
        output_ids, output_str, out_logprobs = vllm_output

        hf_output_str = output_str + "</s>"
        if decoder_prompt_type == DecoderPromptType.NONE:
            hf_output_str = "<s>" + hf_output_str

        return output_ids, hf_output_str, out_logprobs

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("dtype", ["float", "bfloat16"])
    @pytest.mark.parametrize("max_tokens", [64])
    @pytest.mark.parametrize("num_logprobs", [5])
    @pytest.mark.parametrize("decoder_prompt_type", list(DecoderPromptType))
    def test_models(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts,
        model: str,
        dtype: str,
        max_tokens: int,
        num_logprobs: int,
        decoder_prompt_type: DecoderPromptType,
    ) -> None:
        '''
        Test the vLLM BART model for a variety of encoder/decoder input prompts,
        by validating it against HuggingFace (HF) BART.

        Arguments:

        * hf_runner: HuggingFace (HF) test model runner
        * vllm_runner: vLLM test model runner
        * example_encoder_decoder_prompts: test fixture which provides a 
                                           dictionary of dummy prompts
        * model: the HF ID of the specific BART variant under test
        * dtype: the tensor datatype to employ
        * max_tokens
        * num_logprobs
        * decoder_prompt_type: key into the example_encoder_decoder_prompts
                               dictionary; selects specific encoder/decoder
                               prompt scenarios to test

        A note on using HF BART as a baseline for validating vLLM BART,
        specifically when the decoder prompt is None. 
        
        The HF GenerationMixin's default behavior is to force the first
        decoded token to be <BOS> if the prompt does not already contain
        <BOS> (this is accomplished using a logit
        processor setting.)
        
        So when we use HF BART as our baseline for comparison, note that
        when the user provides a request with a None decoder prompt
        (i.e. a singleton encoder prompt, or else an explicit encoder/
        decoder prompt with the decoder sub-prompt set to None), HF and
        vLLM handle this in different ways:
        
        * HF will (1) tokenize the None prompt as an empty token-list, 
          (2) append <decoder-start-token> to the beginning, yielding
          [<decoder-start-token>], (3) pass this token list to the model, and
          then (4) after computing logits during prefill, override the model
          logits & force <BOS> to be the first generated token.
        
        * vLLM will (1) tokenize the None prompt as [<BOS>], (2) append decoder-
          start-token to the beginning, yielding [<decoder-start-token><BOS>],
          (3) pass these tokens to the model & proceed with generation.
        
        The net effect is that compared to vLLM, the list of HF *decoded* tokens
        will contain one more initial <BOS> than the vLLM generated tokens,
        because vLLM's <BOS> token is injected into the prompt rather than into
        the generated output. This is in spite of the fact that overall, the
        complete sequences (prompt + decoded tokens) produced by vLLM will match
        HF.
        
        So when we use HF decoded token output to validate vLLM's decoded token
        output, the testing process must account for the difference in decoded
        token sequences between vLLM and HF specifically in the
        decoder-prompt-is-None case. 
        
        One option is to disable the logit processor feature that forces the
        <BOS> token to be decoded (forced_bos_token_id = None), eliminating
        the problem entirely. However this is not "normal" BART usage.
        
        The other option is - only in the decoder-prompt-is-None case - to
        discard the first decoded token from the HF output before comparing it
        to vLLM.

        To that end, when testing the scenario where the decoder prompt is None
        (and only in that one scenario), this test skips the first HF decoded
        token during the process of validating the vLLM decoded output.
        '''

        test_case_prompts = example_encoder_decoder_prompts[
            decoder_prompt_type]

        # Configuration settings for HF baseline
        hf_kwargs = {
            "top_k": None,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "length_penalty": 1.0,
            "early_stopping": False,
            "no_repeat_ngram_size": None,
            "min_length": 0
        }

        with hf_runner(model, dtype=dtype,
                       is_encoder_decoder_model=True) as hf_model:
            hf_outputs = (
                hf_model.generate_encoder_decoder_greedy_logprobs_limit(
                    test_case_prompts,
                    max_tokens,
                    num_logprobs,
                    **hf_kwargs,
                ))

        # Note: currently encoder/decoder models are only compatible with
        # enforce_eager=True. Normally this is not a problem because
        # for encoder/decoder models vLLM will
        # default to enforce_eager=True if enforce_eager
        # is left unspecified. However, the
        # VllmRunner test fixture (which wraps around the LLM class) defaults to
        # enforce_eager=False (a behavior which a number of already-exisitng
        # decoder-only unit tests expect), so when testing an encoder/decoder
        # model we must explicitly specify enforce_eager=True in the VllmRunner
        # constructor.
        with vllm_runner(model, dtype=dtype, enforce_eager=True) as vllm_model:
            vllm_outputs = vllm_model.generate_encoder_decoder_greedy_logprobs(
                test_case_prompts, max_tokens, num_logprobs)

        hf_skip_tokens = (1 if decoder_prompt_type == DecoderPromptType.NONE
                          else 0)

        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, decoder_prompt_type)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
            num_outputs_0_skip_tokens=hf_skip_tokens,
        )
