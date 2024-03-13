import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

PROMPT = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''


@pytest.mark.parametrize("model", "meta-llama/Llama-2-7b-hf")
@pytest.mark.parametrize("prompt", PROMPT)
@pytest.mark.parametrize("stop", [' ', 'for'])
def test_generate_stop(model, prompt, stop):
    engine_args = EngineArgs(model=model, enable_prefix_caching=True)
    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(stop=stop)
    engine.add_request("0", prompt, sampling_params)
