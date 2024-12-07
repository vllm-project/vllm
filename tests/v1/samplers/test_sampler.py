"""vLLM v1 sampler tests"""
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.metadata import SamplingMetadata

def test_sampler_n_1()->None:
    sampler=Sampler()
    sampler_output=sampler.forward(
        logits,
        sampling_metadata
    )
