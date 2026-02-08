# Speculative Decoding

This document shows how to use [Speculative Decoding](https://arxiv.org/pdf/2302.01318) with vLLM to reduce inter-token latency under medium-to-low QPS (query per second), memory-bound workloads.

To train your own draft models for optimized speculative decoding, see [vllm-project/speculators](speculators.md) for seamless training and integration with vLLM.

## vLLM Speculation Methods

vLLM supports a variety of methods of speculative decoding. Model-based methods such as EAGLE, draft models, and mlp provide the best latency reduction, while simpler methods such as n-gram and and suffix decoding provide modest speedups without increasing workload during peak traffic.

- [EAGLE](eagle.md)
- [Draft Model](draft_model.md)
- [MLP](mlp.md)
- [N-Gram](n_gram.md)
- [Suffix Decoding](suffix.md)

## Lossless guarantees of Speculative Decoding

In vLLM, speculative decoding aims to enhance inference efficiency while maintaining accuracy. This section addresses the lossless guarantees of
speculative decoding, breaking down the guarantees into three key areas:

1. **Theoretical Losslessness**
   \- Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics. Floating-point errors might
   cause slight variations in output distributions, as discussed
   in [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

2. **Algorithmic Losslessness**
   \- vLLM’s implementation of speculative decoding is algorithmically validated to be lossless. Key validation tests include:

    > - **Rejection Sampler Convergence**: Ensures that samples from vLLM’s rejection sampler align with the target
    >   distribution. [View Test Code](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **Greedy Sampling Equality**: Confirms that greedy sampling with speculative decoding matches greedy sampling
    >   without it. This verifies that vLLM's speculative decoding framework, when integrated with the vLLM forward pass and the vLLM rejection sampler,
    >   provides a lossless guarantee. Almost all of the tests in [tests/spec_decode/e2e](/tests/v1/spec_decode).
    >   verify this property using [this assertion implementation](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291)

3. **vLLM Logprob Stability**
   \- vLLM does not currently guarantee stable token log probabilities (logprobs). This can result in different outputs for the
   same request across runs. For more details, see the FAQ section
   titled *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

While vLLM strives to ensure losslessness in speculative decoding, variations in generated outputs with and without speculative decoding
can occur due to following factors:

- **Floating-Point Precision**: Differences in hardware numerical precision may lead to slight discrepancies in the output distribution.
- **Batch Size and Numerical Stability**: Changes in batch size may cause variations in logprobs and output probabilities, potentially
  due to non-deterministic behavior in batched operations or numerical instability.

For mitigation strategies, please refer to the FAQ entry *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

## Known Feature Incompatibility

1. Pipeline parallelism is not composible with speculative decoding as of `vllm<=0.15.0`
2. Speculative decoding with a draft models is not supported in `vllm<=0.10.0`

## Resources for vLLM contributors

- [[vLLM Office Hours #40] Intro to Speculators](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
