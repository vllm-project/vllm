# dLLM plugin for vLLM \- current state

dLLMs are presenting a new competitive frontier for fast inference in production, threatening to completely surpass the smaller range AR LLM class (1B to 32B).

In response, various RedHat associates started discussing this matter, mostly from three main groups:

- AIPCC Model Validation  
- OCTO  
- Inference Engineering (previously NeuralMagic)

During those discussions, we reached the following conclusions:

- dLLMs are a good fit for vLLM  
- However, vLLM contributions directly to the upstream are tough  
- Instead, we could create an independent dLLM plugin under vllm-project  
- Our goal is to merge dLLM support into the vLLM upstream, but only after the dLLM plugin matures and gains community trust

# Alignment between vLLM and dLLMs

Modern dLLMs require the following basic features:

- KV-caching & Prefix-caching  
- Stepwise continuous batching  
- Multi-token decoding  
- Draft context lifecycle management

vLLM supports all of these features built-in, which makes it a great choice for the implementation of dLLMs.

# Current goals

- Next month, the **2026 RedHat summit** content and announcements will be decided. We want to have **a minimal demonstration** of dLLM plugin in time for that.  
- There are only a few **popular open-source dLLMs** (available on huggingface). We want to **support all of them**.  
- Many **vLLM features** will require to be adapted for dLLMs. We want to align **dLLM compatibility** with most vLLM features.  
- We want to **publish a technical report** with the capabilities of dLLMs on vLLM when compared to AR LLMs on vLLM, and dLLMs on other inference engines.  
- Engage with the **community and customers**.

# Technical Roadmap

- MVP implementation, with [LLaDA2.0](https://huggingface.co/collections/inclusionAI/llada-20) inference  
- LLaDA2.0 benchmarking vs SGlang technical report  
- Grammar & Structured outputs support  
- Block Kernel Optimization  
- Implement more architectures ([WeDLM](https://huggingface.co/models?other=wedlm), [SDAR](https://huggingface.co/JetLM/models), [Fast-dLLMv2](https://huggingface.co/collections/Efficient-Large-Model/fast-dllm), [LLaDA2.1](https://huggingface.co/collections/inclusionAI/llada21), [CDLM](https://huggingface.co/minseo25/models), [NBDiff](https://huggingface.co/yuchuantian/NBDiff-7B-Instruct), etc)  
- Wider benchmarking report with more models and frameworks (WeDLM engine, dInfer, Fast-dLLM engine, LMdeploy, Ollama, etc)  
- Prefix Caching with semi-causal (non-triangular) attention masks

# Model prioritization

1. [LLaDA2.0](https://huggingface.co/collections/inclusionAI/llada-20)  
2. [LLaDA2.1](https://huggingface.co/collections/inclusionAI/llada21)  
3. [SDAR](https://huggingface.co/JetLM/models)  
4. [WeDLM](https://huggingface.co/models?other=wedlm)  
5. [Fast-dLLMv2](https://huggingface.co/collections/Efficient-Large-Model/fast-dllm)  
6. [NBDiff](https://huggingface.co/yuchuantian/NBDiff-7B-Instruct) (NTH)

# General features

- \[MVP\]  
  - Plugin scheduler  
  - Plugin worker  
  - Composable remasking architecture  
  - Initial model implementation (LLaDA2.0)  
- Model level grammar & structured outputs (PDA mask)  
- Block kernel optimization  
- Prefix Caching with semi-causal (non-triangular) attention masks

# Method specific features

- Position Remasking  
  - Samplers  
    - Top-K  
    - Threshold  
    - Top-P (uncertainty budget)  
    - Min-P (linear dynamic threshold)  
    - Top-A (quadratic dynamic threshold)  
  - Metrics  
    - Random (baseline)  
    - Causal (baseline)  
    - Confidence ([SDAR](https://arxiv.org/abs/2510.06303), [LLaDA2.0](https://arxiv.org/abs/2512.15745))  
    - Margin ([TWPB](https://arxiv.org/abs/2502.06768))  
    - Entropy ([Dream](https://arxiv.org/abs/2508.15487))  
- Stream Decoding ([WeDLM](https://arxiv.org/abs/2512.22737) §5)  
- Topological Reordering ([WeDLM](https://arxiv.org/abs/2512.22737) §4.1)  
- Token Editing (T2T)  
  - [LLaDA2.1](https://arxiv.org/abs/2602.08676)  
- Intra-Block KV cache  
  - [Fast-dLLMv2](https://arxiv.org/abs/2509.26328)  
    - DualCache ([Fast-dLLMv1](https://arxiv.org/abs/2505.22618))  
  - [FOCUS](https://arxiv.org/abs/2601.23278)  
- Advanced Grammar & Structured outputs ([LAVE](https://arxiv.org/abs/2602.00612))

# User facing features

- Per request sampling param (K/Threshold/Budget/Scalar)  
  - Follow up: Automatic sampling param classification with internal semantic routing  
- Draft streaming  
  - Position+Delta+State (decoded/remasked/redecoded)  
- Per request debug info  
  - Draft history per position  
  - Draft history per step  
- Loglikelihoods (classification)  
  - Monte carlo samples  
- Logprobs  
  - Trajectory/Sampling order aware, non-trivial by definition\!