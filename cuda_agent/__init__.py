"""
cuda_agent — Reproduction of "CUDA Agent: Large-Scale Agentic RL for
High-Performance CUDA Kernel Generation" (ByteDance + Tsinghua, 2026).

arXiv: https://arxiv.org/abs/2602.24286
GitHub: https://github.com/BytedTsinghua-SIA/CUDA-Agent
Dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K

Three-pillar architecture:
  1. Scalable data synthesis  → cuda_agent.data
  2. Skill-augmented agent    → cuda_agent.agent
  3. Multi-stage RL training  → cuda_agent.training

Evaluation:
  → cuda_agent.evaluation
"""

__version__ = "0.1.0"
__paper__ = "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation"
__authors__ = "Dai et al., ByteDance Seed + Tsinghua University, 2026"
