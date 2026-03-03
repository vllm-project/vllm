# CUDA-Agent: Reproduction

Reproduction of the paper:

> **CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation**
> Dai et al., ByteDance Seed + Tsinghua University, 2026
> arXiv: https://arxiv.org/abs/2602.24286

**Resources:**
- Paper: https://cuda-agent.github.io/static/pdf/CUDA_Agent_Arxiv_Version.pdf
- GitHub: https://github.com/BytedTsinghua-SIA/CUDA-Agent
- Dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K

---

## Paper Results (KernelBench, 250 tasks)

| Model               | Pass Rate | Faster vs. Compile | Speedup (Geo-mean) |
|---------------------|-----------|--------------------|---------------------|
| **CUDA-Agent**      | **98.8%** | **96.8%**          | **2.11x**           |
| Claude Opus 4.5     | 95.2%     | ~60%               | 1.46x               |
| Gemini 3 Pro        | 91.2%     | ~58%               | 1.42x               |
| Seed1.6 (base)      | 74.0%     | 14.1%              | 0.69x               |
| GLM 4.6             | 75.6%     | —                  | 0.57x               |
| Kimi K2             | 66.8%     | —                  | 0.66x               |

---

## Architecture Overview

The system has **three pillars**:

### Pillar 1: Scalable Data Synthesis (CUDA-Agent-Ops-6K)
```
Seed problem crawling (torch + transformers operators)
    → LLM-based combinatorial synthesis (2-5 ops combined)
    → Execution-driven filtering (correctness, determinism, anti-hacking, runtime)
    → 6,000 high-quality training tasks (0% KernelBench overlap)
```

### Pillar 2: Skill-Augmented Agent Environment
```
ReAct-style agent loop (up to 150 turns, 128k context)
    → Profile baseline → Write CUDA kernel → Compile → Verify → Profile → Iterate
SKILL.md encodes the canonical CUDA dev workflow
Anti-hacking: file permissions, functional blocking, 5-pass verification
Discrete milestone reward: -1 / +1 / +2 / +3
```

### Pillar 3: Multi-Stage RL Training
```
Stage 1: Single-turn PPO warm-up (32k context, non-agentic)
Stage 2: Actor init via Rejection Fine-Tuning (filter+SFT on trajectories)
Stage 3: Full agentic PPO (131k context, asymmetric clipping ε=0.20/0.28,
         128 H20 GPUs, 150 steps, critic pre-trained from Stage 2 rollouts)
```

---

## Repository Structure

```
cuda_agent/
├── agent_workdir/              # Standardised agent workspace (from paper's GitHub)
│   ├── SKILL.md                # Workflow constraints for the agent
│   ├── model.py                # Reference PyTorch baseline
│   ├── model_new.py            # Agent's optimised CUDA extension model
│   ├── binding.cpp             # PyBind11 module entry-point (do not modify)
│   ├── binding_registry.h      # Auto-registration header (do not modify)
│   ├── kernels/
│   │   ├── axpby.cu            # Example: axpby CUDA kernel
│   │   └── axpby_binding.cpp   # Example: PyTorch wrapper for axpby
│   └── utils/                  # Infrastructure (do not modify)
│       ├── compile.py          # Build kernels via torch.utils.cpp_extension
│       ├── compile.sh          # Shell wrapper with timing
│       ├── verification.py     # 5-pass correctness check (atol=1e-2, rtol=1e-2)
│       └── profiling.py        # Benchmark vs. torch.compile
│
├── data/
│   ├── dataset_loader.py       # Load CUDA-Agent-Ops-6K from HuggingFace
│   └── task_generator.py       # Generate task_workdir from a sample
│
├── agent/
│   ├── prompt_templates.py     # System prompt, ReAct templates, tool definitions
│   ├── environment.py          # Sandboxed env (bash/write/read + reward)
│   └── react_agent.py          # vllm-backed ReAct agent + Trajectory
│
├── training/
│   ├── reward.py               # Milestone reward + GAE advantage computation
│   ├── rft.py                  # Stage 2: Rejection Fine-Tuning
│   └── rl_trainer.py           # Stage 3: Agentic PPO + critic pre-training
│
├── evaluation/
│   └── evaluate.py             # Evaluation pipeline + paper comparison table
│
├── train.py                    # End-to-end training script (all 3 stages)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r cuda_agent/requirements.txt
```

### 2. Explore the agent workspace

```bash
# Read the task and constraints
cat cuda_agent/agent_workdir/SKILL.md

# Inspect the reference model
cat cuda_agent/agent_workdir/model.py

# (After implementing kernels) compile, verify, profile
cd cuda_agent/agent_workdir
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```

### 3. Load the dataset

```python
from cuda_agent.data import load_cuda_agent_dataset, dataset_statistics

samples = load_cuda_agent_dataset(max_samples=100)
print(dataset_statistics(samples))

# See the first task description
print(samples[0].to_task_description())
```

### 4. Run an agent episode (requires vllm server)

```bash
# Start vllm server
vllm serve BytedTsinghua-SIA/Seed1.6 --port 8000

# Run a single episode
python - <<'EOF'
from cuda_agent.agent import CUDAReActAgent, CUDAAgentEnvironment
from cuda_agent.data import load_cuda_agent_dataset
from cuda_agent.data.task_generator import TaskGenerator
from pathlib import Path

samples = load_cuda_agent_dataset(max_samples=1)
agent  = CUDAReActAgent(model="BytedTsinghua-SIA/Seed1.6")
env    = CUDAAgentEnvironment(template_dir="cuda_agent/agent_workdir")
gen    = TaskGenerator("cuda_agent/agent_workdir", "/tmp/tasks")
task   = gen.generate_task(samples[0], "demo_task")
traj   = agent.run_episode(samples[0], env, task)
print(f"Reward: {traj.final_reward.name}, Speedup: {traj.speedup_vs_compile:.2f}x")
EOF
```

### 5. Full training pipeline

```bash
python -m cuda_agent.train \
    --model-path /path/to/seed1.6 \
    --output-dir ./checkpoints \
    --stage all \
    --max-samples 6000 \
    --vllm-host http://localhost:8000 \
    --cuda-arch 9.0
```

### 6. Evaluate and compare to paper

```bash
python -m cuda_agent.evaluation.evaluate \
    --model BytedTsinghua-SIA/Seed1.6 \
    --max-samples 100 \
    --output-dir ./eval_results \
    --print-paper-table
```

---

## Reward Function

| Reward | Milestone | Condition |
|--------|-----------|-----------|
| **-1** | Correctness failure | Compilation error or wrong output |
| **+1** | Correct, no speedup | Output correct, but no speedup milestone |
| **+2** | Faster than eager | Correct + faster than `torch.eager` |
| **+3** | Faster than compile | Correct + ≥5% speedup over `torch.compile` |

The milestone-based discrete reward substantially outperforms raw speedup:
- Discrete reward:    96.8% faster-than-compile, 2.11x speedup
- Continuous reward:  60.4% faster-than-compile, 1.25x speedup

---

## Multi-Stage Training

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: Single-Turn PPO (32k ctx, non-agentic)     │
│   ↓ base model learns basic CUDA generation          │
├─────────────────────────────────────────────────────┤
│ Stage 2: Rejection Fine-Tuning                       │
│   ↓ filters trajectories, SFT actor                  │
│   ↓ constrains entropy, prevents collapse            │
├─────────────────────────────────────────────────────┤
│ Stage 3: Agentic PPO (131k ctx, 150 turns)          │
│   ε_lower=0.2, ε_higher=0.28 (asymmetric clipping) │
│   128 H20 GPUs, 1024 global batch, 150 steps        │
└─────────────────────────────────────────────────────┘
```

---

## Citation

```bibtex
@article{dai2026cudaagent,
  title   = {CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation},
  author  = {Dai, Weinan and Wu, Hanlin and Yu, Qiying and Gao, Huan-ang and
             Li, Jiahao and Jiang, Chengquan and Lou, Weiqiang and Song, Yufan and
             Yu, Hongli and Chen, Jiaze and Ma, Wei-Ying and Zhang, Ya-Qin and
             Liu, Jingjing and Wang, Mingxuan and Liu, Xin and Zhou, Hao},
  journal = {arXiv preprint arXiv:2602.24286},
  year    = {2026}
}
```
