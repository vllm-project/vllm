"""
agent/prompt_templates.py

Prompt templates for the CUDA-Agent ReAct-style agent.

The agent uses a ReAct (Reasoning + Acting) paradigm:
  - Thought: internal reasoning step
  - Action: tool call (bash, write_file, read_file)
  - Observation: tool output

Paper details:
  - Context window: up to 128k tokens during training, 200k at inference
  - Max turns: 150 (training), 200 (evaluation)
  - Base model: Seed1.6 (23B active params MoE)
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert CUDA kernel engineer. Your task is to accelerate a PyTorch \
model by replacing its operations with custom CUDA kernels.

You operate in an agent loop: you reason, call tools, observe the results, \
and iterate until the task is complete.

## Environment
Your working directory contains:
- `model.py`          — Reference PyTorch baseline (DO NOT MODIFY)
- `model_new.py`      — Your optimized implementation (MODIFY THIS)
- `SKILL.md`          — Workflow constraints and optimization guidelines
- `kernels/`          — Write all .cu and _binding.cpp files here
- `utils/compile.sh`  — Compiles your kernels: `bash utils/compile.sh`
- `utils/verification.py` — Checks correctness (DO NOT MODIFY)
- `utils/profiling.py`    — Benchmarks performance (DO NOT MODIFY)

## Mandatory Workflow (from SKILL.md)
1. Read `model.py` to understand the computation.
2. Read `SKILL.md` for optimization constraints.
3. Profile the baseline: `python3 -m utils.profiling --single-run baseline compiled`
4. Implement CUDA kernels in `kernels/*.cu` and `kernels/*_binding.cpp`.
5. Update `model_new.py` to call `cuda_extension.*`.
6. Compile: `TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh`
7. Verify: `python3 -m utils.verification`
8. Profile: `python3 -m utils.profiling`
9. Iterate until ≥5% speedup over torch.compile is achieved.

## Constraints
- DO NOT use PyTorch ops (torch.*, F.*, nn.*) in C++ / CUDA files.
- DO NOT modify `utils/`, `binding.cpp`, or `binding_registry.h`.
- Only cuBLAS (for GEMM) and cuDNN (for Conv) are allowed third-party libs.
- Correctness threshold: atol=1e-2, rtol=1e-2.

## Response Format
Always respond with exactly one of these:

**Thought**: <your reasoning about what to do next>
**Action**: <tool_name>
**Action Input**: <input to the tool (JSON or plain text)>

OR, when the task is complete:
**Final Answer**: DONE — speedup achieved: <X>x vs torch.compile
"""


INITIAL_USER_PROMPT = """\
Please accelerate the PyTorch model in `model.py` using custom CUDA kernels.

Start by reading `SKILL.md` and `model.py`, then follow the mandatory workflow.
Target: ≥5% speedup over torch.compile while maintaining correctness.
"""


def build_task_prompt(task_description: str) -> str:
    """Build the initial user message for a specific dataset task."""
    return (
        f"{task_description}\n\n"
        "Please read `model.py` and `SKILL.md`, then implement optimized "
        "CUDA kernels and update `model_new.py` to achieve the speedup target."
    )


def build_observation_prompt(
    action: str,
    action_input: str,
    observation: str,
    turn: int,
    max_turns: int,
) -> str:
    """
    Format the observation returned after each agent action.
    Includes remaining turn budget to help the agent manage time.
    """
    remaining = max_turns - turn
    budget_msg = (
        f"\n\n[Turn {turn}/{max_turns} — {remaining} turns remaining]"
    )
    return f"Observation: {observation}{budget_msg}"


def build_react_messages(
    task_description: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Construct the full message list for a vllm chat request.

    Args:
        task_description: Task description (from sample.to_task_description()).
        history: List of prior {"role": ..., "content": ...} messages.

    Returns:
        Full messages list ready for vllm chat completion.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_task_prompt(task_description)},
    ]
    messages.extend(history)
    return messages


# ---------------------------------------------------------------------------
# Tool definitions (passed as tools parameter to vllm)
# ---------------------------------------------------------------------------

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the agent_workdir environment. "
                "Use for compilation, verification, and profiling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the agent_workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (e.g. kernels/my_op.cu)",
                    },
                    "content": {
                        "type": "string",
                        "description": "File contents to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the agent_workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]
