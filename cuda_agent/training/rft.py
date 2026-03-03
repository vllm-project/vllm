"""
training/rft.py

Rejection Fine-Tuning (RFT) for actor initialisation.

This is Stage 2 of the CUDA-Agent multi-stage training pipeline (paper §3.3):

  Stage 1: Single-turn PPO warm-up  (produces initial CUDA coding ability)
  Stage 2: RFT actor initialisation ← THIS FILE
  Stage 3: Full agentic PPO

Purpose:
  After Stage 1 PPO, collect multi-turn agent trajectories, filter out
  low-quality ones, and fine-tune the actor via supervised learning.
  This constrains policy entropy, prevents distribution collapse during
  the subsequent agentic PPO stage.

RFT filter criteria (from paper):
  1. Negative reward trajectories are discarded.
  2. Trajectories with tool-call schema violations are discarded.
  3. Trajectories with inefficient patterns (e.g. repeated identical calls)
     are discarded.

Loss:
  L_RFT = -E[Σ_t log π_θ(a_t | s_t, a_{<t})]   (standard SFT cross-entropy)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory filtering
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_negative_reward: int = 0
    dropped_schema_violation: int = 0
    dropped_inefficient: int = 0

    @property
    def keep_rate(self) -> float:
        return self.kept / max(self.total, 1)


def filter_trajectories(
    trajectories: list[dict],
    min_reward: float = 1.0,
) -> tuple[list[dict], FilterStats]:
    """
    Filter collected trajectories for RFT training.

    Args:
        trajectories: List of trajectory dicts (from Trajectory.to_training_dict()).
        min_reward:   Minimum reward to retain (paper: keeps reward ≥ +1).

    Returns:
        (filtered_trajectories, stats)
    """
    stats = FilterStats(total=len(trajectories))
    filtered: list[dict] = []

    for traj in trajectories:
        reward = traj.get("final_reward", -1)

        # 1. Drop negative-reward trajectories
        if reward < min_reward:
            stats.dropped_negative_reward += 1
            continue

        # 2. Drop schema violations (invalid tool calls)
        if _has_schema_violations(traj):
            stats.dropped_schema_violation += 1
            continue

        # 3. Drop inefficient patterns (repeated identical actions)
        if _has_inefficient_patterns(traj):
            stats.dropped_inefficient += 1
            continue

        filtered.append(traj)
        stats.kept += 1

    logger.info(
        "RFT filter: %d/%d kept (%.1f%%) | neg=%d schema=%d inefficient=%d",
        stats.kept, stats.total, stats.keep_rate * 100,
        stats.dropped_negative_reward, stats.dropped_schema_violation,
        stats.dropped_inefficient,
    )
    return filtered, stats


def _has_schema_violations(traj: dict) -> bool:
    """Check if any step has an invalid tool-call schema."""
    valid_actions = {"bash", "write_file", "read_file"}
    for step in traj.get("steps", []):
        action = step.get("action", "")
        if action not in valid_actions:
            return True
        # write_file must have path and content
        if action == "write_file":
            ai = step.get("action_input", {})
            if isinstance(ai, dict):
                if not ai.get("path") or "content" not in ai:
                    return True
    return False


def _has_inefficient_patterns(traj: dict, max_identical: int = 5) -> bool:
    """
    Detect repeated identical actions (sign of a stuck agent).
    Discard if the same command appears more than max_identical times.
    """
    from collections import Counter
    commands = []
    for step in traj.get("steps", []):
        if step.get("action") == "bash":
            ai = step.get("action_input", {})
            cmd = ai.get("command", "") if isinstance(ai, dict) else str(ai)
            commands.append(cmd)
    counts = Counter(commands)
    return any(c > max_identical for c in counts.values())


# ---------------------------------------------------------------------------
# SFT dataset builder
# ---------------------------------------------------------------------------

def build_sft_dataset(
    filtered_trajectories: list[dict],
    output_path: str | Path,
    system_prompt: str = "",
) -> int:
    """
    Convert filtered trajectories into an SFT JSONL dataset.

    Format (one line per training example):
    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},  # task description
            {"role": "assistant", "content": "..."},  # thought + action
            {"role": "user",      "content": "..."},  # observation
            ...
        ],
        "metadata": {
            "sample_id": "...",
            "final_reward": 3,
            "speedup_vs_compile": 1.8,
            "num_turns": 12,
        }
    }

    Returns the number of examples written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from cuda_agent.agent.prompt_templates import SYSTEM_PROMPT as DEFAULT_SYSTEM
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for traj in filtered_trajectories:
            messages = [{"role": "system", "content": system_prompt}]
            for step in traj.get("steps", []):
                # Assistant turn
                thought = step.get("thought", "")
                action = step.get("action", "")
                ai = step.get("action_input", {})
                asst_content = (
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(ai) if isinstance(ai, dict) else ai}"
                )
                messages.append({"role": "assistant", "content": asst_content})
                # Observation turn
                obs = step.get("observation", "")
                messages.append({"role": "user", "content": f"Observation: {obs}"})

            example = {
                "messages": messages,
                "metadata": {
                    "sample_id": traj.get("sample_id", ""),
                    "final_reward": traj.get("final_reward", -1),
                    "speedup_vs_compile": traj.get("speedup_vs_compile", 0.0),
                    "num_turns": traj.get("num_turns", 0),
                },
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Wrote %d SFT examples to %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# RFT training (SFT on filtered trajectories)
# ---------------------------------------------------------------------------

def run_rft(
    model_path: str,
    sft_data_path: str | Path,
    output_dir: str | Path,
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 131_072,
    use_deepspeed: bool = True,
    deepspeed_config: str | None = None,
) -> None:
    """
    Fine-tune the actor model on filtered trajectories (SFT / RFT stage).

    Requires: transformers, trl, accelerate, deepspeed (optional).

    Args:
        model_path:          Path to the base model (Stage 1 PPO checkpoint).
        sft_data_path:       Path to the JSONL file from build_sft_dataset().
        output_dir:          Where to save the fine-tuned actor.
        num_epochs:          Number of training epochs over the filtered data.
        learning_rate:       Optimizer LR (default: 2e-5).
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: For effective batch size scaling.
        max_seq_length:      Max context length (paper: 131,072 tokens).
        use_deepspeed:       Whether to use DeepSpeed ZeRO.
        deepspeed_config:    Path to DeepSpeed config JSON (optional).
    """
    try:
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTTrainer  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install: pip install transformers trl datasets"
        ) from exc

    logger.info("Loading tokenizer and model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # Load SFT data
    import json
    data = []
    with open(sft_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Flatten messages to text for SFTTrainer
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data).map(format_example)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        deepspeed=deepspeed_config,
        report_to="none",
        max_steps=-1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    logger.info("Starting RFT training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("RFT complete. Model saved to %s", output_dir)
