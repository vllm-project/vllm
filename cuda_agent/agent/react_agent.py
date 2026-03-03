"""
agent/react_agent.py

ReAct-style CUDA Agent powered by vllm.

Architecture mirrors the paper:
  - System prompt encodes the SKILL.md workflow
  - Agent interleaves Thought → Action → Observation loops
  - Context window: up to 128k tokens (training), 200k (evaluation)
  - Max turns: 150 (training), 200 (evaluation)
  - Base model: Seed1.6-equivalent or any instruction-tuned LLM

The agent is used both during RL training (generating rollouts)
and at evaluation time.

Usage:
    agent = CUDAReActAgent(model="seed1.6", max_turns=150)
    trajectory = agent.run_episode(sample, env)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cuda_agent.agent.environment import (
    CUDAAgentEnvironment,
    EpisodeState,
    Reward,
)
from cuda_agent.agent.prompt_templates import (
    AGENT_TOOLS,
    SYSTEM_PROMPT,
    build_react_messages,
    build_task_prompt,
)
from cuda_agent.data.dataset_loader import CUDAAgentSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory (for RL training)
# ---------------------------------------------------------------------------

@dataclass
class Step:
    turn: int
    thought: str
    action: str
    action_input: dict | str
    observation: str
    reward_so_far: float = 0.0


@dataclass
class Trajectory:
    sample_id: str
    steps: list[Step] = field(default_factory=list)
    final_reward: Reward = Reward.CORRECTNESS_FAILURE
    speedup_vs_compile: float = 0.0
    done: bool = False

    @property
    def total_turns(self) -> int:
        return len(self.steps)

    @property
    def succeeded(self) -> bool:
        return self.final_reward == Reward.FASTER_THAN_COMPILE

    def to_training_dict(self) -> dict:
        """Serialise for RFT / PPO training."""
        return {
            "sample_id": self.sample_id,
            "final_reward": int(self.final_reward),
            "speedup_vs_compile": self.speedup_vs_compile,
            "num_turns": self.total_turns,
            "steps": [
                {
                    "turn": s.turn,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CUDAReActAgent:
    """
    vllm-based ReAct agent for CUDA kernel optimization.

    At each turn:
      1. Build the full message history.
      2. Call vllm to generate the next thought + action.
      3. Parse the action and call env.step().
      4. Append the observation to history.
      5. Repeat until done or max_turns reached.
    """

    def __init__(
        self,
        model: str = "BytedTsinghua-SIA/Seed1.6",
        max_turns: int = 150,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vllm_host: str = "http://localhost:8000",
        use_openai_api: bool = True,
    ):
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.vllm_host = vllm_host
        self.use_openai_api = use_openai_api

        self._client = None  # initialised lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        sample: CUDAAgentSample,
        env: CUDAAgentEnvironment,
        task_dir: str | Path,
    ) -> Trajectory:
        """
        Run a full agent episode for one dataset sample.

        Args:
            sample:   The CUDA-Agent-Ops-6K sample to optimise.
            env:      The CUDAAgentEnvironment instance.
            task_dir: Pre-generated task directory path.

        Returns:
            Trajectory containing all steps and the final reward.
        """
        state = env.reset(task_dir)
        trajectory = Trajectory(sample_id=sample.sample_id)
        messages = build_react_messages(
            sample.to_task_description(), history=[]
        )

        while not state.done and state.turn < self.max_turns:
            # Generate next thought + action
            response_text = self._generate(messages)
            thought, action, action_input = _parse_react_response(response_text)

            if action is None:
                # Model output a final answer or couldn't parse — end episode
                trajectory.done = True
                break

            # Execute action in environment
            obs, state = env.step(state, action, action_input)

            # Record step
            step = Step(
                turn=state.turn,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=obs,
            )
            trajectory.steps.append(step)

            # Append to message history for next turn
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user",      "content": f"Observation: {obs}"})

            logger.debug("Turn %d: action=%s | obs[:100]=%s",
                         state.turn, action, obs[:100])

        # Compute final reward
        final_reward = env.compute_reward(state)
        trajectory.final_reward = final_reward
        trajectory.speedup_vs_compile = state.speedup_vs_compile
        trajectory.done = True

        logger.info(
            "Episode done: sample=%s reward=%s speedup=%.2f turns=%d",
            sample.sample_id, final_reward.name,
            state.speedup_vs_compile, trajectory.total_turns,
        )
        return trajectory

    # ------------------------------------------------------------------
    # vllm inference
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(
                api_key="EMPTY",
                base_url=f"{self.vllm_host}/v1",
            )
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        return self._client

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Call vllm (via OpenAI-compatible API) and return the response text."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tools=AGENT_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        # Handle both tool-call and text responses
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return (
                f"Thought: Calling tool {tc.function.name}\n"
                f"Action: {tc.function.name}\n"
                f"Action Input: {tc.function.arguments}"
            )
        return choice.message.content or ""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"(?:Thought|Think|Reasoning)[:：]\s*(.+?)(?=\n(?:Action|$))", re.S | re.I)
_ACTION_RE  = re.compile(r"Action[:：]\s*(\w+)", re.I)
_INPUT_RE   = re.compile(r"Action Input[:：]\s*(.+?)(?=\n(?:Thought|Observation|$)|\Z)", re.S | re.I)
_FINAL_RE   = re.compile(r"Final Answer[:：]", re.I)


def _parse_react_response(text: str) -> tuple[str, str | None, dict | str]:
    """
    Parse a ReAct-style response into (thought, action, action_input).

    Returns (thought, None, {}) if the model produced a Final Answer.
    """
    if _FINAL_RE.search(text):
        return text, None, {}

    thought = ""
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()

    action: str | None = None
    m = _ACTION_RE.search(text)
    if m:
        action = m.group(1).strip()

    action_input: dict | str = {}
    m = _INPUT_RE.search(text)
    if m:
        raw = m.group(1).strip()
        try:
            action_input = json.loads(raw)
        except json.JSONDecodeError:
            action_input = raw

    return thought, action, action_input
