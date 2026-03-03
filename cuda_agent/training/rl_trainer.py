"""
training/rl_trainer.py

Multi-stage PPO training pipeline for CUDA-Agent.

Reproduces the three-stage algorithm from the paper (§3.3):

  ┌─────────────────────────────────────────────────────────────────┐
  │ Stage 1: Single-Turn PPO Warm-up                                │
  │   - Context: 32,768 tokens                                      │
  │   - Non-agentic: model generates CUDA code in one shot          │
  │   - Gives the base model basic CUDA coding ability              │
  │   - Addresses distribution mismatch (CUDA < 0.01% of pretrain) │
  ├─────────────────────────────────────────────────────────────────┤
  │ Stage 2: Actor Initialisation via RFT  (see rft.py)             │
  │   - Collect Stage 1 rollouts, filter, SFT fine-tune actor       │
  │   - Constrains entropy, prevents distribution collapse           │
  ├─────────────────────────────────────────────────────────────────┤
  │ Stage 3: Full Agentic PPO                                        │
  │   - Context: 131,072 tokens                                      │
  │   - Max turns: 150 (training), 200 (evaluation)                  │
  │   - Global batch: 1,024                                          │
  │   - Actor LR: 3e-6, Critic LR: 6e-6                             │
  │   - Asymmetric clipping: ε_lower=0.2, ε_higher=0.28            │
  │   - 150 training steps on 128 NVIDIA H20 GPUs                  │
  └─────────────────────────────────────────────────────────────────┘

Hardware: 128 NVIDIA H20 GPUs in dedicated sandbox pool.
CPU-GPU resource decoupling avoids contention between training cluster
and agent execution sandboxes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from cuda_agent.agent.environment import CUDAAgentEnvironment, Reward
from cuda_agent.agent.react_agent import CUDAReActAgent, Trajectory
from cuda_agent.data.dataset_loader import CUDAAgentSample, load_cuda_agent_dataset
from cuda_agent.data.task_generator import TaskGenerator
from cuda_agent.training.reward import compute_gae, compute_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PPO hyperparameters (from paper Table 2)
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Stage 1 (warm-up)
    stage1_max_context: int   = 32_768
    stage1_max_turns: int     = 1      # single-turn
    stage1_num_steps: int     = 50

    # Stage 3 (full agentic PPO)
    stage3_max_context: int   = 131_072
    stage3_max_turns: int     = 150
    stage3_num_steps: int     = 150
    global_batch_size: int    = 1024
    actor_lr: float           = 3e-6
    critic_lr: float          = 6e-6
    eps_lower: float          = 0.20   # asymmetric clipping
    eps_higher: float         = 0.28   # asymmetric clipping
    gamma: float              = 1.0    # no discounting (sparse reward)
    lam: float                = 0.95   # GAE lambda

    # Infrastructure
    num_gpus: int             = 128    # paper: 128 H20 GPUs
    vllm_host: str            = "http://localhost:8000"
    rollout_workers: int      = 64     # parallel sandbox workers


@dataclass
class TrainingState:
    stage: int = 1
    step: int = 0
    total_episodes: int = 0
    cumulative_reward: float = 0.0
    metrics: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(
    agent: CUDAReActAgent,
    env: CUDAAgentEnvironment,
    samples: list[CUDAAgentSample],
    task_generator: TaskGenerator,
    num_episodes: int,
) -> list[Trajectory]:
    """
    Collect agent trajectories (rollouts) for PPO training.

    Args:
        agent:          CUDAReActAgent instance.
        env:            CUDAAgentEnvironment instance.
        samples:        Dataset samples to use for this batch.
        task_generator: Generates task directories from samples.
        num_episodes:   Number of episodes to collect.

    Returns:
        List of completed Trajectory objects.
    """
    trajectories: list[Trajectory] = []
    sample_idx = 0

    for ep_id in range(num_episodes):
        sample = samples[sample_idx % len(samples)]
        sample_idx += 1

        task_dir = task_generator.generate_task(sample, task_id=f"ep_{ep_id:05d}")
        try:
            traj = agent.run_episode(sample, env, task_dir)
            trajectories.append(traj)
            logger.info(
                "Episode %d/%d: reward=%s speedup=%.2fx turns=%d",
                ep_id + 1, num_episodes,
                traj.final_reward.name,
                traj.speedup_vs_compile,
                traj.total_turns,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Episode %d failed: %s", ep_id, exc)
        finally:
            env.cleanup(env.reset(task_dir))  # cleanup

    return trajectories


# ---------------------------------------------------------------------------
# PPO loss computation
# ---------------------------------------------------------------------------

def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_lower: float = 0.20,
    eps_higher: float = 0.28,
) -> torch.Tensor:
    """
    PPO clipped surrogate loss with asymmetric clipping.

    Paper: ε_lower=0.2, ε_higher=0.28

    L_CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε_l, 1+ε_h) * A_t)]
    where r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """
    ratios = torch.exp(log_probs - old_log_probs)

    # Asymmetric clipping: different bounds for positive and negative advantages
    clipped = torch.where(
        advantages >= 0,
        torch.clamp(ratios, 1.0 - eps_lower, 1.0 + eps_higher),
        torch.clamp(ratios, 1.0 - eps_higher, 1.0 + eps_lower),
    )

    surrogate = torch.min(ratios * advantages, clipped * advantages)
    return -surrogate.mean()


def compute_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    """
    MSE value loss for critic pre-training and critic updates.

    L_V = (1/2) * E[(1/T) * Σ_t (V_φ(s_t) - V_t^target)²]
    """
    return 0.5 * torch.mean((predicted_values - target_values) ** 2)


# ---------------------------------------------------------------------------
# Stage 3: Full Agentic PPO Trainer
# ---------------------------------------------------------------------------

class AgenticPPOTrainer:
    """
    Full multi-turn PPO trainer (Stage 3).

    This class orchestrates:
      1. Rollout collection (via CUDAReActAgent + CUDAAgentEnvironment)
      2. GAE advantage estimation
      3. Actor + critic gradient updates
      4. Logging and checkpoint saving
    """

    def __init__(
        self,
        actor_model,        # HuggingFace model with .forward()
        critic_model,       # Value head model
        actor_optimizer,    # torch.optim.Optimizer
        critic_optimizer,   # torch.optim.Optimizer
        config: PPOConfig,
        agent: CUDAReActAgent,
        env: CUDAAgentEnvironment,
        task_generator: TaskGenerator,
        output_dir: str | Path,
    ):
        self.actor  = actor_model
        self.critic = critic_model
        self.actor_opt  = actor_optimizer
        self.critic_opt = critic_optimizer
        self.config = config
        self.agent  = agent
        self.env    = env
        self.task_gen = task_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state  = TrainingState(stage=3)

    def train(
        self,
        samples: list[CUDAAgentSample],
        num_steps: int | None = None,
    ) -> list[dict]:
        """
        Run the full agentic PPO training loop.

        Args:
            samples:   Training samples from CUDA-Agent-Ops-6K.
            num_steps: Override the number of PPO steps (default: config.stage3_num_steps).

        Returns:
            List of per-step metric dicts.
        """
        num_steps = num_steps or self.config.stage3_num_steps
        metrics: list[dict] = []

        for step in range(1, num_steps + 1):
            self.state.step = step
            t0 = time.perf_counter()

            # 1. Collect rollouts
            logger.info("Step %d/%d: collecting rollouts …", step, num_steps)
            episodes_per_step = max(1, self.config.global_batch_size // self.config.stage3_max_turns)
            trajectories = collect_rollouts(
                self.agent, self.env, samples,
                self.task_gen, num_episodes=episodes_per_step,
            )

            # 2. Compute advantages
            all_advantages, all_returns = self._compute_batch_advantages(trajectories)

            # 3. Actor update
            actor_loss = self._update_actor(trajectories, all_advantages)

            # 4. Critic update
            critic_loss = self._update_critic(trajectories, all_returns)

            elapsed = time.perf_counter() - t0
            step_metrics = self._log_step(
                step, trajectories, actor_loss, critic_loss, elapsed
            )
            metrics.append(step_metrics)

            # Save checkpoint every 10 steps
            if step % 10 == 0:
                self._save_checkpoint(step)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_batch_advantages(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[list[float]], list[list[float]]]:
        all_advantages, all_returns = [], []
        for traj in trajectories:
            T = max(traj.total_turns, 1)
            rewards = [0.0] * T
            if T > 0:
                rewards[-1] = float(traj.final_reward)
            values = [0.0] * T  # placeholder; critic provides real values
            advantages = compute_gae(rewards, values, self.config.gamma, self.config.lam)
            returns    = compute_returns(rewards, self.config.gamma)
            all_advantages.append(advantages)
            all_returns.append(returns)
        return all_advantages, all_returns

    def _update_actor(
        self, trajectories: list[Trajectory], all_advantages: list[list[float]]
    ) -> float:
        """One actor gradient update step."""
        self.actor.train()
        self.actor_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        # NOTE: In a real implementation, batch the trajectory token log-probs
        # and compute PPO loss using compute_ppo_loss().
        # Placeholder for the training loop structure.
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()
        return total_loss.item()

    def _update_critic(
        self, trajectories: list[Trajectory], all_returns: list[list[float]]
    ) -> float:
        """One critic gradient update step."""
        self.critic.train()
        self.critic_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        # NOTE: Real implementation calls critic forward and compute_value_loss().
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()
        return total_loss.item()

    def _log_step(
        self,
        step: int,
        trajectories: list[Trajectory],
        actor_loss: float,
        critic_loss: float,
        elapsed: float,
    ) -> dict:
        n = len(trajectories)
        rewards = [float(t.final_reward) for t in trajectories]
        speedups = [t.speedup_vs_compile for t in trajectories if t.speedup_vs_compile > 0]
        faster_compile = sum(1 for t in trajectories if t.final_reward == Reward.FASTER_THAN_COMPILE)

        metrics = {
            "step": step,
            "episodes": n,
            "mean_reward": sum(rewards) / max(n, 1),
            "faster_than_compile_rate": faster_compile / max(n, 1),
            "mean_speedup": sum(speedups) / max(len(speedups), 1),
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "elapsed_s": elapsed,
        }

        logger.info(
            "Step %d | reward=%.2f | faster_compile=%.1f%% | speedup=%.2fx | "
            "actor_loss=%.4f | critic_loss=%.4f | time=%.1fs",
            step, metrics["mean_reward"],
            metrics["faster_than_compile_rate"] * 100,
            metrics["mean_speedup"],
            actor_loss, critic_loss, elapsed,
        )

        # Append to JSONL log
        log_path = self.output_dir / "training_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        self.state.metrics.append(metrics)
        return metrics

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.output_dir / f"checkpoint_step_{step:05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.actor.save_pretrained(str(ckpt_dir / "actor"))
            self.critic.save_pretrained(str(ckpt_dir / "critic"))
            logger.info("Checkpoint saved: %s", ckpt_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# Critic pre-training
# ---------------------------------------------------------------------------

def pretrain_critic(
    critic_model,
    optimizer,
    trajectories: list[Trajectory],
    gamma: float = 1.0,
    num_epochs: int = 3,
) -> list[float]:
    """
    Pre-train the critic on target values from collected trajectories (Stage 3 init).

    Paper: Critic pretrained on GAE target values computed from Stage 2 RFT rollouts.
    Loss: L = (1/2) E[(1/T) Σ_t (V_φ(s_t) - V_t^target)²]

    Args:
        critic_model:  Value head model.
        optimizer:     torch.optim.Optimizer for the critic.
        trajectories:  Collected trajectories from RFT stage.
        gamma:         Discount factor.
        num_epochs:    Pre-training epochs.

    Returns:
        List of per-epoch loss values.
    """
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for traj in trajectories:
            T = max(traj.total_turns, 1)
            rewards = [0.0] * T
            if T > 0:
                rewards[-1] = float(traj.final_reward)
            target_values = torch.tensor(
                compute_returns(rewards, gamma), dtype=torch.float32
            )
            # NOTE: Real implementation: encode states and call critic_model.forward()
            # predicted_values = critic_model(encoded_states)
            predicted_values = torch.zeros_like(target_values)
            loss = compute_value_loss(predicted_values, target_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        epoch_loss = total_loss / max(count, 1)
        epoch_losses.append(epoch_loss)
        logger.info("Critic pretraining epoch %d/%d: loss=%.4f", epoch + 1, num_epochs, epoch_loss)

    return epoch_losses
