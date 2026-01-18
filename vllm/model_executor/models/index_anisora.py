"""Experimental Index-AniSora (text-to-video) wrapper for vLLM.

This module provides a minimal integration of Index-AniSora via
Hugging Face diffusers AutoPipelineForText2Video.

Notes
-----
- This is an experimental, non-streaming video generation path.
- It loads the diffusers pipeline and runs a full generation call
  per request. It does not use token streaming or KV cache.
- Designed to be called directly for now; registry/runner wiring
  can be added incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class IndexAniSoraParams:
    """Generation parameters with safe defaults for VRAM.

    These defaults aim to reduce OOM risk on common GPUs.
    """

    height: int = 360
    width: int = 640
    num_frames: int = 16
    num_inference_steps: int = 6
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class IndexAniSoraForVideoGeneration:
    """Minimal wrapper around diffusers text-to-video pipeline.

    Usage
    -----
    model = IndexAniSoraForVideoGeneration(
        model_id="IndexTeam/Index-anisora",
        torch_dtype=torch.float16,
        device="cuda",
    )
    frames = model.forward("A girl running", num_frames=8, height=360, width=640)
    """

    def __init__(
        self,
        model_id: str = "IndexTeam/Index-anisora",
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token
        self.pipeline_kwargs = pipeline_kwargs or {}

        self.pipe = None
        self._load_model()

    # Keep a simple API aligned with the docs drafts
    @staticmethod
    def get_supported_models() -> List[str]:
        return ["IndexTeam/Index-anisora"]

    def _load_model(self) -> None:
        try:
            from diffusers import AutoPipelineForText2Video
        except Exception as e:  # pragma: no cover - import path
            raise ImportError(
                "diffusers is required for Index-AniSora. Install with `pip install diffusers transformers accelerate`"
            ) from e

        # Lazily enable common memory savers; users can override via pipeline_kwargs
        self.pipe = AutoPipelineForText2Video.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            trust_remote_code=self.trust_remote_code,
            token=self.hf_token,
            **self.pipeline_kwargs,
        )

        # Move to device and apply typical optimizations
        self.pipe = self.pipe.to(self.device)
        # Enable attention slicing if available to reduce VRAM
        if hasattr(self.pipe, "enable_attention_slicing"):
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        # Optionally compile the UNet for speed (PyTorch 2.x)
        try:  # pragma: no cover - runtime dependent
            self.pipe.unet = torch.compile(self.pipe.unet)  # type: ignore[attr-defined]
        except Exception:
            # Compilation may fail depending on env; ignore safely
            pass

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        image: Optional[Any] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_type: str = "tensor",  # "tensor" | "pil_list"
        **kwargs: Any,
    ) -> Any:
        """Generate a video.

        Parameters mirror diffusers Text2Video pipeline where relevant.
        Returns either a torch.Tensor [F,H,W,3] uint8 or list of PIL.Image.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call __init__ again.")

        params = IndexAniSoraParams()
        h = height or params.height
        w = width or params.width
        f = num_frames or params.num_frames
        steps = num_inference_steps or params.num_inference_steps
        gs = guidance_scale if guidance_scale is not None else params.guidance_scale

        if generator is None:
            if seed is None:
                seed = params.seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prepare kwargs for pipeline
        pipe_kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=h,
            width=w,
            num_frames=f,
            num_inference_steps=steps,
            guidance_scale=gs,
            generator=generator,
        )
        if image is not None:
            # Some anisora variants support img-to-video guidance
            pipe_kwargs["image"] = image

        pipe_kwargs.update(kwargs)

        output = self.pipe(**pipe_kwargs)

        # diffusers returns a list of PIL images as output.frames
        frames = getattr(output, "frames", None)
        if frames is None:
            # Some pipelines use .images naming
            frames = getattr(output, "images", None)
        if frames is None:
            raise RuntimeError("Unexpected output from diffusers pipeline.")

        if return_type == "pil_list":
            return frames

        # Convert to torch uint8 tensor [F, H, W, 3]
        from vllm.model_executor.layers.video_utils import (
            pil_frames_to_uint8_tensor,
        )

        return pil_frames_to_uint8_tensor(frames)


__all__ = ["IndexAniSoraForVideoGeneration", "IndexAniSoraParams"]
