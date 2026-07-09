# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo
from vllm.model_executor.models.glm_ocr import (
    GlmOcrPatchMerger,
    GlmOcrVisionTransformer,
)


class Glm5NextVisionPatchMerger(GlmOcrPatchMerger):
    pass


class Glm5NextVisionTransformer(GlmOcrVisionTransformer):
    def __init__(
        self,
        text_config,
        vision_config,
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            text_config,
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Override the merger to use the GLM5-Next-specific bottleneck width
        # (vision_config.projection_intermediate_size) instead of
        # text_config.intermediate_size used by GLM-OCR.
        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Wires up the HF processor for the multimodal checkpoint.

    The checkpoint's ``processor_config.json`` names the image/video processors
    ``Glm5NextImageProcessor`` / ``Glm5NextVideoProcessor`` -- custom classes
    that exist only on the training side. transformers' ``AutoProcessor``
    cannot resolve them, so it degrades to a bare ``TokenizersBackend``, which
    vLLM rejects (it needs a ``ProcessorMixin``). The GLM5-Next vision tower is
    architecturally identical to GLM-OCR / GLM-4V, so construct the equivalent
    ``Glm4vProcessor`` directly from the checkpoint's image/video configs.
    """

    def get_hf_processor(self, **kwargs: object):
        proc = getattr(self, "_glm5_hf_processor", None)
        if proc is None:
            import json
            import os

            import transformers
            from transformers import (
                AutoTokenizer,
                Glm4vImageProcessor,
                Glm4vProcessor,
            )
            from transformers.models.auto.image_processing_auto import (
                get_image_processor_config,
            )

            model_path = self.ctx.model_config.model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            ip_cfg = get_image_processor_config(model_path)
            # Cap max pixels: the checkpoint ships an absurd ``size.longest_edge``
            # (9.6M image / 100M video pixels) that makes vLLM's startup
            # encoder-profiling reserve huge activation memory and starve the KV
            # cache (300b weights already fill the GPU). 1.25M px (~6400 patches,
            # ~1600 vision tokens) is a sane serving cap.
            _MM_MAX_PIXELS = 1_254_400
            if isinstance(ip_cfg.get("size"), dict):
                ip_cfg["size"]["longest_edge"] = min(
                    ip_cfg["size"].get("longest_edge", _MM_MAX_PIXELS),
                    _MM_MAX_PIXELS,
                )
            image_processor = Glm4vImageProcessor(
                **{k: v for k, v in ip_cfg.items() if k != "image_processor_type"}
            )
            with open(os.path.join(model_path, "processor_config.json")) as f:
                vp_cfg = json.load(f)["video_processor"]
            if isinstance(vp_cfg.get("size"), dict):
                vp_cfg["size"]["longest_edge"] = min(
                    vp_cfg["size"].get("longest_edge", _MM_MAX_PIXELS),
                    _MM_MAX_PIXELS,
                )
            video_cls = transformers.Glm4vVideoProcessor
            video_processor = video_cls(
                **{k: v for k, v in vp_cfg.items() if k != "video_processor_type"}
            )
            proc = Glm4vProcessor(
                image_processor=image_processor,
                video_processor=video_processor,
                tokenizer=tokenizer,
            )
            self._glm5_hf_processor = proc
        return proc
