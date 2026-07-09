# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM5-Next multimodal wrapper.

Composes the GLM5-Next vision tower (:class:`Glm5NextVisionTransformer`) with
the text model (:class:`Glm5NextForCausalLM`). The tower is architecturally
identical to GLM-OCR's, so we reuse the GLM-4V multimodal machinery
(processor, mrope, ``embed_multimodal``, encoder CUDA graph, weight mapping)
by subclassing :class:`Glm4vForConditionalGeneration` and swapping
``self.visual`` and ``self.language_model``.
"""

from typing import ClassVar, Literal

from vllm.config import VllmConfig
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .vision_tower import Glm5NextVisionTransformer


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
                **{
                    k: v
                    for k, v in ip_cfg.items()
                    if k != "image_processor_type"
                }
            )
            with open(os.path.join(model_path, "processor_config.json")) as f:
                vp_cfg = json.load(f)["video_processor"]
            if isinstance(vp_cfg.get("size"), dict):
                vp_cfg["size"]["longest_edge"] = min(
                    vp_cfg["size"].get("longest_edge", _MM_MAX_PIXELS),
                    _MM_MAX_PIXELS,
                )
            video_cls = getattr(transformers, "Glm4vVideoProcessor")
            video_processor = video_cls(
                **{
                    k: v
                    for k, v in vp_cfg.items()
                    if k != "video_processor_type"
                }
            )
            proc = Glm4vProcessor(
                image_processor=image_processor,
                video_processor=video_processor,
                tokenizer=tokenizer,
            )
            self._glm5_hf_processor = proc
        return proc


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-5),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.
