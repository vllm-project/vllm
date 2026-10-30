# Overlay image: stock vllm/vllm-openai:v0.32.0 + native Nanbeige4.2 support
# (fork Nanbeige/vllm, branch nanbeige42, commit 62f6de733d, "Add support for
# Nanbeige4.2").
#
# ÉVAL po-2025 2026-10-30 (GO user "test comparatif", issue jsboige/vllm#70 /
# registre Q9) — NON-PROD pour l'instant.
#
# Pourquoi cet overlay : stock v0.32.0 porte l'entrée de registre de la PR
# upstream #56071 ("NanbeigeForCausalLM": ("transformers",
# "TransformersForCausalLM")) mais le backend transformers CRASH au boot sur
# les noms de poids loopés du Looped Transformer :
#   AssertionError: layer name model.layers.0.self_attn.loop_attns.0.attn
#   should only contain one integer   (models/utils.py:1029 extract_layer_index)
# Mesuré sur ce siège 30/10 : crash-loop 6 restarts, moteur jamais sain.
# L'implémentation NATIVE du fork (vllm/model_executor/models/nanbeige.py, 573
# lignes) contourne ce chemin et apporte les parsers reasoning+tool "nanbeige".
#
# Greffe (8 fichiers, base = fichiers v0.32.0 extraits de l'image, hunks
# Nanbeige uniquement — le fork date du 27/08, ne PAS prendre ses versions
# entières) :
#   - models/nanbeige.py              : NOUVEAU, copié tel quel (imports 100 %
#                                       compatibles v0.32.0, vérifié en
#                                       conteneur avant build)
#   - configs/nanbeige.py             : NOUVEAU, copié tel quel
#   - models/registry.py              : la ligne #56071 ("transformers", ...)
#                                       REMPLACÉE par ("nanbeige", ...)
#   - transformers_utils/config.py    : +1  nanbeige="NanbeigeConfig"
#   - transformers_utils/configs/__init__.py : +1 entrée _CLASS_TO_MODULE
#   - reasoning/__init__.py           : +4  "nanbeige" → Qwen3ParserReasoningAdapter
#                                       (lazy_parsers, comme le fork)
#   - tool_parsers/__init__.py        : +4  "nanbeige" → Qwen3EngineToolParser
#   - transformers_utils/model_arch_config_convertor.py : classe
#                                       NanbeigeModelArchConfigConvertor
#                                       (num_hidden_layers × num_loops) + ligne
#                                       de registre
#
# Retirer cet overlay dès qu'une release upstream porte le support natif ou un
# backend transformers qui parse les noms loopés.

FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
COPY vllm/model_executor/models/nanbeige.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nanbeige.py
COPY vllm/model_executor/models/registry.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py
COPY vllm/transformers_utils/config.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py
COPY vllm/transformers_utils/configs/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/__init__.py
COPY vllm/transformers_utils/configs/nanbeige.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/nanbeige.py
COPY vllm/reasoning/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py
COPY vllm/tool_parsers/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/__init__.py
COPY vllm/transformers_utils/model_arch_config_convertor.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nanbeige.py \
 && python3 -c "\
from vllm.model_executor.models.registry import _VLLM_MODELS; \
assert _VLLM_MODELS.get('NanbeigeForCausalLM') == ('nanbeige', 'NanbeigeForCausalLM'); \
import vllm.model_executor.models.nanbeige; \
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager; \
assert 'nanbeige' in ReasoningParserManager.lazy_parsers; \
import vllm.tool_parsers as tp; \
assert 'nanbeige' in tp.ToolParserManager.lazy_parsers"
LABEL org.myia.overlay="v0320-nanbeige" org.myia.base="vllm/vllm-openai:v0.32.0" org.myia.upstream="Nanbeige/vllm@nanbeige42 62f6de733d"
