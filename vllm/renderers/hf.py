# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import itertools
from collections import defaultdict, deque
from collections.abc import Set
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormat,
    ChatTemplateContentFormatOption,
    ChatTemplateResolutionError,
    ConversationMessage,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.hf import CachedHfTokenizer, HfTokenizer
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.func_utils import supports_kw

from .params import ChatParams
from .protocol import BaseRenderer

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalUUIDDict
else:
    MultiModalDataDict = dict[str, Any]
    MultiModalUUIDDict = dict[str, Any]


logger = init_logger(__name__)


_PROCESSOR_CHAT_TEMPLATES = dict[tuple[str, bool], str | None]()
"""
Used in `_try_get_processor_chat_template` to avoid calling
`cached_get_processor` again if the processor fails to be loaded.

This is needed because `lru_cache` does not cache when an exception happens.
"""


def _try_get_processor_chat_template(
    tokenizer: HfTokenizer,
    *,
    trust_remote_code: bool,
) -> str | None:
    cache_key = (tokenizer.name_or_path, trust_remote_code)
    if cache_key in _PROCESSOR_CHAT_TEMPLATES:
        return _PROCESSOR_CHAT_TEMPLATES[cache_key]

    from transformers import (
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )

    try:
        processor = cached_get_processor(
            tokenizer.name_or_path,
            processor_cls=(
                PreTrainedTokenizer,
                PreTrainedTokenizerFast,
                ProcessorMixin,
            ),
            trust_remote_code=trust_remote_code,
        )
        if (
            isinstance(processor, ProcessorMixin)
            and hasattr(processor, "chat_template")
            and (chat_template := processor.chat_template) is not None
        ):
            _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
            return chat_template
    except Exception:
        logger.debug(
            "Failed to load AutoProcessor chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    _PROCESSOR_CHAT_TEMPLATES[cache_key] = None
    return None


def resolve_chat_template(
    tokenizer: HfTokenizer,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: "ModelConfig",
) -> str | None:
    # 1st priority: The given chat template
    if chat_template is not None:
        return chat_template

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        chat_template = _try_get_processor_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        if chat_template is not None:
            return chat_template

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug(
            "Failed to load AutoTokenizer chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    if path is not None:
        logger.info_once(
            "Loading chat template fallback for %s as there isn't one "
            "defined on HF Hub.",
            tokenizer.name_or_path,
        )
        chat_template = load_chat_template(path)
    else:
        logger.debug_once(
            "There is no chat template fallback for %s", tokenizer.name_or_path
        )

    return chat_template


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str | None = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


# NOTE: The proper way to handle this is to build a CFG so that we can handle
# the scope in which each variable is defined, but that is too complicated
def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node):
    messages_varnames = [
        varname for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
    ]

    # Search for {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Search for {%- for content in message['content'] -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _try_extract_ast(chat_template: str) -> jinja2.nodes.Template | None:
    import transformers.utils.chat_template_utils as hf_chat_utils

    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception:
        logger.exception("Error when compiling Jinja template")
        return None


@lru_cache(maxsize=32)
def _detect_content_format(
    chat_template: str,
    *,
    default: ChatTemplateContentFormat,
) -> ChatTemplateContentFormat:
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return default

    try:
        next(_iter_nodes_assign_content_item(jinja_ast))
    except StopIteration:
        return "string"
    except Exception:
        logger.exception("Error when parsing AST of Jinja template")
        return default
    else:
        return "openai"


def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: HfTokenizer,
    *,
    model_config: "ModelConfig",
) -> ChatTemplateContentFormat:
    resolved_chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    jinja_text = (
        resolved_chat_template
        if isinstance(resolved_chat_template, str)
        else load_chat_template(chat_template, is_literal=True)
    )

    detected_format = (
        "string"
        if jinja_text is None
        else _detect_content_format(jinja_text, default="string")
    )

    return detected_format


@lru_cache
def _log_chat_template_content_format(
    chat_template: str | None,  # For caching purposes
    given_format: ChatTemplateContentFormatOption,
    detected_format: ChatTemplateContentFormatOption,
):
    logger.info(
        "Detected the chat template content format to be '%s'. "
        "You can set `--chat-template-content-format` to override this.",
        detected_format,
    )

    if given_format != "auto" and given_format != detected_format:
        logger.warning(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )


def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: HfTokenizer,
    *,
    model_config: "ModelConfig",
) -> ChatTemplateContentFormat:
    if given_format != "auto":
        return given_format

    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    _log_chat_template_content_format(
        chat_template,
        given_format=given_format,
        detected_format=detected_format,
    )

    return detected_format


# adapted from https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py#L398-L412
# only preserve the parse function used to resolve chat template kwargs
class AssistantTracker(jinja2.ext.Extension):
    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_generation_support")
        call_block = jinja2.nodes.CallBlock(call, [], [], body)
        return call_block.set_lineno(lineno)


def _resolve_chat_template_kwargs(chat_template: str) -> Set[str]:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    parsed_content = env.parse(chat_template)
    template_vars = jinja2.meta.find_undeclared_variables(parsed_content)
    return template_vars


_cached_resolve_chat_template_kwargs = lru_cache(_resolve_chat_template_kwargs)


@lru_cache
def _get_hf_base_chat_template_params() -> frozenset[str]:
    from transformers import PreTrainedTokenizer

    # Get standard parameters from HuggingFace's base tokenizer class.
    # This dynamically extracts parameters from PreTrainedTokenizer's
    # apply_chat_template method, ensuring compatibility with tokenizers
    # that use **kwargs to receive standard parameters.

    # Read signature from HF's base class - the single source of truth
    base_sig = inspect.signature(PreTrainedTokenizer.apply_chat_template)

    # Exclude VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) placeholders
    return frozenset(
        p.name
        for p in base_sig.parameters.values()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


def resolve_chat_template_kwargs(
    tokenizer: HfTokenizer,
    chat_template: str,
    chat_template_kwargs: dict[str, Any],
    raise_on_unexpected: bool = True,
) -> dict[str, Any]:
    # We exclude chat_template from kwargs here, because
    # chat template has been already resolved at this stage
    unexpected_vars = {"chat_template", "tokenize"}
    if raise_on_unexpected and (
        unexpected_in_kwargs := unexpected_vars & chat_template_kwargs.keys()
    ):
        raise ValueError(
            "Found unexpected chat template kwargs from request: "
            f"{unexpected_in_kwargs}"
        )

    fn_kw = {
        k
        for k in chat_template_kwargs
        if supports_kw(tokenizer.apply_chat_template, k, allow_var_kwargs=False)
    }
    template_vars = _cached_resolve_chat_template_kwargs(chat_template)

    # Allow standard HF parameters even if tokenizer uses **kwargs to receive them
    hf_base_params = _get_hf_base_chat_template_params()

    accept_vars = (fn_kw | template_vars | hf_base_params) - unexpected_vars
    return {k: v for k, v in chat_template_kwargs.items() if k in accept_vars}


def safe_apply_chat_template(
    model_config: "ModelConfig",
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    tokenize: bool = True,
    **kwargs,
) -> str | list[int]:
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )
    if chat_template is None:
        raise ChatTemplateResolutionError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        )

    resolved_kwargs = resolve_chat_template_kwargs(
        tokenizer=tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=kwargs,
    )

    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=chat_template,
            tokenize=tokenize,
            **resolved_kwargs,
        )
    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template"
        )
        raise ValueError(str(e)) from e


def rebuild_mm_uuids_from_mm_data(
    mm_uuids: "MultiModalUUIDDict",
    mm_data: "MultiModalDataDict",
) -> "MultiModalUUIDDict":
    """Rebuild mm_uuids after vision_chunk processing.

    When videos are split into chunks, the original UUIDs need to be updated
    to reflect the new UUIDs generated for each chunk.

    Args:
        mm_uuids: Original UUIDs dictionary
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        Updated UUIDs dictionary with chunk UUIDs
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return mm_uuids

    assert all(isinstance(item, dict) for item in vision_chunks), (
        "Expected all vision_chunk items to be dicts"
    )
    vision_chunks = cast(list[dict[str, Any]], vision_chunks)
    vision_chunk_uuids = [
        uuid_val for item in vision_chunks if (uuid_val := item.get("uuid")) is not None
    ]

    if vision_chunk_uuids:
        mm_uuids = dict(mm_uuids)
        mm_uuids["vision_chunk"] = vision_chunk_uuids

    return mm_uuids


def build_video_prompts_from_mm_data(
    mm_data: "MultiModalDataDict",
) -> list[str]:
    """Build video prompts from vision_chunk data.

    Collects prompts from video chunks and groups them by video_idx.

    Args:
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        List of video prompts, one per video.
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return []

    # Group chunks by video_idx
    video_prompts_dict: dict[int, list[str]] = defaultdict(list)

    for item in vision_chunks:
        # vision_chunk items are always dicts (VisionChunkImage/VisionChunkVideo)
        assert isinstance(item, dict)
        if item.get("type") == "video_chunk":
            video_idx = item.get("video_idx", 0)
            prompt = item.get("prompt", "")
            video_prompts_dict[video_idx].append(prompt)

    # Build prompts in video order
    video_prompts = [
        "".join(video_prompts_dict[video_idx])
        for video_idx in sorted(video_prompts_dict.keys())
    ]

    return video_prompts


def replace_vision_chunk_video_placeholder(
    prompt_raw: str | list[int],
    mm_data: "MultiModalDataDict",
    video_placeholder: str | None,
) -> str | list[int]:
    # get video placehoder, replace it with runtime video-chunk prompts
    if video_placeholder and isinstance(prompt_raw, str):
        video_prompts = build_video_prompts_from_mm_data(mm_data)

        # replace in order
        prompt_raw_parts = prompt_raw.split(video_placeholder)
        if len(prompt_raw_parts) == len(video_prompts) + 1:
            prompt_raw = "".join(
                itertools.chain.from_iterable(zip(prompt_raw_parts, video_prompts))
            )
            prompt_raw += prompt_raw_parts[-1]
        else:
            logger.warning(
                "Number of video placeholders (%d) does not match "
                "number of videos (%d) in the request.",
                len(prompt_raw_parts) - 1,
                len(video_prompts),
            )
    return prompt_raw


class HfRenderer(BaseRenderer):
    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "BaseRenderer":
        return cls(config, tokenizer_kwargs)

    def __init__(
        self,
        config: ModelConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(config)

        self.use_unified_vision_chunk = getattr(
            config.hf_config, "use_unified_vision_chunk", False
        )

        if config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cast(
                HfTokenizer,
                cached_get_tokenizer(
                    tokenizer_cls=CachedHfTokenizer,  # type: ignore[type-abstract]
                    **tokenizer_kwargs,
                ),
            )

        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> HfTokenizer | None:
        return self._tokenizer

    def get_tokenizer(self) -> HfTokenizer:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        model_config = self.config
        tokenizer = self.get_tokenizer()

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
        )

        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            mm_uuids = rebuild_mm_uuids_from_mm_data(mm_uuids, mm_data)

            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = replace_vision_chunk_video_placeholder(
                prompt_raw,
                mm_data,
                video_placeholder,
            )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], TextPrompt | TokensPrompt | EmbedsPrompt]:
        model_config = self.config
        tokenizer = self.get_tokenizer()

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
        )

        prompt_raw = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        # NOTE: use_unified_vision_chunk is currently specific to Kimi-K2.5
        # model which uses unified vision chunks for both images and videos.
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            # get video placeholder, replace it with runtime video-chunk prompts
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = replace_vision_chunk_video_placeholder(
                prompt_raw,
                mm_data,
                video_placeholder,
            )

        prompt = self.render_completion(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
