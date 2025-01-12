from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

from transformers import PretrainedConfig, AutoConfig

class BaseVisualTokenizerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=16384,
        tokenize_function="softmax",
        tau=1.0,
        depths=None,
        drop_cls_token=False,
        backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
        hidden_stride: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split('|')]
        self.depths = depths
        self.backbone_kwargs = {}
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "siglip_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]

class OvisConfig(PretrainedConfig):
    model_type = "ovis"

    def __init__(
        self,
        llm_config: Optional[Union[PretrainedConfig, dict]] = None,
        visual_tokenizer_config: Optional[Union[PretrainedConfig, dict]] = None,
        multimodal_max_length=8192,
        hidden_size=None,
        conversation_formatter_class=None,
        llm_attn_implementation=None,
        disable_tie_weight=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PretrainedConfig, dict)), \
                f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config['model_type']
                llm_config.pop('model_type')
                llm_config = AutoConfig.for_model(model_type, **llm_config)
        self.llm_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(visual_tokenizer_config, (PretrainedConfig, dict)), \
                f"expect `visual_tokenizer_config` to be instance of PretrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            if not isinstance(visual_tokenizer_config, PretrainedConfig):
                model_type = visual_tokenizer_config['model_type']
                visual_tokenizer_config.pop('model_type')
                visual_tokenizer_config = AutoConfig.for_model(model_type, **visual_tokenizer_config)
        self.visual_tokenizer_config = visual_tokenizer_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight

class ConversationFormatter(ABC):
    support_tokenizer_types = None

    def __init__(self, tokenizer):
        tokenizer_type = type(tokenizer).__name__
        assert tokenizer_type in self.support_tokenizer_types, \
            f'Invalid tokenizer type, expected one from `{self.support_tokenizer_types}`, but got `{tokenizer_type}`'
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID

    def _tokenize_with_image_symbol(self, text):
        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(self.image_token)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(self.image_token_id)
        return token_ids

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass


class GemmaConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['GemmaTokenizer', 'GemmaTokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Gemma does not support system prompt
        self.from2role = {
            "human": "<start_of_turn>user\n",
            "gpt": "<start_of_turn>model\n",
        }
        self.gpt_token_num = None
        self.im_end = "<end_of_turn>\n"
        self.bos_token = "<bos>"
        self.bos_token_ids = None

    def format(self, conversations: List[Dict], generation_preface=None):
        if self.gpt_token_num is None:
            self.gpt_token_num = len(self.tokenizer(self.from2role["gpt"], add_special_tokens=False).input_ids)

        if self.bos_token_ids is None:
            self.bos_token_ids = self.tokenizer(self.bos_token, add_special_tokens=False).input_ids

        if conversations[0]["from"] == "system":
            raise ValueError("Gemma does not support system prompt")

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = "" + self.bos_token
        input_ids = [] + self.bos_token_ids
        labels = [] + [IGNORE_ID] * len(input_ids)
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"].strip()
            text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt":
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[self.gpt_token_num:-1] = token_ids[self.gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids