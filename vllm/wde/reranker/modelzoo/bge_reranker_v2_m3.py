# Adapted from
# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/modeling.py
# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/flag_reranker.py
# FlagEmbedding is licensed under the MIT License.

from vllm.wde.encode_only.modelzoo.xlm_roberta import (
    XLMRobertaForSequenceClassification)


class BGERerankerV2M3(XLMRobertaForSequenceClassification):
    pass
