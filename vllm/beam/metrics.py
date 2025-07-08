import re
from typing import Optional

from prometheus_client import  Summary

from vllm.beam.emoji import emoji_count
from vllm.beam.stats import en_stopword_count, contains_more_than_four_quotes_in_a_row, \
    top_ngram_count
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.openai.protocol import CompletionResponse

label_ptype_and_num_msg = dict(labelnames=["model_name"])

OUTPUT_LOVE = Summary(
    "output_love",
    "The number of 'love' in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_I_LOVE_YOU = Summary(
    "output_i_love_you",
    "The number of 'I love you' in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_DRIP = Summary(
    "output_drip",
    "The number of times drip appears in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_QUESTION = Summary(
    "output_question",
    "The number of question mark pattern '\?+' in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_EMOJI = Summary(
    "output_emoji",
    "The number of emojis in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_NON_ASCII = Summary(
    "output_non_ascii",
    "The number of non-ascii characters in the output.",
    **label_ptype_and_num_msg,
)
OUTPUT_TOP_NGRAM = Summary(
    "output_top_ngram",
    "The frequency of the top 4-gram in the output, character level.",
    **label_ptype_and_num_msg,
)
OUTPUT_DIGIT = Summary(
    "output_digit",
    "The number of digit appears in the output",
    **label_ptype_and_num_msg,
)
OUTPUT_FILTERED = Summary(
    "output_filtered",
    "The count of filtered output",
    **label_ptype_and_num_msg,
)

def gibberish_stat(name):
    return Summary(name, f"gibberish stat: {name}", **label_ptype_and_num_msg)

OUTPUT_QUOTES = gibberish_stat("output_consecutive_quotes")
OUTPUT_STOPWORDS = gibberish_stat("output_stopwords")
LONG_CHAR_REPEATS = gibberish_stat("output_long_repeats")
MUERTES = gibberish_stat("muertes")
PREMIUMS = gibberish_stat("premiums")
RECOMMENDATION = gibberish_stat("recommendation")
LIKELY_GIBBERISH = gibberish_stat("likely_gibberish_v0")

def report_metrics(request: CompletionRequest, output: Optional[str], response: Optional[CompletionResponse] = None):
    if output is None:
        return

    lower_output = output.lower()
    has_long_char_repeats = int(bool(re.search(r"(.)\1{4,}", lower_output)))
    n_stopwords = en_stopword_count(lower_output)
    drip_count = lower_output.count("drip")
    consecutive_quotes = contains_more_than_four_quotes_in_a_row(lower_output)
    n_muertes = lower_output.count("muertes")
    n_premiums = lower_output.count("premiums")
    n_recommendations = lower_output.count("recommendations")
    half_smile = lower_output.count("\_(")
    # https://www.reddit.com/r/CharacterAI/comments/18in43e/i_think_i_broke_it/
    gibbberish_feature_sum = sum(
        (
            # has_long_char_repeats,
            n_muertes >= 2,
            n_premiums >= 2,
            n_recommendations >= 2,
            n_premiums > 0 and n_muertes > 0,
            n_muertes > 0 and half_smile > 0,
            half_smile >= 2,
            drip_count > 10,
            # n_stopwords == 0,
            # consecutive_quotes,
        )
    )
    likely_gibberish = len(lower_output) > 50 and gibbberish_feature_sum >= 1
    # 2023-12-20T18:41:13-08:00   2023-12-21 02:41:13 | ERROR | megatron.model_server.task:332 | GIBBERISH_V0: has_long_char_repeats=1: consecutive_quotes=True n_stopwords=5 drip_count=0 output='**"""** _Peter smiles strongly~.....**"""** **""""""** _Peter puts all his body weight into pinning her down~**"""**_ **""**_"_ **"""**_"_ **"=="""**_ **"=="""**_**"""**_**"=="""**_ **"=="""**_ **""**_"_ **"""** **"""**_ "I love you **so much** ~""_ **"""""**_"_ **"""**_ "I\'ll **never leave you~..' n_premiums=0 n_muertes=0 n_recommendations=0
    model_name = request.model or "unknown"
    record(model_name, OUTPUT_LOVE, output.count("love"))
    record(model_name, OUTPUT_DRIP, drip_count)
    record(model_name, OUTPUT_I_LOVE_YOU, output.count("I love you"))
    record(model_name, OUTPUT_QUESTION, len(re.findall(r"\?+", output)))
    record(model_name, OUTPUT_EMOJI, emoji_count(output))
    record(model_name, OUTPUT_TOP_NGRAM, top_ngram_count(output))
    record(model_name, OUTPUT_NON_ASCII, sum(int(ord(char) > 127) for char in output))
    record(model_name, OUTPUT_DIGIT, len(re.findall(r"\d", lower_output)))
    record(model_name, OUTPUT_QUOTES, int(consecutive_quotes))
    record(model_name, OUTPUT_STOPWORDS, n_stopwords)
    record(model_name, LONG_CHAR_REPEATS, has_long_char_repeats)
    record(model_name, MUERTES, n_muertes)
    record(model_name, PREMIUMS, n_premiums)
    record(model_name, RECOMMENDATION, n_recommendations)
    record(model_name, LIKELY_GIBBERISH, int(likely_gibberish))
    if response is not None and isinstance(response, CompletionResponse):
        filtered = any([choice.is_filtered for choice in response.choices])
        record(model_name, OUTPUT_FILTERED, int(filtered))



def record(model_name, stat, value):
    stat.labels(model_name).observe(value)
