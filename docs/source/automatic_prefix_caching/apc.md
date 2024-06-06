
# Introduction

## What is automatic prefix caching

Automatic prefix caching (APC in short) caches the intermediate computation results of existing queries (in the format of KV cache), so that a new query can directly reuse the intermediate computation results if it shares the same prefix with one of existing queries, allowing the new query to be served much quicker.

## Enabling APC in vllm

Set `enable_prefix_caching=True` in vllm engine to enable APC. Here is a minimum example (this example is extracted from [vllm/benchmarks/benchmark_prefix_caching.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_prefix_caching.py)):

```python

from vllm import LLM, SamplingParams

LONG_PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n"  # noqa: E501

# let enable_prefix_caching=True to enable APC
llm = LLM(model='baichuan-inc/Baichuan2-13B-Chat',
            enable_prefix_caching=True)

sampling_params = SamplingParams(temperature=0, max_tokens=10)

# Asking the content of cell (1,1)
llm.generate(
    LONG_PROMPT + "# Question\nWhat' s the content in the (1,1) cells\n",
    sampling_params=sampling_params
)

# Asking the content of cell (2,2)
# Will be faster than the last query
# as the computation of LONG_PROMPT can be reused
llm.generate(
    LONG_PROMPT + "# Question\nWhat' s the content in the (2,2) cells\n",
    sampling_params=sampling_params
)

```

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:
- Long document query, where the user repeatedly query the same long document (e.g. software mannual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vllm to process this long document (and generate its corresponding KV cache) *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vllm to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vllm to reuse the processing results of the chat history across all future rounds of conversation, allowing vllm to serve future requests with much higher throughput and much lower latency.


APC in general does not reduce the performance of vllm. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vllm spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).