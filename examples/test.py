import os
from vllm import LLM, SamplingParams
import time
import numpy as np

os.environ["VLLM_USE_V1"] = "1"


llm = LLM(
    model="/net/storage149/autofs/css22/nmg/models/granite3.1-8b/base/",
    dtype='float16',
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

doc = "Switzerland,[d] officially the Swiss Confederation,[e] is a landlocked country located in west-central Europe.[f][13] It is bordered by Italy to the south, France to the west, Germany to the north, and Austria and Liechtenstein to the east. Switzerland is geographically divided among the Swiss Plateau, the Alps and the Jura; the Alps occupy the greater part of the territory, whereas most of the country's nearly 9 million people are concentrated on the plateau, which hosts its largest cities and economic centres, including Zurich, Geneva, and Lausanne.[14]"

batch_size = 64

docs = []

for i in range(batch_size):
    docs.append(doc)

res = []
for i in range(10):
    t0 = time.time()
    responses = llm.generate(docs, sampling_params)
    t_elap = time.time()-t0
    res.append(t_elap)

print(res)

print("t_elap: %.2f seconds" % (np.median(res)))

#print(responses)

