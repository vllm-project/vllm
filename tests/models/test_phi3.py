import os
import sys
os.environ['NCCL_DEBUG']='WARN'

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import (get_tokenizer)
import json

tp = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

prompts = ['''
William III (William Henry; Dutch: Willem Hendrik; 4 November 1650 – 8 March 1702),[b] also widely known as William of Orange, was the sovereign Prince of Orange from birth, Stadtholder of Holland, Zeeland, Utrecht, Guelders, and Overijssel in the Dutch Republic from the 1670s, and King of England, Ireland, and Scotland from 1689 until his death in 1702. As King of Scotland, he is known as William II.[2] He ruled Britain and Ireland alongside his wife, Queen Mary II, and their joint reign is known as that of William and Mary.

William was the only child of William II, Prince of Orange, and Mary, Princess Royal, the daughter of King Charles I of England, Scotland, and Ireland. His father died a week before his birth, making William III the prince of Orange from birth. In 1677, he married his first cousin Mary, the eldest daughter of his maternal uncle James, Duke of York, the younger brother and later successor of King Charles II.

A Protestant, William participated in several wars against the powerful Catholic French ruler Louis XIV in coalition with both Protestant and Catholic powers in Europe. Many Protestants heralded William as a champion of their faith. In 1685, his Catholic uncle and father-in-law, James, became king of England, Scotland, and Ireland. James's reign was unpopular with the Protestant majority in Britain, who feared a revival of Catholicism. Supported by a group of influential British political and religious leaders, William invaded England in what became known as the Glorious Revolution. In 1688, he landed at the south-western English port of Brixham; James was deposed shortly afterward.
''' * 20 + '''William's reputation ''',

'''
William III (William Henry; Dutch: Willem Hendrik; 4 November 1650 – 8 March 1702),[b] also widely known as William of Orange, was the sovereign Prince of Orange from birth, Stadtholder of Holland, Zeeland, Utrecht, Guelders, and Overijssel in the Dutch Republic from the 1670s, and King of England, Ireland, and Scotland from 1689 until his death in 1702. As King of Scotland, he is known as William II.[2] He ruled Britain and Ireland alongside his wife, Queen Mary II, and their joint reign is known as that of William and Mary.

William was the only child of William II, Prince of Orange, and Mary, Princess Royal, the daughter of King Charles I of England, Scotland, and Ireland. His father died a week before his birth, making William III the prince of Orange from birth. In 1677, he married his first cousin Mary, the eldest daughter of his maternal uncle James, Duke of York, the younger brother and later successor of King Charles II.
'''
]

# prompts = ['The president of Microsoft is ' * 10]
# prompts = [prompts[1]]

# model_path="/data/users/yunanzhang/hf/checkpoints/TLG4.7.3/iter_0078678_hf/"
# model_path='/mnt/std-cache/users/xihlin/checkpoints/tlgv4.7-phase2/tlgv4-copy'
model_path="/data/data/users/bapatra/cache/post-training/phi_7B_phase2_iter_165462_for_inference_20240411"

sampling_params = SamplingParams(temperature=0)
llm = LLM(model=model_path, tokenizer=model_path, enforce_eager=False,
            trust_remote_code=True, block_size=16, tensor_parallel_size=tp)


outputs = llm.generate(prompts, sampling_params)



print(f'>>>> 1st run:')
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")


# -----
exit()

prompts = ['The president of Microsoft is ' * 300, 'Wikipedia\n' * 10 + 'Wikipedia is a']
prompts.append('''
            Nvidia's professional line of GPUs are used for edge-to-cloud computing and in supercomputers and workstations for applications in such fields as architecture, engineering and construction, media and entertainment, automotive, scientific research, and manufacturing design.[9] Its GeForce line of GPUs are aimed at the consumer market and are used in applications such as video editing, 3D rendering and PC gaming. In the second quarter of 2023, Nvidia had a market share of 80.2% in the discrete desktop GPU market.[10] The company expanded its presence in the gaming industry with the introduction of the Shield Portable (a handheld game console), Shield Tablet (a gaming tablet) and Shield TV (a digital media player), as well as its cloud gaming service GeForce Now.[11]

In addition to GPU design and manufacturing, Nvidia provides the CUDA software platform and API that allows the creation of massively parallel programs which utilize GPUs.[12][13] They are deployed in supercomputing sites around the world.[14][15] In the late 2000s, Nvidia had moved into the mobile computing market, where it produces Tegra mobile processors for smartphones and tablets as well as vehicle navigation and entertainment systems.[16][17][18] Its competitors include AMD, Intel,[19] Qualcomm[20] and AI accelerator companies such as Cerebras and Graphcore. It also makes AI-powered software for audio and video'''
)

outputs2 = llm.generate(prompts, sampling_params)


print(f'>>>> 2nd run')
for output in outputs2:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")