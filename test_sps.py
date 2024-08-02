import random
import json
from typing import List, Optional, Tuple
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

prompts = [
    "Pretend that you are one the greatest entrepreneurial thought leaders of our time. Make the for and against case on why someone should work at high growth start-ups versus quantitative trading.",
    "We already have html test reports. The info is in a xml and it is formatted with xsl. We also have javascript code in it.\n\nAll this is old but works as we want, but we want to we want to modernize this. The main reason is that it also uses activex, which is unsafe. It forces us to view the reports in Internet Explorer, which is deprecated, or to use IETab plugin to view them in Chrome.\n\nI am looking for ways to modernize it so wevcan use it on Chrome without plugin and without disabling security features.\n\nThe main reason we use activeX is that we have to access the file system and modify files on it. The test references, which are in tcml files (they are xml in fact, that's just an extension), have to be modified if the developer clicks on an Accept button to accept the new result of a test.",
    # "Please re-write so that question 2 assesses biases without directly mentioning them so that people can answer honestly and not feel defensive. Please re-write question 5 to specify about primary care and its interactions with mental health care",
    # "Cleanup the following: A startup incubator firm specializes in helping startups prepare and launch. The organization is is divided into groups that each focuses on supporting a specific startup. Each group is supposed to led by leader who has prior experience as a startup assistants. In scaling the firm and taking on a new client, a new group is formed without such a leader. A month into supporting this startup, one of their members leaves temporarily to join a startup as they launch, gaining experience as a startup assistant. Upon returning to his original group, his new experience places him as the missing leader of the group. This group now must navigate this disruption of a new hierarchy as well as this leader having been an equal previously. Furthermore, this new leader has returned with new affectations he naturally picked up from the startup group he joined for launch, now appears different from how they remembered him. 2 / 2",
    # "Improve this email to university faculty and staff to encourage feedback about switching from Groupwise & Filr to the possibility of Google Suite or MS Office 365. This needs to be in a welcoming and encouraging tone, but not too formal given the audience. Here's the email to improve: Dear members of the Campus community,\n\nAs previously announced, a project is underway to select a new collaboration platform for the campus. This would include email, calendar, file sharing, video conferencing, team chat, standard document tools that support real-time collaborative editing, as well as the underlying IT architecture to support these services. \n\nAs we gather requirements on a potential future state, the Collaboration Platform Advisory Group invites all Faculty and Staff to provide feedback on their requirements by submitting the survey at: https://uregina.eu.qualtrics.com/jfe/form/SV\\_0Goe2XgXue51qKy\n\nThe survey is open now, and will close on Monday, March 27 at 9:00 AM. It will take about 5 minutes to complete and will assist us in defining requirements. There are 8 short questions, asking how you currently collaborate, what features of a collaboration platform are most important to you, and a request for additional comments.\n\nFurther information on this project is available at: https://ursource.uregina.ca/is/collaboration-project.html\n\nIf you have any questions about the project, or wish to provide further feedback to the advisory group, email can be sent to: Collaboration.Platform@uregina.ca",
    # "I am going to right here some input and I'd like you to use it afterwords.\nProject name: Team work onboarding\nIndustry : Saas product for teams\nAbout project: project management tool. In this project we'll work on the onboarding screen\nMain user personas: people that need project management tools\nDefault language / brand language (playful, informative, informal etc.)\\*: smart, interesting\nBackground story: user just signed up for the platform, now they need to learn how to use it. They need to name their project in this screen\n\nContent elements of this screen:\nHeader\nShort instructional description\nButton\n\nGenerate microcopy for the content elements header, Short instructional description, Button.",
    # "Roy goes to the grocery store to buy a can of juice that has 100 calories of calories and contains 1 gram of sugar. If he buys another can that has 150 calories and 1 gram of sugar and then buys another can that has 200 calories and 2 gram of sugar, how many calories and grams of sugar in total will he buy?",
    # "Which command entered on a switch configured with Rapid PVST\\* listens and learns for a specific\ntime period?\nA. switch(config)#spanning-tree vlan 1 max-age 6\nB. switch(config)#spanning-tree vlan 1 hello-time 10\nC. switch(config)#spanning-tree vlan 1 priority 4096\nD. switch(config)#spanning-tree vlan 1 forward-time 20",
]
tokenizer = AutoTokenizer.from_pretrained("NousResearch/llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=4, stop_token_ids=[tokenizer.eos_token_id])

llm = LLM(
    model="NousResearch/llama-2-7b-chat-hf",                    # "NousResearch/llama-2-7b-chat-hf",
    speculative_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    num_speculative_tokens=5,
    use_v2_block_manager=True,
    enforce_eager=True,
    gpu_memory_utilization = 0.9,
    max_num_seqs=16
)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")

print("------------------------------------------------")