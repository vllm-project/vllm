r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import io
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(prefix_lines +
                                random.choices(poem_lines, k=num_lines_needed))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len, None))

    return sampled_requests


def sample_mmmu_pro_vision_requests(
    dataset,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:
    sampled_requests: List[Tuple[str, int, int, Dict[str,
                                                     Collection[str]]]] = []
    for data in dataset:
        if len(sampled_requests) == num_requests:
            break

        # MMMU-Pro vision direct prompt
        # Ref: https://github.com/MMMU-Benchmark/MMMU/blob/6ce42f4d8f70c1841c67867152648974415b5cac/mmmu-pro/prompts.yaml#L5
        prompt = (
            "Answer with the option letter from the given choices directly. "
            "The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
            "options.")

        prompt_token_ids = tokenizer(prompt).input_ids
        if fixed_output_len is None:
            # Default max output len is set to 128
            print("--hf-output-len is not provided. Using default value 128.")
            fixed_output_len = 128

        prompt_len = len(prompt_token_ids)
        output_len = fixed_output_len

        assert isinstance(
            data["image"],
            Image), ("Input image format must be `PIL.Image.Image`, "
                     f"given {type(data['image'])}.")
        image: Image = data["image"]
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        mm_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_hf_requests(
    dataset_path: str,
    dataset_subset: str,
    dataset_split: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    random_seed: int,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, str, int, Optional[Dict[str, Collection[str]]]]]:

    # Special case for MMMU-Pro vision dataset
    if dataset_path == 'MMMU/MMMU_Pro' and dataset_subset == 'vision':
        assert dataset_split == "test"
        dataset = load_dataset(dataset_path,
                               name=dataset_subset,
                               split=dataset_split,
                               streaming=True)
        assert "image" in dataset.features, (
            "MMMU/MMMU_Pro vision dataset must have 'image' column.")
        filter_func = lambda x: isinstance(x["image"], Image)
        dataset = dataset.shuffle(seed=random_seed).filter(filter_func)
        return sample_mmmu_pro_vision_requests(dataset, num_requests,
                                               tokenizer, fixed_output_len)

    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split,
                           streaming=True)
    assert "conversations" in dataset.features, (
        "HF Dataset must have 'conversations' column.")
    filter_func = lambda x: len(x["conversations"]) >= 2
    filtered_dataset = dataset.shuffle(seed=random_seed).filter(filter_func)
    sampled_requests: List[Tuple[str, int, int, Dict[str,
                                                     Collection[str]]]] = []
    for data in filtered_dataset:
        if len(sampled_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = data["conversations"][0]["value"]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = data["conversations"][1]["value"]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if fixed_output_len is None and (prompt_len < 4 or output_len < 4):
            # Prune too short sequences.
            continue
        if fixed_output_len is None and \
            (prompt_len > 1024 or prompt_len + output_len > 2048):
            # Prune too long sequences.
            continue

        if "image" in data and isinstance(data["image"], Image):
            image: Image = data["image"]
            image = image.convert("RGB")
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
            mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }
        elif "image" in data and isinstance(data["image"], str):
            if (data["image"].startswith("http://") or \
                data["image"].startswith("file://")):
                image_url = data["image"]
            else:
                image_url = f"file://{data['image']}"

            mm_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            }
        else:
            mm_content = None

        sampled_requests.append((prompt, prompt_len, output_len, mm_content))

    return sampled_requests


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []

    prompt="""
MANY YEARS LATER as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice. At that time Macondo was a village of twenty adobe houses, built on the bank of a river of clear water that ran along a bed of polished stones, which were white and enormous, like prehistoric eggs. The world was so recent that many things lacked names, and in order to indicate them it was necessary to point. Every year during the month of March a family of ragged gypsies would set up their tents near the village, and with a great uproar of pipes and kettledrums they would display new inventions. First they brought the magnet. A heavy gypsy with an untamed beard and sparrow hands, who introduced himself as Melquíades, put on a bold public demonstration of what he himself called the eighth wonder of the learned alchemists of Macedonia. He went from house to house dragging two metal ingots and everybody was amazed to see pots, pans, tongs, and braziers tumble down from their places and beams creak from the desperation of nails and screws trying to emerge, and even objects that had been lost for a long time appeared from where they had been searched for most and went dragging along in turbulent confusion behind Melquíades’ magical irons. “Things have a life of their own,” the gypsy proclaimed with a harsh accent. “It’s simply a matter of waking up their souls.” José Arcadio Buendía, whose unbridled imagination always went beyond the genius of nature and even beyond miracles and magic, thought that it would be possible to make use of that useless invention to extract gold from the bowels of the earth. Melquíades, who was an honest man, warned him: “It won’t work for that.” But José Arcadio Buendía at that time did not believe in the honesty of gypsies, so he traded his mule and a pair of goats for the two magnetized ingots. Úrsula Iguarán, his wife, who relied on those animals to increase their poor domestic holdings, was unable to dissuade him. “Very soon well have gold enough and more to pave the floors of the house,” her husband replied. For several months he worked hard to demonstrate the truth of his idea. He explored every inch of the region, even the riverbed, dragging the two iron ingots along and reciting Melquíades’ incantation aloud. The only thing he succeeded in doing was to unearth a suit of fifteenth-century armor which had all of its pieces soldered together with rust and inside of which there was the hollow resonance of an enormous stone-filled gourd. When José Arcadio Buendía and the four men of his expedition managed to take the armor apart, they found inside a calcified skeleton with a copper locket containing a woman’s hair around its neck. In March the gypsies returned. This time they brought a telescope and a magnifying glass the size of a drum, which they exhibited as the latest discovery of the Jews of Amsterdam. They placed a gypsy woman at one end of the village and set up the telescope at the entrance to the tent. For the price of five reales, people could look into the telescope and see the gypsy woman an arm’s length away. “Science has eliminated distance,” Melquíades proclaimed. “In a short time, man will be able to see what is happening in any place in the world without leaving his own house.” A burning noonday sun brought out a startling demonstration with the gigantic magnifying glass: they put a pile of dry hay in the middle of the street and set it on fire by concentrating the sun’s rays. José Arcadio Buendía, who had still not been consoled for the failure of big magnets, conceived the idea of using that invention as a weapon of war. Again Melquíades tried to dissuade him, but he finally accepted the two magnetized ingots and three colonial coins in exchange for the magnifying glass. Úrsula wept in consternation. That money was from a chest of gold coins that her father had put together ova an entire life of privation and that she had buried underneath her bed in hopes of a proper occasion to make use of it. José Arcadio Buendía made no at. tempt to console her, completely  absorbed in his tactical experiments with the abnegation of a scientist and even at the risk of his own life. In an attempt to show the effects of the glass on enemy troops, he exposed himself to the concentration of the sun’s rays and suffered burns which turned into sores that took a long time to heal. Over the protests of his wife, who was alarmed at such a dangerous invention, at one point he was ready to set the house on fire. He would spend hours on end in his room, calculating the strategic possibilities of his novel weapon until he succeeded in putting together a manual of startling instructional clarity and an irresistible power of conviction. He sent it to the government, accompanied by numerous descriptions of his experiments and several pages of explanatory sketches; by a messenger who crossed the mountains, got lost in measureless swamps, forded stormy rivers, and was on the point of perishing under the lash of despair, plague, and wild beasts until he found a route that joined the one used by the mules that carried the mail. In spite of the fact that a trip to the capital was little less than impossible at that time, José Arcadio Buendía promised to undertake it as soon as the government ordered him to so that he could put on some practical demonstrations of his invention for the military authorities and could train them himself in the complicated art of solar war. For several years he waited for an answer. Finally, tired of waiting, he bemoaned to Melquíades the failure of his project and the gypsy then gave him a convincing proof of his honesty: he gave him back the doubloons in exchange for the magnifying glass, and he left him in addition some Portuguese maps and several instruments of navigation. In his own handwriting he set down a concise synthesis of the studies by Monk Hermann. which he left José Arcadio so that he would be able to make use of the astrolabe, the compass, and the sextant. José Arcadio Buendía spent the long months of the rainy season shut up in a small room that he had built in the rear of the house so that no one would disturb his experiments. Having completely abandoned his domestic obligations, he spent entire nights in the courtyard watching the course of the stars and he almost contracted sunstroke from trying to establish an exact method to ascertain noon. When he became an expert in the use and manipulation of his instruments, he conceived a notion of space that allowed him to navigate across unknown seas, to visit uninhabited territories, and to establish relations with splendid beings without having to leave his study. That was the period in which he acquired the habit of talking to himself, of walking through the house without paying attention to anyone, as Úrsula and the children broke their backs in the garden, growing banana and caladium, cassava and yams, ahuyama roots and eggplants. Suddenly, without warning, his feverish activity was interrupted and was replaced by a kind of fascination. He spent several days as if he were bewitched, softly repeating to himself a string of fearful conjectures without giving credit to his own understanding. Finally, one Tuesday in December, at lunchtime, all at once he released the whole weight of his torment. The children would remember for the rest of their lives the august solemnity with which their father, devastated by his prolonged vigil and by the wrath of his imagination, revealed his discovery to them: “The earth is round, like an orange.” Úrsula lost her patience. “If you have to go crazy, please go crazy all by yourself!” she shouted. “But don’t try to put your gypsy ideas into the heads of the children.” José Arcadio Buendía, impassive, did not let himself be frightened by the desperation of his wife, who, in a seizure of rage, mashed the astrolabe against the floor. He built another one, he gathered the men of the village in his little room, and he demonstrated to them, with theories that none of them could understand, the possibility of returning to where one had set out by consistently sailing east. The whole village was convinced that José Arcadio Buendía had lost his reason, when Melquíades returned to set things straight. He gave public praise to the intelligence of a man who from pure astronomical speculation had evolved a theory that had already been proved in practice, although unknown in Macondo until then, and as a proof of his admiration he made him a gift that was to have a profound influence on the future of the village: the laboratory of an alchemist.  By then Melquíades had aged with surprising rapidity. On his first trips he seemed to be the same age as José Arcadio Buendía. But while the latter had preserved his extraordinary strength, which permitted him to pull down a horse by grabbing its ears, the gypsy seemed to have been worn dowse by some tenacious illness. It was, in reality, the result of multiple and rare diseases contracted on his innumerable trips around the world. According to what he himself said as he spoke to José Arcadio Buendía while helping him set up the laboratory, death followed him everywhere, sniffing at the cuffs of his pants, but never deciding to give him the final clutch of its claws. He was a fugitive from all the plagues and catastrophes that had ever lashed mankind. He had survived pellagra in Persia, scurvy in the Malayan archipelago, leprosy in Alexandria, beriberi in Japan, bubonic plague in Madagascar, an earthquake in Sicily, and a disastrous shipwreck in the Strait of Magellan. That prodigious creature, said to possess the keys of Nostradamus, was a gloomy man, enveloped in a sad aura, with an Asiatic look that seemed to know what there was on the other side of things. He wore a large black hat that looked like a raven with widespread wings, and a velvet vest across which the patina of the centuries had skated. But in spite of his immense wisdom and his mysterious breadth, he had a human burden, an earthly condition that kept him involved in the small problems of daily life. He would complain of the ailments of old age, he suffered from the most insignificant economic difficulties, and he had stopped laughing a long time back because scurvy had made his teeth drop out. On that suffocating noontime when the gypsy revealed his secrets, José Arcadio Buendía had the certainty that it was the beginning of a great friendship. The children were startled by his fantastic stories. Aureliano, who could not have been more than five at the time, would remember him for the rest of his life as he saw him that afternoon, sitting against the metallic and quivering light from the window, lighting up with his deep organ voice the darkest reaches of the imagination, while down over his temples there flowed the grease that was being melted by the heat. José Arcadio, his older brother, would pass on that wonderful image as a hereditary memory to all of his descendants. Úrsula on the other hand, held a bad memory of that visit, for she had entered the room just as Melquíades had carelessly broken a flask of bichloride of mercury. “It’s the smell of the devil,” she said. “Not at all,” Melquíades corrected her. “It has been proven that the devil has sulphuric properties and this is just a little corrosive sublimate.” Always didactic, he went into a learned exposition of the diabolical properties of cinnabar, but Úrsula paid no attention to him, although she took the children off to pray. That biting odor would stay forever in her mind linked to the memory of Melquíades. The rudimentary laboratory—in addition to a profusion of pots, funnels, retorts, filters, and sieves—was made up of a primitive water pipe, a glass beaker with a long, thin neck, a reproduction of the philosopher’s egg, and a still the gypsies themselves had built in accordance with modern descriptions of the three-armed alembic of Mary the Jew. Along with those items, Melquíades left samples of the seven metals that corresponded to the seven planets, the formulas of Moses and Zosimus for doubling the quantity of gold, and a set of notes and sketches concerning the processes of the Great Teaching that would permit those who could interpret them to undertake the manufacture of the philosopher’s stone. Seduced by the simplicity of the formulas to double the quantity of gold, José Arcadio Buendía paid court to Úrsula for several weeks so that she would let him dig up her colonial coins and increase them by as many times as it was possible to subdivide mercury. Úrsula gave in, as always, to her husband’s unyielding obstinacy. Then José Arcadio Buendía threw three doubloons into a pan and fused them with copper filings, orpiment, brimstone, and lead. He put it all to boil in a pot of castor oil until he got a thick and pestilential syrup which was more like common caramel than valuable gold. In risky and desperate processes of distillation, melted with the seven planetary metals, mixed with hermetic mercury and vitriol of Cyprus, and put  back to cook in hog fat for lack of any radish oil, Úrsula’s precious inheritance was reduced to a large piece of burnt hog cracklings that was firmly stuck to the bottom of the pot. When the gypsies came back, Úrsula had turned the whole population of the village against them. But curiosity was greater than fear, for that time the gypsies went about the town making a deafening noise with all manner of musical instruments while a hawker announced the exhibition of the most fabulous discovery of the Naciancenes. So that everyone went to the tent and by paying one cent they saw a youthful Melquíades, recovered, unwrinkled, with a new and flashing set of teeth. Those who remembered his gums that had been destroyed by scurvy, his flaccid cheeks, and his withered lips trembled with fear at the final proof of the gypsy’s supernatural power. The fear turned into panic when Melquíades took out his teeth, intact, encased in their gums, and showed them to the audience for an instant—a fleeting instant in which he went back to being the same decrepit man of years past—and put them back again and smiled once more with the full control of his restored youth. Even José Arcadio Buendía himself considered that Melquíades’ knowledge had reached unbearable extremes, but he felt a healthy excitement when the gypsy explained to him atone the workings of his false teeth. It seemed so simple and so prodigious at the same time that overnight he lost all interest in his experiments in alchemy. He underwent a new crisis of bad humor. He did not go back to eating regularly, and he would spend the day walking through the house. “Incredible things are happening in the world,” he said to Úrsula. “Right there across the river there are all kinds of magical instruments while we keep on living like donkeys.” Those who had known him since the foundation of Macondo were startled at how much he had changed under Melquíades’ influence. At first José Arcadio Buendía had been a kind of youthful patriarch who would give instructions for planting and advice for the raising of children and animals, and who collaborated with everyone, even in the physical work, for the welfare of the community. Since his house from the very first had been the best in the village, the others had been built in its image and likeness. It had a small, welllighted living roost, a dining room in the shape of a terrace with gaily colored flowers, two bedrooms, a courtyard with a gigantic chestnut tree, a well kept garden, and a corral where goats, pigs, and hens lived in peaceful communion. The only animals that were prohibited, not just in his house but in the entire settlement, were fighting cocks. Úrsula’s capacity for work was the same as that of her husband. Active, small, severe, that woman of unbreakable nerves who at no moment in her life had been heard to sing seemed to be everywhere, from dawn until quite late at night, always pursued by the soft whispering of her stiff, starched petticoats. Thanks to her the floors of tamped earth, the unwhitewashed mud walls, the rustic, wooden furniture they had built themselves were always dean, and the old chests where they kept their clothes exhaled the warm smell of basil. José Arcadio Buendía, who was the most enterprising man ever to be seen in the village, had set up the placement of the houses in such a way that from all of them one could reach the river and draw water with the same effort, and he had lined up the streets with such good sense that no house got more sun than another during the hot time of day. Within a few years Macondo was a village that was more orderly and hard working than any known until then by its three hundred inhabitants. It was a truly happy village where no one was over thirty years of age and where no one had died. Since the time of its founding, José Arcadio Buendía had built traps and cages. In a short time he filled not only his own house but all of those in the village with troupials, canaries, bee eaters, and redbreasts. The concert of so many different birds became so disturbing that Úrsula would plug her ears with beeswax so as not to lose her sense of reality. The first time that Melquíades’ tribe arrived, selling glass balls for headaches, everyone was surprised that they had been able to find that village lost in the drowsiness of the swamp, and the gypsies confessed that they had found their way by the song of the birds.  That spirit of social initiative disappeared in a short time, pulled away by the fever of the magnets, the astronomical calculations, the dreams of transmutation, and the urge to discover the wonders of the world. From a clean and active man, José Arcadio Buendía changed into a man lazy in appearance, careless in his dress, with a wild beard that Úrsula managed to trim with great effort and a kitchen knife. There were many who considered him the victim of some strange spell. But even those most convinced of his madness left work and family to follow him when he brought out his tools to clear the land and asked the assembled group to open a way that would put Macondo in contact with the great inventions. José Arcadio Buendía was completely ignorant of the geography of the region. He knew that to the east there lay an impenetrable mountain chain and that on the other side of the mountains there was the ardent city of Riohacha, where in times past—according to what he had been told by the first Aureliano Buendía, his grandfather—Sir Francis Drake had gone crocodile hunting with cannons and that he repaired hem and stuffed them with straw to bring to Queen Elizabeth. In his youth, José Arcadio Buendía and his men, with wives and children, animals and all kinds of domestic implements, had crossed the mountains in search of an outlet to the sea, and after twenty-six months they gave up the expedition and founded Macondo, so they would not have to go back. It was, therefore, a route that did not interest him, for it could lead only to the past. To the south lay the swamps, covered with an eternal vegetable scum and the whole vast universe of the great swamp, which, according to what the gypsies said, had no limits. The great swamp in the west mingled with a boundless extension of water where there were soft-skinned cetaceans that had the head and torso of a woman, causing the ruination of sailors with the charm of their extraordinary breasts. The gypsies sailed along that route for six months before they reached the strip of land over which the mules that carried the mail passed. According to José Arcadio Buendía’s calculations, the only possibility of contact with civilization lay along the northern route. So he handed out clearing tools and hunting weapons to the same men who had been with him during the founding of Macondo. He threw his directional instruments and his maps into a knapsack, and he undertook the reckless adventure. During the first days they did not come across any appreciable obstacle. They went down along the stony bank of the river to the place where years before they had found the soldier’s armor, and from there they went into the woods along a path between wild orange trees. At the end of the first week they killed and roasted a deer, but they agreed to eat only half of it and salt the rest for the days that lay ahead. With that precaution they tried to postpone the necessity of having to eat macaws, whose blue flesh had a harsh and musky taste. Then, for more than ten days, they did not see the sun again. The ground became soft and damp, like volcanic ash, and the vegetation was thicker and thicker, and the cries of the birds and the uproar of the monkeys became more and more remote, and the world became eternally sad. The men on the expedition felt overwhelmed by their most ancient memories in that paradise of dampness and silence, going back to before original sin, as their boots sank into pools of steaming oil and their machetes destroyed bloody lilies and golden salamanders. For a week, almost without speaking, they went ahead like sleepwalkers through a universe of grief, lighted only by the tenuous reflection of luminous insects, and their lungs were overwhelmed by a suffocating smell of blood. They could not return because the strip that they were opening as they went along would soon close up with a new vegetation that. almost seemed to grow before their eyes. “It’s all right,” José Arcadio Buendía would say. “The main thing is not to lose our bearings.” Always following his compass, he kept on guiding his men toward the invisible north so that they would be able to get out of that enchanted region. It was a thick night, starless, but the darkness was becoming impregnated with a fresh and clear air. Exhausted by the long crossing, they hung up their hammocks and slept deeply for the first time in two weeks. When they woke up, with the sun already high in the sky, they were speechless with fascination. Before them, surrounded by  ferns and palm trees, white and powdery in the silent morning light, was an enormous Spanish galleon. Tilted slightly to the starboard, it had hanging from its intact masts the dirty rags of its sails in the midst of its rigging, which was adorned with orchids. The hull, covered with an armor of petrified barnacles and soft moss, was firmly fastened into a surface of stones. The whole structure seemed to occupy its own space, one of solitude and oblivion, protected from the vices of time and the habits of the birds. Inside, where the expeditionaries explored with careful intent, there was nothing but a thick forest of flowers. The discovery of the galleon, an indication of the proximity of the sea, broke José Arcadio Buendía’s drive. He considered it a trick of his whimsical fate to have searched for the sea without finding it, at the cost of countless sacrifices and suffering, and to have found it all of a sudden without looking for it, as if it lay across his path like an insurmountable object. Many years later Colonel Aureliano Buendía crossed the region again, when it was already a regular mail route, and the only part of the ship he found was its burned-out frame in the midst of a field of poppies. Only then, convinced that the story had not been some product of his father’s imagination, did he wonder how the galleon had been able to get inland to that spot. But José Arcadio Buendía did not concern himself with that when he found the sea after another four days’ journey from the galleon. His dreams ended as he faced that ashen, foamy, dirty sea, which had not merited the risks and sacrifices of the adventure. “God damn it!” he shouted. “Macondo is surrounded by water on all sides.” The idea of a peninsular Macondo prevailed for a long time, inspired by the arbitrary map that José Arcadio Buendía sketched on his return from the expedition. He drew it in rage, evilly, exaggerating the difficulties of communication, as if to punish himself for the absolute lack of sense with which he had chosen the place. “We’ll never get anywhere,” he lamented to Úrsula. “We’re going to rot our lives away here without receiving the benefits of science.” That certainty, mulled over for several months in the small room he used as his laboratory, brought him to the conception of the plan to move Maeondo to a better place. But that time Úrsula had anticipated his feverish designs. With the secret and implacable labor of a small ant she predisposed the women of the village against the flightiness of their husbands, who were already preparing for the move. José Arcadio Buendía did not know at what moment or because of what adverse forces his plan had become enveloped in a web of pretexts, disappointments, and evasions until it turned into nothing but an illusion. Úrsula watched him with innocent attention and even felt some pity for him on the morning when she found him in the back room muttering about his plans for moving as he placed his laboratory pieces in their original boxes. She let him finish. She let him nail up the boxes and put his initials on them with an inked brush, without reproaching him, but knowing now that he knew (because she had heard him say so in his soft monologues) that the men of the village would not back him up in his undertaking. Only when he began to take down the door of the room did Úrsula dare ask him what he was doing, and he answered with a certain bitterness. “Since no one wants to leave, we’ll leave all by ourselves.” Úrsula did not become upset. “We will not leave,” she said. “We will stay here, because we have had a son here.” “We have still not had a death,” he said. “A person does not belong to a place until there is someone dead under the ground.” Úrsula replied with a soft firmness: “If I have to die for the rest of you to stay here, I will die.” José Arcadio Buendía had not thought that his wife’s will was so firm. He tried to seduce her with the charm of his fantasy, with the promise of a prodigious world where all one had to do was sprinkle some magic liquid on the ground and the plants would bear fruit whenever a man wished, and where all manner of instruments against pain were sold at bargain prices. But Úrsula was insensible to his clairvoyance. “Instead of going around thinking about your crazy inventions, you should be worrying about your sons,” she replied. “Look at the state they’re in, running wild just like donkeys.” José Arcadio Buendía took his wife’s words literally. He looked out the window and saw the barefoot children in the sunny garden and he had the impression that only at that instant had they begun to exist, conceived by Úrsula’s spell, Something occurred inside of him then, something mysterious and definitive that uprooted him from his own time and carried him adrift through an unexplored region of his memory. While Úrsula continued sweeping the house, which was safe now from being abandoned for the rest of her life, he stood there with an absorbed look, contemplating the children until his eyes became moist and he dried them with the back of his hand, exhaling a deep sigh of resignation. “All right,” he said. “Tell them to come help me take the things out of the boxes.” José Arcadio, the older of the children, was fourteen. He had a square head, thick hair, and his father’s character. Although he had the same impulse for growth and physical strength, it was early evident that he lacked imagination. He had been conceived and born during the difficult crossing of the mountains, before the founding of Macondo, and his parents gave thanks to heaven when they saw he had no animal features. Aureliano, the first human being to be born in Macondo, would be six years old in March. He was silent and withdrawn. He had wept in his mother’s womb and had been born with his eyes open. As they were cutting the umbilical cord, he moved his head from side to side, taking in the things in the room and examining the faces of the people with a fearless curiosity. Then, indifferent to those who came close to look at him, he kept his attention concentrated on the palm roof, which looked as if it were about to collapse under the tremendous pressure of the rain. Úrsula did not remember the intensity of that look again until one day when little Aureliano, at the age of three, went into the kitchen at the moment she was taking a pot of boiling soup from the stove and putting it on the table. The child, Perplexed, said from the doorway, “It’s going to spill.” The pot was firmly placed in the center of the table, but just as soon as the child made his announcement, it began an unmistakable movement toward the edge, as if impelled by some inner dynamism, and it fell and broke on the floor. Úrsula, alarmed, told her husband about the episode, but he interpreted it as a natural phenomenon. That was the way he always was alien to the existence of his sons, partly because he considered childhood as a period of mental insufficiency, and partly because he was always too absorbed in his fantastic speculations. But since the afternoon when he called the children in to help him unpack the things in the laboratory, he gave them his best hours. In the small separate room, where the walls were gradually being covered by strange maps and fabulous drawings, he taught them to read and write and do sums, and he spoke to them about the wonders of the world, not only where his learning had extended, but forcing the limits of his imagination to extremes. It was in that way that the boys ended up learning that in the southern extremes of Africa there were men so intelligent and peaceful that their only pastime was to sit and think, and that it was possible to cross the Aegean Sea on foot by jumping from island to island all the way to the port of Salonika. Those hallucinating sessions remained printed on the memories of the boys in such a way that many years later, a second before the regular army officer gave the firing squad the command to fire, Colonel Aureliano Buendía saw once more that warm March afternoon on which his father had interrupted the lesson in physics and stood fascinated, with his hand in the air and his eyes motionless, listening to the distant pipes, drums, and jingles of the gypsies, who were coming to the village once more, announcing the latest and most startling discovery of the sages of Memphis. They were new gypsies, young men and women who knew only their own language, handsome specimens with oily skins and intelligent hands, whose dances and music sowed a panic of uproarious joy through the streets, with parrots painted all colors reciting Italian arias, and a hen who laid a hundred golden eggs to the sound of a tambourine, and a trained monkey who read minds, and the multi-use machine that could be used at the same time to sew on buttons and reduce fevers, and the apparatus to make a person forget his bad memories, and a poultice to lose time, and a thousand more inventions so ingenious and unusual that José Arcadio Buendía must have wanted to invent a memory machine so that he could remember them all. In an instant they transformed the village. The inhabitants of Macondo found themselves lost is their own streets, confused by the crowded fair. Holding a child by each hand so as not to lose them in the tumult, bumping into acrobats with gold-capped teeth and jugglers with six arms, suffocated by the mingled breath of manure and sandals that the crowd exhaled, José Arcadio Buendía went about everywhere like a madman, looking for Melquíades so that he could reveal to him the infinite secrets of that fabulous nightmare. He asked several gypsies, who did not understand his language. Finally he reached the place where Melquíades used to set up his tent and he found a taciturn Armenian who in Spanish was hawking a syrup to make oneself invisible. He had drunk down a glass of the amber substance in one gulp as José Arcadio Buendía elbowed his way through the absorbed group that was witnessing the spectacle, and was able to ask his question. The gypsy wrapped him in the frightful climate of his look before he turned into a puddle of pestilential and smoking pitch over which the echo of his reply still floated: “Melquíades is dead.” Upset by the news, José Arcadio Buendía stood motionless, trying to rise above his affliction, until the group dispersed, called away by other artifices, and the puddle of the taciturn Armenian evaporated completely. Other gypsies confirmed later on that Melquíades had in fact succumbed to the fever on the beach at Singapore and that his body had been thrown into the deepest part of the Java Sea. The children had no interest in the news. They insisted that their father take them to see the overwhelming novelty of the sages of Memphis that was being advertised at the entrance of a tent that, according to what was said, had belonged to King Solomon. They insisted so much that José Arcadio Buendía paid the thirty reales and led them into the center of the tent, where there was a giant with a hairy torso and a shaved head, with a copper ring in his nose and a heavy iron chain on his ankle, watching over a pirate chest. When it was opened by the giant, the chest gave off a glacial exhalation. Inside there was only an enormous, transparent block with infinite internal needles in which the light of the sunset was broken up into colored stars. Disconcerted, knowing that the children were waiting for an immediate explanation, José Arcadio Buendía ventured a murmur: “It’s the largest diamond in the world.” “No,” the gypsy countered. “It’s ice.” José Arcadio Buendía, without understanding, stretched out his hand toward the cake, but the giant moved it away. “Five reales more to touch it,” he said. José Arcadio Buendía paid them and put his hand on the ice and held it there for several minutes as his heart filled with fear and jubilation at the contact with mystery. Without knowing what to say, he paid ten reales more so that his sons could have that prodigious experience. Little José Arcadio refused to touch it. Aureliano, on the other hand, took a step forward and put his hand on it, withdrawing it immediately. “It’s boiling,” he exclaimed, startled. But his father paid no attention to him. Intoxicated by the evidence of the miracle, he forgot at that moment about the frustration of his delirious undertakings and Melquíades’ body, abandoned to the appetite of the squids. He paid another five reales and with his hand on the cake, as if giving testimony on the holy scriptures, he exclaimed: “This is the great invention of our time.” WHEN THE PIRATE Sir Francis Drake attacked Riohacha in the sixteenth century, Úrsula Iguarán’s great-great-grandmother became so frightened with the ringing of alarm bells and the firing of cannons that she lost control of her nerves and sat down on a lighted stove. The burns changed her into a useless wife for the rest of her days. She could only sit on one side, cushioned by pillows, and something strange must have happened to her way of walking, for she never walked again in public. She gave up all kinds of social activity, obsessed with the notion that her body gave off a singed odor. Dawn would find her in the courtyard, for she did not dare fall asleep lest she dream of the English and their ferocious attack dogs as they came through the windows of her bedroom to submit her to shameful tortures with their red-hot irons. Her husband, an Aragonese merchant by whom she had two children, spent half the value of his store on medicines and pastimes in an attempt to alleviate her terror. Finally he sold the business and took the family to live far from the sea in a settlement of peaceful Indians located in the foothills, where he built his wife a bedroom without windows so that the pirates of her dream would have no way to get in. In that hidden village there was a native-born tobacco planter who had lived there for some time, Don José Arcadio Buendía, with whom Úrsula’s great-great-grandfather established a partnership that was so lucrative that within a few years they made a fortune. Several centuries later the greatgreat-grandson of the native-born planter married the great-great-granddaughter of the Aragonese. Therefore, every time that Úrsula became exercised over her husband’s mad ideas, she would leap back over three hundred years of fate and curse the day that Sir Francis Drake had attacked Riohacha. It was simply a way. of giving herself some relief, because actually they were joined till death by a bond that was more solid that love: a common prick of conscience. They were cousins. They had grown up together in the old village that both of their ancestors, with their work and their good habits, had transformed into one of the finest towns in the province. Although their marriage was predicted from the time they had come into the world, when they expressed their desire to be married their own relatives tried to stop it. They were afraid that those two healthy products of two races that had interbred over the centuries would suffer the shame of breeding iguanas. There had already been a horrible precedent. An aunt of Úrsula’s, married to an uncle of José Arcadio Buendía, had a son who went through life wearing loose, baggy trousers and who bled to death after having lived forty-two years in the purest state of virginity, for he had been born and had grown up with a cartilaginous tail in the shape of a corkscrew and with a small tuft of hair on the tip. A pig’s tail that was never allowed to be seen by any woman and that cost him his life when a butcher friend did him the favor of chopping it off with his cleaver. José Arcadio Buendía, with the whimsy of his nineteen years, resolved the problem with a single phrase: “I don’t care if I have piglets as long as they can talk.” So they were married amidst a festival of fireworks and a brass band that went on for three days. They would have been happy from then on if Úrsula’s mother had not terrified her with all manner of sinister predictions about their offspring, even to the extreme of advising her to refuse to consummate the marriage. Fearing that her stout and willful husband would rape her while she slept, Úrsula, before going to bed, would put on a rudimentary kind of drawers that her mother had made out of sailcloth and had reinforced with a system of crisscrossed leather straps and that was closed in the front by a thick iron buckle. That was how they lived for several months. During the day he would take care of his fighting cocks and she would do frame embroidery with her mother. At night they would wrestle for several hours in an anguished violence that seemed to be a substitute for the act of love, until popular intuition got a whiff of something irregular and the rumor spread that  Úrsula was still a virgin a year after her marriage because her husband was impotent. José Arcadio Buendía was the last one to hear the rumor. “Look at what people are going around saying, Úrsula,” he told his wife very calmly. “Let them talk,” she said. “We know that it’s not true.” So the situation went on the same way for another six months until that tragic Sunday when José Arcadio Buendía won a cockfight from Prudencio Aguilar. Furious, aroused by the blood of his bird, the loser backed away from José Arcadio Buendía so that everyone in the cockpit could hear what he was going to tell him. “Congratulations!” he shouted. “Maybe that rooster of yours can do your wife a favor.” José Arcadio Buendía serenely picked up his rooster. “I’ll be right back,” he told everyone. And then to Prudencio Aguilar: “You go home and get a weapon, because I’m going to kill you.” Ten minutes later he returned with the notched spear that had belonged to his grandfather. At the door to the cockpit, where half the town had gathered, Prudencio Aguilar was waiting for him. There was no time to defend himself. José Arcadio Buendía’s spear, thrown with the strength of a bull and with the same good aim with which the first Aureliano Buendía had exterminated the jaguars in the region, pierced his throat. That night, as they held a wake over the corpse in the cockpit, José Arcadio Buendía went into the bedroom as his wife was putting on her chastity pants. Pointing the spear at her he ordered: “Take them off.” Úrsula had no doubt about her husband’s decision. “You’ll be responsible for what happens,” she murmured. José Arcadio Buendía stuck the spear into the dirt floor. “If you bear iguanas, we’ll raise iguanas,” he said. “But there’ll be no more killings in this town because of you.” It was a fine June night, cool and with a moon, and they were awake and frolicking in bed until dawn, indifferent to the breeze that passed through the bedroom, loaded with the weeping of Prudencio Aguilar’s kin. The matter was put down as a duel of honor, but both of them were left with a twinge in their conscience. One night, when she could not sleep, Úrsula went out into the courtyard to get some water and she saw Prudencio Aguilar by the water jar. He was livid, a sad expression on his face, trying to cover the hole in his throat with a plug made of esparto grass. It did not bring on fear in her, but pity. She went back to the room and told her husband what she had seen, but he did not think much of it. “This just means that we can’t stand the weight of our conscience.” Two nights later Úrsula saw Prudencio Aguilar again, in the bathroom, using the esparto plug to wash the clotted blood from his throat. On another night she saw him strolling in the rain. José Arcadio Buendía, annoyed by his wife’s hallucinations, went out into the courtyard armed with the spear. There was the dead man with his sad expression. “You go to hell,” José Arcadio Buendía shouted at him. “Just as many times as you come back, I’ll kill you again.” Prudencio Aguilar did not go away, nor did José Arcadio Buendía dare throw the spear. He never slept well after that. He was tormented by the immense desolation with which the dead man had looked at him through the rain, his deep nostalgia as he yearned for living people, the anxiety with which he searched through the house looking for some water with which to soak his esparto plug. “He must be suffering a great deal,” he said to Úrsula. “You can see that he’s so very lonely.” She was so moved that the next time she saw the dead man uncovering the pots on the stove she understood what he was looking for, and from then on she placed water jugs all about the house. One night when he found him washing his wound in his own room, José Anedio Buendía could no longer resist.  “It’s all right, Prudencio,” he told him. “We’re going to leave this town, just as far away as we can go, and we’ll never come back. Go in peace now.” That was how they undertook the crossing of the mountains. Several friends of José Arcadio Buendía, young men like him, excited, by the adventure, dismantled their houses and packed up, along with their wives and children, to head toward the land that no one had promised them. Before he left, José Arcadio Buendía buried the spear in the courtyard and, one after the other, he cut the throats of his magnificent fighting cocks, trusting that in that way he could give some measure of peace to Prudencio Aguilar. All that Úrsula took along were a trunk with her bridal clothes, a few household utensils, and the small chest with the gold pieces that she had inherited from her father. They did not lay out any definite itinerary. They simply tried to go in a direction opposite to the road to Riohacha so that they would not leave any trace or meet any people they knew. It was an absurd journey. After fourteen months, her stomach corrupted by monkey meat and snake stew, Úrsula gave birth to a son who had all of his features human. She had traveled half of the trip in a hammock that two men carried on their shoulders, because swelling had disfigured her legs and her varicose veins had puffed up like bubbles. Although it was pitiful to see them with their sunken stomachs and languid eyes, the children survived the journey better than their parents, and most of the time it was fun for them. One morning, after almost two years of crossing, they became the first mortals to see the western slopes of the mountain range. From the cloudy summit they saw the immense aquatic expanse of the great swamp as it spread out toward the other side of the world. But they never found the sea. One night, after several months of lost wandering through the swamps, far away now from the last Indians they had met on their way, they camped on the banks of a stony river whose waters were like a torrent of frozen glass. Years later, during the second civil war, Colonel Aureliano Buendía tried to follow that same route in order to take Riohacha by surprise and after six days of traveling he understood that it was madness. Nevertheless, the night on which they camped beside the river, his father’s host had the look of shipwrecked people with no escape, but their number had grown during the crossing and they were all prepared (and they succeeded) to die of old age. José Arcadio Buendía dreamed that night that right there a noisy city with houses having mirror wails rose up. He asked what city it was and they answered him with a name that he had never heard, that had no meaning at all, but that had a supernatural echo in his dream: Macondo. On the following day he convinced his men that they would never find the sea. He ordered them to cut down the trees to make a clearing beside the river, at the coolest spot on the bank, and there they founded the village. José Arcadio Buendía did not succeed in deciphering the dream of houses with mirror walls until the day he discovered ice. Then he thought he understood its deep meaning. He thought that in the near future they would be able to manufacture blocks of ice on a large scale from such a common material as water and with them build the new houses of the village. Macondo would no longer be a burning place, where the hinges and door knockers twisted with the heat, but would be changed into a wintry city. If he did not persevere in his attempts to build an ice factory, it was because at that time he was absolutely enthusiastic over the education of his sons, especially that of Aureliano, who from the first had revealed a strange intuition for alchemy. The laboratory had been dusted off. Reviewing Melquíades’ notes, serene now, without the exaltation of novelty, in prolonged and patient sessions they tried to separate Úrsula’s gold from the debris that was stuck to the bottom of the pot. Young José Arcadio scarcely took part in the process. While his father was involved body and soul with his water pipe, the willful first-born, who had always been too big for his age, had become a monumental adolescent. His voice had changed. An incipient fuzz appeared on his upper lip. One night, as Úrsula went into the room where he was undressing to go to bed, she felt a mingled sense of shame and pity: he was the first man that she had seen naked after her husband,  and he was so well-equipped for life that he seemed abnormal. Úrsula, pregnant for the third time, relived her newlywed terror. Around that time a merry, foul-mouthed, provocative woman came to the house to help with the chorea, and she knew how to read the future in cards. Úrsula spoke to her about her son. She thought that his disproportionate size was something as unnatural as her cousin’s tail of a pig. The woman let out an expansive laugh that resounded through the house like a spray of broken glass. “Just the opposite,” she said. “He’ll be very lucky.” In order to confirm her prediction she brought her cards to the house a few days later and locked herself up with José Arcadio in a granary off the kitchen. She calmly placed her cards on an old carpenter’s bench. saying anything that came into her head, while the boy waited beside her, more bored than intrigued. Suddenly she reached out her hand and touched him. “Lordy!” she said, sincerely startled, and that was all she could say. José Arcadio felt his bones filling up with foam, a languid fear, and a terrible desire to weep. The woman made no insinuations. But José Arcadio kept looking for her all night long, for the smell of smoke that she had under her armpits and that had got caught under his skin. He wanted to be with her all the time, he wanted her to be his mother, for them never to leave the granary, and for her to say “Lordy!” to him. One day he could not stand it any more and. he went looking for her at her house: He made a formal visit, sitting uncomprehendingly in the living room without saying a word. At that moment he had no desire for her. He found her different, entirely foreign to the image that her smell brought on, as if she were someone else. He drank his coffee and left the house in depression. That night, during the frightful time of lying awake, he desired her again with a brutal anxiety, but he did not want her that time as she had been in the granary but as she had been that afternoon. Days later the woman suddenly called him to her house, where she was alone with her mother, and she had him come into the bedroom with the pretext of showing him a deck of cards. Then she touched him with such freedom that he suffered a delusion after the initial shudder, and he felt more fear than pleasure. She asked him to come and see her that night. He agreed. in order to get away, knowing that he was incapable of going. But that night, in his burning bed, he understood that he had to go we her, even if he were not capable. He got dressed by feel, listening in the dark to his brother’s calm breathing, the dry cough of his father in the next room, the asthma of the hens in the courtyard, the buzz of the mosquitoes, the beating of his heart, and the inordinate bustle of a world that he had not noticed until then, and he went out into the sleeping street. With all his heart he wanted the door to be barred and not just closed as she had promised him. But it was open. He pushed it with the tips of his fingers and the hinges yielded with a mournful and articulate moan that left a frozen echo inside of him. From the moment he entered, sideways and trying not to make a noise, he caught the smell. He was still in the hallway, where the woman’s three brothers had their hammocks in positions that he could not see and that he could not determine in the darkness as he felt his way along the hall to push open the bedroom door and get his bearings there so as not to mistake the bed. He found it. He bumped against the ropes of the hammocks, which were lower than he had suspected, and a man who had been snoring until then turned in his sleep and said in a kind of delusion, “It was Wednesday.” When he pushed open the bedroom door, he could not prevent it from scraping against the uneven floor. Suddenly, in the absolute darkness, he understood with a hopeless nostalgia that he was completely disoriented. Sleeping in the narrow room were the mother, another daughter with her husband and two children, and the woman, who may not have been there. He could have guided himself by the smell if the smell had not been all over the house, so devious and at the same time so definite, as it had always been on his skin. He did not move for a long time, wondering in fright how he had ever got to that abyss of abandonment, when a hand with all its fingers extended and feeling about in the darkness touched his face. He was not surprised, for without knowing, he had been expecting it. Then he gave himself over to that hand, and in a terrible state of exhaustion he let himself be led to a shapeless place where his clothes were taken off and he  was heaved about like a sack of potatoes and thrown from one side to the other in a bottomless darkness in which his arms were useless, where it no longer smelled of woman but of ammonia, and where he tried to remember her face and found before him the face of Úrsula, confusedly aware that he was doing something that for a very long time he had wanted to do but that he had imagined could really never be done, not knowing what he was doing because he did not know where his feet were or where his head was, or whose feet or whose head, and feeling that he could no longer resist the glacial rumbling of his kidneys and the air of his intestines, and fear, and the bewildered anxiety to flee and at the same time stay forever in that exasperated silence and that fearful solitude. Her name was Pilar Ternera. She had been part of the exodus that ended with the founding of Macondo, dragged along by her family in order to separate her from the man who had raped her at fourteen and had continued to love her until she was twenty-two, but who never made up his mind to make the situation public because he was a man apart. He promised to follow her to the ends of the earth, but only later on, when he put his affairs in order, and she had become tired of waiting for him, always identifying him with the tall and short, blond and brunet men that her cards promised from land and sea within three days, three months, or three years. With her waiting she had lost the strength of her thighs, the firmness of her breasts, her habit of tenderness, but she kept the madness of her heart intact. Maddened by that prodigious plaything, José Arcadio followed her path every night through the labyrinth of the room. On a certain occasion he found the door barred, and he knocked several times, knowing that if he had the boldness to knock the first time he would have had to knock until the last, and after an interminable wait she opened the door for him. During the day, lying down to dream, he would secretly enjoy the memories of the night before. But when she came into the house, merry, indifferent, chatty, he did not have to make any effort to hide his tension, because that woman, whose explosive laugh frightened off the doves, had nothing to do with the invisible power that taught him how to breathe from within and control his heartbeats, and that had permitted him to understand why man are afraid of death. He was so wrapped up in himself that he did not even understand the joy of everyone when his father and his brother aroused the household with the news that they had succeeded in penetrating the metallic debris and had separated Úrsula’s gold. They had succeeded, as a matter of fact, after putting in complicated and persevering days at it. Úrsula was happy, and she even gave thanks to God for the invention of alchemy, while the people of the village crushed into the laboratory, and they served them guava jelly on crackers to celebrate the wonder, and José Arcadio Buendía let them see the crucible with the recovered gold, as if he had just invented it. Showing it all around, he ended up in front of his older son, who during the past few days had barely put in an appearance in the laboratory. He put the dry and yellowish mass in front of his eyes and asked him: “What does it look like to you?” José Arcadio answered sincerely: “Dog shit.” His father gave him a blow with the back of his hand that brought out blood and tears. That night Pilar Ternera put arnica compresses on the swelling, feeling about for the bottle and cotton in the dark, and she did everything she wanted with him as long as it did not bother him, making an effort to love him without hurting him. They reached such a state of intimacy that later, without realizing it, they were whispering to each other. “I want to be alone with you,” he said. “One of these days I’m going to tell everybody and we can stop all of this sneaking around.” She did not try to calm him down. “That would be fine,” she said “If we’re alone, we’ll leave the lamp lighted so that we can see each other, and I can holler as much as I want without anybody’s having to butt in, and you can whisper in my ear any crap you can think of.”  That conversation, the biting rancor that he felt against his father, and the imminent possibility of wild love inspired a serene courage in him. In a spontaneous way, without any preparation, he told everything to his brother. At first young Aureliano understood only the risk, the immense possibility of danger that his brother’s adventures implied, and he could not understand the fascination of the subject. Little by little he became contaminated with the anxiety. He wondered about the details of the dangers, he identified himself with the suffering and enjoyment of his brother, he felt frightened and happy. He would stay awake waiting for him until dawn in the solitary bed that seemed to have a bottom of live coals, and they would keep on talking until it was time to get up, so that both of them soon suffered from the same drowsiness, felt the same lack of interest in alchemy and the wisdom of their father, and they took refuge in solitude. “Those kids are out of their heads,” Úrsula said. “They must have worms.” She prepared a repugnant potion for them made out of mashed wormseed, which they both drank with unforeseen stoicism, and they sat down at the same time on their pots eleven times in a single day, expelling some rose-colored parasites that they showed to everybody with great jubilation, for it allowed them to deceive Úrsula as to the origin of their distractions and drowsiness. Aureliano not only understood by then, he also lived his brother’s experiences as something of his own, for on one occasion when the latter was explaining in great detail the mechanism of love, he interrupted him to ask: “What does it feel like?” José Arcadio gave an immediate reply: “It’s like an earthquake.” One January Thursday at two o’clock in the morning, Amaranta was born. Before anyone came into the room, Úrsula examined her carefully. She was light and watery, like a newt, but all of her parts were human: Aureliano did not notice the new thing except when the house became full of people. Protected by the confusion, he went off in search of his brother, who had not been in bed since eleven o’clock, and it was such an impulsive decision that he did not even have time to ask himself how he could get him out of Pilar Ternera’s bedroom. He circled the house for several hours, whistling private calls, until the proximity of dawn forced him to go home. In his mother’s room, playing with the newborn little sister and with a face that drooped with innocence, he found José Arcadio. Úrsula was barely over her forty days’ rest when the gypsies returned. They were the same acrobats and jugglers that had brought the ice. Unlike Melquíades’ tribe, they had shown very quickly that they were not heralds of progress but purveyors of amusement. Even when they brought the ice they did not advertise it for its usefulness in the life of man but as a simple circus curiosity. This time, along with many other artifices, they brought a flying carpet. But they did not offer it as a fundamental contribution to the development of transport, rather as an object of recreation. The people at once dug up their last gold pieces to take advantage of a quick flight over the houses of the village. Protected by the delightful cover of collective disorder, José Arcadio and Pilar passed many relaxing hours. They were two happy lovers among the crowd, and they even came to suspect that love could be a feeling that was more relaxing and deep than the happiness, wild but momentary, of their secret nights. Pilar, however, broke the spell. Stimulated by the enthusiasm that José Arcadio showed in her companionship, she confused the form and the occasion, and all of a sudden she threw the whole world on top of him. “Now you really are a man,” she told him. And since he did not understand what she meant, she spelled it out to him. “You’re going to be a father.” José Arcadio did not dare leave the house for several days. It was enough for him to hear the rocking laughter of Pilar in the kitchen to run and take refuge in the laboratory, where the artifacts of alchemy had come alive again with Úrsula’s blessing. José Arcadio Buendía received his errant son with joy and initiated him in the search for the philosopher’s stone, which he had finally undertaken. One afternoon the boys grew enthusiastic over the flying carpet that went swiftly by the laboratory at window level carrying the gypsy who was driving it and several children from the village who were merrily waving their hands, but José Arcadio Buendía did not even look at it. “Let them dream,” he said. “We’ll do better flying than they are doing, and with more scientific resources than a miserable bedspread.” In spite of his feigned interest, José Arcadio must understood the powers of the philosopher’s egg, which to him looked like a poorly blown bottle. He did not succeed in escaping from his worries. He lost his appetite and he could not sleep. He fell into an ill humor, the same as his father’s over the failure of his undertakings, and such was his upset that José Arcadio Buendía himself relieved him of his duties in the laboratory, thinking that he had taken alchemy too much to heart. Aureliano, of course, understood that his brother’s affliction did not have its source in the search for the philosopher’s stone but he could not get into his confidence. He had lost his former spontaneity. From an accomplice and a communicative person he had become withdrawn and hostile. Anxious for solitude, bitten by a virulent rancor against the world, one night he left his bed as usual, but he did not go to Pilar Ternera’s house, but to mingle is the tumult of the fair. After wandering about among all kinds of contraptions with out becoming interested in any of them, he spotted something that was not a part of it all: a very young gypsy girl, almost a child, who was weighted down by beads and was the most beautiful woman that José Arcadio had ever seen in his life. She was in the crowd that was witnessing the sad spectacle of the man who had been turned into a snake for having disobeyed his parents. José Arcadio paid no attention. While the sad interrogation of the snake-man was taking place, he made his way through the crowd up to the front row, where the gypsy girl was, and he stooped behind her. He pressed against her back. The girl tried to separate herself, but José Arcadio pressed more strongly against her back. Then she felt him. She remained motionless against him, trembling with surprise and fear, unable to believe the evidence, and finally she turned her head and looked at him with a tremulous smile. At that instant two gypsies put the snake-man into his cage and carried him into the tent. The gypsy who was conducting the show announced: “And now, ladies and gentlemen, we are going to show the terrible test of the woman who must have her head chopped off every night at this time for one hundred and fifty years as punishment for having seen what she should not have.” José Arcadio and the gypsy girl did not witness the decapitation. They went to her tent, where they kissed each other with a desperate anxiety while they took off their clothes. The gypsy girl removed the starched lace corsets she had on and there she was, changed into practically nothing. She was a languid little frog, with incipient breasts and legs so thin that they did not even match the size of José Arcadio’s arms, but she had a decision and a warmth that compensated for her fragility. Nevertheless, José Arcadio could not respond to her because they were in a kind of public tent where the gypsies passed through with their circus things and did their business, and would even tarry by the bed for a game of dice. The lamp hanging from the center pole lighted the whole place up. During a pause in the caresses, José Arcadio stretched out naked on the bed without knowing what to do, while the girl tried to inspire him. A gypsy woman with splendid flesh came in a short time after accompanied by a man who was not of the caravan but who was not from the village either, and they both began to undress in front of the bed. Without meaning to, the woman looked at José Arcadio and examined his magnificent animal in repose with a kind of pathetic fervor. “My boy,” she exclaimed, “may God preserve you just as you are.” José Arcadio’s companion asked them to leave them alone, and the couple lay down on the ground, close to the bed. The passion of the others woke up José Arcadio’s fervor. On the first contact the bones of the girl seemed to become disjointed with a disorderly crunch like the sound of a box of dominoes, and her skin broke out into a pale sweat and her eyes filled with tears as her whole body exhaled a lugubrious lament and a vague smell of mud. But she bore the impact with a firmness of character and a bravery that were admirable. José Arcadio felt himself lifted up into the  air toward a state of seraphic inspiration, where his heart burst forth with an outpouring of tender obscenities that entered the girl through her ears and came out of her mouth translated into her language. It was Thursday. On Saturday night, José Arcadio wrapped a red cloth around his head and left with the gypsies. When Úrsula discovered his absence she searched for him all through the village. In the remains of the gypsy camp there was nothing but a garbage pit among the still smoking ashes of the extinguished campfires. Someone who was there looking for beads among the trash told Úrsula that the night before he had seen her son in the tumult of the caravan pushing the snake-man’s cage on a cart. “He’s become a gypsy” she shouted to her husband, who had not shown the slightest sign of alarm over the disappearance. “I hope it’s true,” José Arcadio Buendía said, grinding in his mortar the material that had been ground a thousand times and reheated and ground again. “That way he’ll learn to be a man.” Úrsula asked where the gypsies had gone. She went along asking and following the road she had been shown, thinking that she still had time to catch up to them. She kept getting farther away from the village until she felt so far away that she did not think about returning. José Arcadio Buendía did not discover that his wife was missing until eight o’clock at night, when he left the material warming in a bed of manure and went to see what was wrong with little Amaranta, who was getting hoarse from crying. In a few hours he gathered a group of well-equipped men, put Amaranta in the hands of a woman who offered to nurse her, and was lost on invisible paths in pursuit of Úrsula. Aureliano went with them. Some Indian fishermen, whose language they could not understand, told them with signs that they had not seen anyone pass. After three days of useless searching they returned to the village. For several weeks José Arcadio Buendía let himself be overcome by consternation. He took care of little Amaranta like a mother. He bathed and dressed her, took her to be nursed four times a day, and even sang to her at night the songs that Úrsula never knew how to sing. On a certain occasion Pilar Ternera volunteered to do the household chores until Úrsula came back. Aureliano, whose mysterious intuition had become sharpened with the misfortune, felt a glow of clairvoyance when he saw her come in. Then he knew that in some inexplicable way she was to blame for his brother’s flight and the consequent disappearance of his mother, and he harassed her with a silent and implacable hostility in such a way that the woman did not return to the house. Time put things in their place. José Arcadio Buendía and his son did not know exactly when they returned to the laboratory, dusting things, lighting the water pipe, involved once more in the patient manipulation of the material that had been sleeping for several months in its bed of manure. Even Amaranta, lying in a wicker basket, observed with curiosity the absorbing work of her father and her brother in the small room where the air was rarefied by mercury vapors. On a certain occasion, months after Úrsula’s departure, strange things began to happen. An empty flask that had been forgotten in a cupboard for a long time became so heavy that it could not be moved. A pan of water on the worktable boiled without any fire under it for a half hour until it completely evaporated. José Arcadio Buendía and his son observed those phenomena with startled excitement, unable to explain them but interpreting them as predictions of the material. One day Amaranta’s basket began to move by itself and made a complete turn about the room, to the consternation of Auerliano, who hurried to stop it. But his father did not get upset. He put the basket in its place and tied it to the leg of a table, convinced that the long-awaited event was imminent. It was on that occasion that Auerliano heard him say: “If you don’t fear God, fear him through the metals. Suddenly, almost five months after her disappearance, Úrsula came back. She arrived exalted, rejuvenated, with new clothes in a style that was unknown in the village. José Arcadio Buendía could barely stand up under the impact. “That was it!” he shouted. “I knew it was going to happen.” And  he really believed it, for during his prolonged imprisonment as he manipulated the material, he begged in the depth of his heart that the longed-for miracle should not be the discovery of the philosopher’s stone, or the freeing of the breath that makes metals live, or the faculty to convert the hinges and the locks of the house into gold, but what had just happened: Úrsula’s return. But she did not share his excitement. She gave him a conventional kiss, as if she had been away only an hour, and she told him: “Look out the door.” José Arcadio Buendía took a long time to get out of his perplexity when he went out into the street and saw the crowd. They were not gypsies. They were men and women like them, with straight hair and dark skin, who spoke the same language and complained of the same pains. They had mules loaded down with things to eat, oxcarts with furniture and domestic utensils, pure and simple earthly accessories put on sale without any fuss by peddlers of everyday reality. They came from the other side of the swamp, only two days away, where there were towns that received mail every month in the year and where they were familiar with the implements of good living. Úrsula had not caught up with the gypsies, but she had found the route that her husband had been unable to discover in his frustrated search for the great inventions. PILAR TERNERA’S son was brought to his grand parents’ house two weeks after he was born. Úrsula admitted him grudgingly, conquered once more by the obstinacy of her husband, who could not tolerate the idea that an offshoot of his blood should be adrift, but he imposed the condition that the child should never know his true identity. Although he was given the name José Arcadio, they ended up calling him simply Arcadio so as to avoid confusion. At that time there was so much activity in the town and so much bustle in the house that the care of the children was relegated to a secondary level. They were put in the care of Visitación, a Guajiro Indian woman who had arrived in town with a brother in flight from a plague of insomnia that had been scourging their tribe for several years. They were both so docile and willing to help that Úrsula took them on to help her with her household chores. That was how Arcadio and Amaranta came to speak the Guajiro language before Spanish, and they learned to drink lizard broth and eat spider eggs without Úrsula’s knowing it, for she was too busy with a promising business in candy animals. Macondo had changed. The people who had come with Úrsula spread the news of the good quality of its soil and its privileged position with respect to the swamp, so that from the narrow village of past times it changed into an active town with stores and workshops and a permanent commercial route over which the first Arabs arrived with their baggy pants and rings in their ears, swapping glass beads for macaws. José Arcadio Buendía did not have a moment’s rest. Fascinated by an immediate reality that came to be more fantastic than the vast universe of his imagination, he lost all interest in the alchemist’s laboratory, put to rest the material that had become attenuated with months of manipulation, and went back to being the enterprising man of earlier days when he had decided upon the layout of the streets and the location of the new houses so that no one would enjoy privileges that everyone did not have. He acquired such authority among the new arrivals that foundations were not laid or walls built without his being consulted, and it was decided that he should be the one in charge of the distribution of the land. When the acrobat gypsies returned, with their vagabond carnival transformed now into a gigantic organization of games of luck and chance, they were received with great joy, for it was thought that José Arcadio would be coming back with them. But José Arcadio did not return, nor did they come with the snake-man, who, according to what Úrsula thought, was the only one who could tell them about their son, so the gypsies were not allowed to camp in town or set foot in it in the future, for they were considered the bearers of concupiscence and perversion. José Arcadio Buendía, however, was explicit in maintaining that the old tribe of Melquíades, who had contributed so much to the growth of the village with his age-old wisdom and his fabulous inventions, would always find the gates open. But Melquíades’ tribe, according to what the wanderers said, had been wiped off the face of the earth because they had gone beyond the limits of human knowledge. Emancipated for the moment at least from the torment of fantasy, José Arcadio Buendía in a short time set up a system of order and work which allowed for only one bit of license: the freeing of the birds, which, since the time of the founding, had made time merry with their flutes, and installing in their place musical clocks in every house. They were wondrous clocks made of carved wood, which the Arabs had traded for macaws and which José Arcadio Buendía had synchronized with such precision that every half hour the town grew merry with the progressive chords of the same song until it reached the climax of a noontime that was as exact and unanimous as a complete waltz. It was also José Arcadio Buendía who decided during those years that they should plant almond trees instead of acacias on the streets, and who discovered, without ever revealing it, a way to make them live forever. Many years later, when Macondo was a field of wooden houses with zinc  roofs, the broken and dusty almond trees still stood on the oldest streets, although no one knew who had planted them. While his father was putting the town in order and his mother was increasing their wealth with her marvelous business of candied little roosters and fish, which left the house twice a day strung along sticks of balsa wood, Aureliano spent interminable hours in the abandoned laboratory, learning the art of silverwork by his own experimentation. He had shot up so fast that in a short time the clothing left behind by his brother no longer fit him and he began to wear his father’s, but Visitación had to sew pleats in the shirt and darts in the pants, because Aureliano had not sequined the corpulence of the others. Adolescence had taken away the softness of his voice and had made him silent and definitely solitary, but, on the other hand, it had restored the intense expression that he had had in his eyes when he was born. He concentrated so much on his experiments in silverwork that he scarcely left the laboratory to eat. Worried ever his inner withdrawal, José Arcadio Buendía gave him the keys to the house and a little money, thinking that perhaps he needed a woman. But Aureliano spent the money on muriatic acid to prepare some aqua regia and he beautified the keys by plating them with gold. His excesses were hardly comparable to those of Arcadio and Amaranta, who had already begun to get their second teeth and still went about all day clutching at the Indians’ cloaks, stubborn in their decision not to speak Spanish but the Guajiro language. “You shouldn’t complain.” Úrsula told her husband. “Children inherit their parents’ madness.” And as she was lamenting her misfortune, convinced that the wild behavior of her children was something as fearful as a pig’s tail, Aureliano gave her a look that wrapped her in an atmosphere of uncertainty. “Somebody is coming,” he told her. Úrsula, as she did whenever he made a prediction, tried to break it down with her housewifely logic. It was normal for someone to be coming. Dozens of strangers came through Macondo every day without arousing suspicion or secret ideas. Nevertheless, beyond all logic, Aureliano was sure of his prediction. “I don’t know who it will be,” he insisted, “but whoever it is is already on the way.” That Sunday, in fact, Rebeca arrived. She was only eleven years old. She had made the difficult trip from Manaure with some hide dealers who had taken on the task of delivering her along with a letter to José Arcadio Buendía, but they could not explain precisely who the person was who had asked the favor. Her entire baggage consisted of a small trunk, a little rocking chair with small handpainted flowers, and a canvas sack which kept making a cloc-cloc-cloc sound, where she carried her parents’ bones. The letter addressed to José Arcadio Buendía was written is very warm terms by someone who still loved him very much in spite of time and distance, and who felt obliged by a basic humanitarian feeling to do the charitable thing and send him that poor unsheltered orphan, who was a second cousin of Úrsula’s and consequently also a relative of José Arcadio Buendía, although farther removed, because she was the daughter of that unforgettable friend Nicanor Ulloa and his very worthy wife Rebeca Montiel, may God keep them in His holy kingdom, whose remains the girl was carrying so that they might be given Christian burial. The names mentioned, as well as the signature on the letter, were perfectly legible, but neither José Arcadio, Buendía nor Úrsula remembered having any relatives with those names, nor did they know anyone by the name of the sender of the letter, much less the remote village of Manaure. It was impossible to obtain any further information from the girl. From the moment she arrived she had been sitting in the rocker, sucking her finger and observing everyone with her large, startled eyes without giving any sign of understanding what they were asking her. She wore a diagonally striped dress that had been dyed black, worn by use, and a pair of scaly patent leather boots. Her hair was held behind her ears with bows of black ribbon. She wore a scapular with the images worn away by sweat, and on her right wrist the fang of a carnivorous animal mounted on a backing of copper as an amulet against the evil eye. Her greenish skin, her stomach, round and tense as a drum. revealed poor health and hunger  that were older than she was, but when they gave her something to eat she kept the plate on her knees without tasting anything. They even began to think that she was a deaf-mute until the Indians asked her in their language if she wanted some water and she moved her eyes as if she recognized them and said yes with her head. They kept her, because there was nothing else they could do. They decided to call her Rebeca, which according to the letter was her mother’s name, because Aureliano had the patience to read to her the names of all the saints and he did not get a reaction from any one of them. Since there was no cemetery in Macondo at that time, for no one had died up till then, they kept the bag of bones to wait for a worthy place of burial, and for a long time it got in the way everywhere and would be found where least expected, always with its clucking of a broody hen. A long time passed before Rebeca became incorporated into the life of the family. She would sit in her small rocker sucking her finger in the most remote corner of the house. Nothing attracted her attention except the music of the clocks, which she would look for every half hour with her frightened eyes as if she hoped to find it someplace in the air. They could not get her to eat for several days. No one understood why she had not died of hunger until the Indians, who were aware of everything, for they went ceaselessly about the house on their stealthy feet, discovered that Rebeca only liked to eat the damp earth of the courtyard and the cake of whitewash that she picked of the walls with her nails. It was obvious that her parents, or whoever had raised her, had scolded her for that habit because she did it secretively and with a feeling of guilt, trying to put away supplies so that she could eat when no one was looking. From then on they put her under an implacable watch. They threw cow gall onto the courtyard and, rubbed hot chili on the walls, thinking they could defeat her pernicious vice with those methods, but she showed such signs of astuteness and ingenuity to find some earth that Úrsula found herself forced to use more drastic methods. She put some orange juice and rhubarb into a pan that she left in the dew all night and she gave her the dose the following day on an empty stomach. Although no one had told her that it was the specific remedy for the vice of eating earth, she thought that any bitter substance in an empty stomach would have to make the liver react. Rebeca was so rebellious and strong in spite of her frailness that they had to tie her up like a calf to make her swallow the medicine, and they could barely keep back her kicks or bear up under the strange hieroglyphics that she alternated with her bites and spitting, and that, according to what the scandalized Indians said, were the vilest obscenities that one could ever imagine in their language. When Úrsula discovered that, she added whipping to the treatment. It was never established whether it was the rhubarb or the beatings that had effect, or both of them together, but the truth was that in a few weeks Rebeca began to show signs of recovery. She took part in the games of Arcadio and Amaranta, who treated her like an older sister, and she ate heartily, using the utensils properly. It was soon revealed that she spoke Spanish with as much fluency as the Indian language, that she had a remarkable ability for manual work, and that she could sing the waltz of the clocks with some very funny words that she herself had invented. It did not take long for them to consider her another member of the family. She was more affectionate to Úrsula than any of her own children had been, and she called Arcadio, and Amaranta brother and sister, Aureliano uncle, and José Arcadio Buendía grandpa. So that she finally deserved, as much as the others, the name of Rebeca Buendía, the only one that she ever had and that she bore with dignity until her death. One night about the time that Rebeca was cured of the vice of eating earth and was brought to sleep in the other children’s room, the Indian woman, who slept with them awoke by chance and heard a strange, intermittent sound in the corner. She got up in alarm, thinking that an animal had come into the room, and then she saw Rebeca in the rocker, sucking her finger and with her eyes lighted up in the darkness like those of a cat. Terrified, exhausted by her fate, Visitación recognized in those eyes the symptoms of the sickness whose threat had obliged her and her brother to exile themselves forever from an age-old kingdom where they had been prince and princess. It was the insomnia plague. Cataure, the Indian, was gone from the house by morning. His sister stayed because her fatalistic heart told her that the lethal sickness would follow her, no matter what, to the farthest corner of the earth. No one understood Visitación’s alarm. “If we don’t ever sleep again, so much the better,” José Arcadio Buendía said in good humor. “That way we can get more out of life.” But the Indian woman explained that the most fearsome part of the sickness of insomnia was not the impossibility of sleeping, for the body did not feel any fatigue at all, but its inexorable evolution toward a more critical manifestation: a loss of memory. She meant that when the sick person became used to his state of vigil, the recollection of his childhood began to be erased from his memory, then the name and notion of things, and finally the identity of people and even the awareness of his own being, until he sank into a kind of idiocy that had no past. José Arcadio Buendía, dying with laughter, thought that it was just a question of one of the many illnesses invented by the Indians’ superstitions. But Úrsula, just to be safe, took the precaution of isolating Rebeca from the other children. After several weeks, when Visitación’s terror seemed to have died down, José Arcadio Buendía found himself rolling over in bed, unable to fall asleep. Úrsula, who had also awakened, asked him what was wrong, and he answered: “I’m thinking about Prudencio Aguilar again.” They did not sleep a minute, but the following day they felt so rested that they forgot about the bad night. Aureliano commented with surprise at lunchtime that he felt very well in spite of the fact that he had spent the whole night in the laboratory gilding a brooch that he planned to give to Úrsula for her birthday. They did not become alarmed until the third day, when no one felt sleepy at bedtime and they realized that they had gone more than fifty hours without sleeping. “The children are awake too,” the Indian said with her fatalistic conviction. “Once it gets into a house no one can escape the plague.” They had indeed contracted the illness of insomnia. Úrsula, who had learned from her mother the medicinal value of plants, prepared and made them all drink a brew of monkshood, but they could not get to sleep and spent the whole day dreaming on their feet. In that state of hallucinated lucidity, not only did they see the images of their own dreams, but some saw the images dreamed by others. It was as if the house were full of visitors. Sitting in her rocker in a corner of the kitchen, Rebeca dreamed that a man who looked very much like her, dressed in white linen and with his shirt collar closed by a gold button, was bringing her a bouquet of roses. He was accompanied by a woman with delicate hands who took out one rose and put it in the child’s hair. Úrsula understood that the man and woman were Rebeca’s parents, but even though she made a great effort to recognize them, she confirmed her certainty that she had never seen them. In the meantime, through an oversight that José Arcadio Buendía never forgave himself for, the candy animals made in the house were still being sold in the town. Children and adults sucked with delight on the delicious little green roosters of insomnia, the exquisite pink fish of insomnia, and the tender yellow ponies of insomnia, so that dawn on Monday found the whole town awake. No one was alarmed at first. On the contrary, they were happy at not sleeping because there was so much to do in Macondo in those days that there was barely enough time. They worked so hard that soon they had nothing else to do and they could be found at three o’clock in the morning with their arms crossed, counting the notes in the waltz of the clock. Those who wanted to sleep, not from fatigue but because of the nostalgia for dreams, tried all kinds of methods of exhausting themselves. They would gather together to converse endlessly, to tell over and over for hours on end the same jokes, to complicate to the limits of exasperation the story about the capon, which was an endless game in which the narrator asked if they wanted him to tell them the story about the capon, and when they answered yes, the narrator would say that he had not asked them to say yes, but whether they wanted him to tell them the story about the capon, and when they answered no, the narrator told them that he had not asked them to say no, but whether they wanted him to tell them the story about the capon, and when they remained silent the narrator told them that he had not asked them to remain silent but whether they wanted him to tell them the story about the capon, and no one could leave because the narrator would say that he had not asked them to leave but whether they wanted him to tell them the story about the capon, and so on and on in a vicious circle that lasted entire nights. When José Arcadio Buendía realized that the plague had invaded the town, he gathered together the heads of families to explain to them what he knew about the sickness of insomnia, and they agreed on methods to prevent the scourge from spreading to other towns in the swamp. That was why they took the bells off the goats, bells that the Arabs had swapped them for macaws, and put them at the entrance to town at the disposal of those who would not listen to the advice and entreaties of the sentinels and insisted on visiting the town. All strangers who passed through the streets of Macondo at that time had to ring their bells so that the sick people would know that they were healthy. They were not allowed to eat or drink anything during their stay, for there was no doubt but that the illness was transmitted by mouth, and all food and drink had been contaminated by insomnia. In that way they kept the plague restricted to the perimeter of the town. So effective was the quarantine that the day came when the emergency situation was accepted as a natural thing and life was organized in such a way that work picked up its rhythm again and no one worried any more about the useless habit of sleeping. It was Aureliano who conceived the formula that was to protect them against loss of memory for several months. He discovered it by chance. An expert insomniac, having been one of the first, he had learned the art of silverwork to perfection. One day he was looking for the small anvil that he used for laminating metals and he could not remember its name. His father told him: “Stake.” Aureliano wrote the name on a piece of paper that he pasted to the base of the small anvil: stake. In that way he was sure of not forgetting it in the future. It did not occur to him that this was the first manifestation of a loss of memory, because the object had a difficult name to remember. But a few days later be, discovered that he had trouble remembering almost every object in the laboratory. Then he marked them with their respective names so that all he had to do was read the inscription in order to identify them. When his father told him about his alarm at having forgotten even the most impressive happenings of his childhood, Aureliano explained his method to him, and José Arcadio Buendía put it into practice all through the house and later on imposed it on the whole village. With an inked brush he marked everything with its name: table, chair, clock, door, wall, bed, pan. He went to the corral and marked the animals and plants: cow, goat, pig, hen, cassava, caladium, banana. Little by little, studying the infinite possibilities of a loss of memory, he realized that the day might come when things would be recognized by their inscriptions but that no one would remember their use. Then he was more explicit. The sign that he hung on the neck of the cow was an exemplary proof of the way in which the inhabitants of Macondo were prepared to fight against loss of memory: This is the cow. She must be milked every morning so that she will produce milk, and the milk must be boiled in order to be mixed with coffee to make coffee and milk. Thus they went on living in a reality that was slipping away, momentarily captured by words, but which would escape irremediably when they forgot the values of the written letters. At the beginning of the road into the swamp they put up a sign that said MACONDO and another larger one on the main street that said GOD EXISTS. In all the houses keys to memorizing objects and feelings had been written. But the system demanded so much vigilance and moral strength that many succumbed to the spell of an imaginary reality, one invented by themselves, which was less practical for them but more comforting. Pilar Ternera was the one who contributed most to popularize that mystification when she conceived the trick of reading the past in cards as she had read the future before. By means of that recourse the insomniacs began to live in a world built on the uncertain alternatives of the cards, where a father was remembered faintly as the dark man who had arrived at the beginning of April and a mother was remembered only as the dark woman who wore a gold ring on her left hand, and where a birth date was reduced to the last Tuesday on which a lark sang in the laurel tree. Defeated by those practices of consolation, José Arcadio Buendía then decided to build the memory machine that he had desired once in order to remember the marvelous inventions of the gypsies. The artifact was based on the possibility of reviewing every morning, from beginning to end, the totality of knowledge acquired during one’s life. He conceived of it as a spinning dictionary that a person placed on the axis could operate by means of a lever, so that in a very few hours there would pass before his eyes the notions most necessary for life. He had succeeded in writing almost fourteen thousand entries when along the road from the swamp a strange-looking old man with the sad sleepers’ bell appeared, carrying a bulging suitcase tied with a rope and pulling a cart covered with black cloth. He went straight to the house of José Arcadio Buendía. Visitación did not recognize him when she opened the door and she thought he had come with the idea of selling something, unaware that nothing could be sold in a town that was sinking irrevocably into the quicksand of forgetfulness. He was a decrepit man. Although his voice was also broken by uncertainty and his hands seemed to doubt the existence of things, it was evident that he came from the world where men could still sleep and remember. José Arcadio Buendía found him sitting in the living room fanning himself with a patched black hat as he read with compassionate attention the signs pasted to the walls. He greeted him with a broad show of affection, afraid that he had known him at another time and that he did not remember him now. But the visitor was aware of his falseness, He felt himself forgotten, not with the irremediable forgetfulness of the heart, but with a different kind of forgetfulness, which was more cruel and irrevocable and which he knew very well because it was the forgetfulness of death. Then he understood. He opened the suitcase crammed with indecipherable objects and from among then he took out a little case with many flasks. He gave José Arcadio Buendía a drink of a gentle color and the light went on in his memory. His eyes became moist from weeping even before he noticed himself in an absurd living room where objects were labeled and before he was ashamed of the solemn nonsense written on the walls, and even before he recognized the newcomer with a dazzling glow of joy. It was Melquíades. While Macondo was celebrating the recovery of its memory, José Arcadio Buendía and Melquíades dusted off their old friendship. The gypsy was inclined to stay in the town. He really had been through death, but he had returned because he could not bear the solitude. Repudiated by his tribe, having lost all of his supernatural faculties because of his faithfulness to life, he decided to take refuge in that corner of the world which had still not been discovered by death, dedicated to the operation of a daguerreotype laboratory. José Arcadio Buendía had never heard of that invention. But when he saw himself and his whole family fastened onto a sheet of iridescent metal for an eternity, he was mute with stupefaction. That was the date of the oxidized daguerreotype in which José Arcadio Buendía appeared with his bristly and graying hair, his card board collar attached to his shirt by a copper button, and an expression of startled solemnity, whom Úrsula described, dying with laughter, as a “frightened general.” José Arcadio Buendía was, in fact, frightened on that dear December morning when the daguerreotype was made, for he was thinking that people were slowly wearing away while his image would endure an a metallic plaque. Through a curious reversal of custom, it was Úrsula who got that idea out of his head, as it was also she who forgot her ancient bitterness and decided that Melquíades would stay on in the house, although she never permitted them to make a daguerreotype of her because (according to her very words) she did not want to survive as a laughingstock for her grandchildren. That morning she dressed the children in their best clothes, powdered their faces, and gave a spoonful of marrow syrup to each one so that they would all remain absolutely motionless during the nearly two minutes in front of Melquíades fantastic camera. In the family daguerreotype, the only one that ever existed, Aureliano appeared dressed in black velvet between Amaranta and Rebeca. He had the same languor and the same clairvoyant look that he would have years later as he faced the firing squad. But he still had not sensed the premonition of his fate. He was an expert silversmith, praised all over the swampland for the delicacy of his work. In the workshop, which he shared with Melquíades’ mad laboratory, he could barely be heard breathing. He seemed to be taking refuge in some other time, while his father and the gypsy with shouts interpreted the predictions of Nostradamus amidst a noise of flasks and trays and the disaster of spilled acids and silver bromide that was lost in the twists and turns it gave at every instant. That dedication to his work, the good judgment with which he directed his attention, had allowed Aureliano to earn in a short time more money than Úrsula had with her delicious candy fauna, but everybody thought it strange that he was now a full-grown man and had not known a woman. It was true that he had never had one. Several months later saw the return of Francisco the Man, as ancient vagabond who was almost two hundred years old and who frequently passed through Macondo distributing songs that he composed himself. In them Francisco the Man told in great detail the things that had happened in the towns along his route, from Manaure to the edge of the swamp, so that if anyone had a message to send or an event to make public, he would pay him two cents to include it in his repertory. That was how Úrsula learned of the death of her mother, as a simple consequence of listening to the songs in the hope that they would say something about her son José Arcadio. Francisco the Man, called that because he had once defeated the devil in a duel of improvisation, and whose real name no one knew, disappeared from Macondo during the insomnia plague and one night he appeared suddenly in Catarino’s store. The whole town went to listen to him to find out what had happened in the world. On that occasion there arrived with him a woman who was so fat that four Indians had to carry her in a rocking chair, and an adolescent mulatto girl with a forlorn look who protected her from the sun with an umbrella. Aureliano went to Catarino’s store that night. He found Francisco the Man, like a monolithic chameleon, sitting in the midst of a circle of bystanders. He was singing the news with his old, out-of-tune voice, accompanying himself with the same archaic accordion that Sir Walter Raleigh had given him in the Guianas and keeping time with his great walking feet that were cracked from saltpeter. In front of a door at the rear through which men were going and coming, the matron of the rocking chair was sitting and fanning herself in silence. Catarino, with a felt rose behind his ear, was selling the gathering mugs of fermented cane juice, and he took advantage of the occasion to go over to the men and put his hand on them where he should not have. Toward midnight the heat was unbearable. Aureliano listened to the news to the end without hearing anything that was of interest to his family. He was getting ready to go home when the matron signaled him with her hand. “You go in too.” she told him. “It only costs twenty cents.” Aureliano threw a coin into the hopper that the matron had in her lap and went into the room without knowing why. The adolescent mulatto girl, with her small bitch’s teats, was naked on the bed. Before Aureliano sixty-three men had passed through the room that night. From being used so much, kneaded with sweat and sighs, the air in the room had begun to turn to mud. The girl took off the soaked sheet and asked Aureliano to hold it by one side. It was as heavy as a piece of canvas. They squeezed it, twisting it at the ends until it regained its natural weight. They turned over the mat and the sweat came out of the other side. Aureliano was anxious for that operation never to end. He knew the theoretical mechanics of love, but he could not stay on his feet because of the weakness of his knees, and although he had goose pimples on his burning skin he could not resist the urgent need to expel the weight of his bowels. When the girl finished fixing up the bed and told him to get undressed, he gave her a confused explanation: “They made me come in. They told me to throw twenty cents into the hopper and hurry up.” The girl understood his confusion. “If you throw in twenty cents more when you go out, you can stay a little longer,” she said softly. Aureliano got undressed, tormented by shame, unable to get rid of the idea that-his nakedness could not stand comparison with that of his brother. In spite of the girl’s efforts he felt more and more indifferent and terribly alone. “I’ll throw in other twenty cents,” he said with a desolate voice. The girl thanked him in silence. Her back was raw. Her skin was stuck to her ribs and her breathing was forced because of an immeasurable exhaustion. Two years before, far away from there, she had fallen asleep without putting out the candle and had awakened surrounded by flames. The house where she lived with the grandmother who had raised her was reduced to ashes. Since then her grandmother carried her from town to town, putting her to bed for twenty cents in order to make up the value of the burned house. According to the girl’s calculations, she still had ten years of seventy men per night, because she also had to pay the expenses of the trip and food for both of them as well as the pay of the Indians who carried the rocking chair. When the matron knocked on the door the second time, Aureliano left the room without having done anything, troubled by a desire to weep. That night he could not sleep, thinking about the girl, with a mixture of desire and pity. He felt an irresistible need to love her and protect her. At dawn, worn out by insomnia and fever, he made the calm decision to marry her in order to free her from the despotism of her grandmother and to enjoy all the nights of satisfaction that she would give the seventy men. But at ten o’clock in the morning, when he reached Catarino’s store, the girl had left town. Time mitigated his mad proposal, but it aggravated his feelings of frustration. He took refuge in work. He resigned himself to being a womanless man for all his life in order to hide the shame of his uselessness. In the meantime, Melquíades had printed on his plates everything that was printable in Macondo, and he left the daguerreotype laboratory to the fantasies of José Arcadio Buendía who had resolved to use it to obtain scientific proof of the existence of God. Through a complicated process of superimposed exposures taken in different parts of the house, he was sure that sooner or later he would get a daguerreotype of God, if He existed, or put an end once and for all to the supposition of His existence. Melquíades got deeper into his interpretations of Nostradamus. He would stay up until very late, suffocating in his faded velvet vest, scribbling with his tiny sparrow hands, whose rings had lost the glow of former times. One night he thought he had found a prediction of the future of Macondo. It was to be a luminous city with great glass houses where there was no trace remaining of the race of the Buendía. “It’s a mistake,” José Arcadio Buendía thundered. “They won’t be houses of glass but of ice, as I dreamed, and there will always be a Buendía, per omnia secula seculorum.” Úrsula fought to preserve common sense in that extravagant house, having broadened her business of little candy animals with an oven that went all night turning out baskets and more baskets of bread and a prodigious variety of puddings, meringues, and cookies, which disappeared in a few hours on the roads winding through the swamp. She had reached an age where she had a right to rest, but she was nonetheless more and more active. So busy was she in her prosperous enterprises that one afternoon she looked distractedly toward the courtyard while the Indian woman helped her sweeten the dough and she saw two unknown and beautiful adolescent girls doing frame embroidery in the light of the sunset. They were Rebeca and Amaranta. As soon as they had taken off the mourning clothes for their grandmother, which they wore with inflexible rigor for three years, their bright clothes seemed to have given them a new place in the world. Rebeca, contrary to what might have been expected, was the more beautiful. She had a light complexion, large and peaceful eyes, and magical hands that seemed to work out the design of the embroidery with invisible threads. Amaranta, the younger, was somewhat graceless, but she had the natural distinction, the inner tightness of her dead grandmother. Next to them, although he was already revealing the physical drive of his father, Arcadio looked like a child. He set about learning the art of silverwork with Aureliano, who had also taught him how to read and write. Úrsula suddenly realized that the house had become full of people, that her children were on the point of marrying and having children, and that they would be obliged to scatter for lack of space. Then she took out the money she had accumulated over long years of hard labor, made some arrangements with her customers, and undertook the enlargement of the house. She had a formal parlor for visits built, another one that was more comfortable and cool for daily use, a dining room with a table with twelve places where the family could sit with all of their guests, nine bedrooms with windows on the courtyard and a long porch protected from the heat of noon by a rose garden with a railing on which to place pots of ferns and begonias. She had the kitchen enlarged to hold two ovens. The granary where Pilar Ternera had read José Arcadio’s future was torn down and another twice as large built so that there would never be a lack of food in the house. She had baths built is the courtyard in the shade of the chestnut tree, one for the women and another for the men, and in the rear a large stable, a fenced-in chicken yard, a shed for the milk cows, and an aviary open to the four winds so that wandering birds could roost there at their pleasure. Followed by dozens of masons and carpenters, as if she had contracted her husband’s hallucinating fever, Úrsula fixed the position of light and heat and distributed space without the least sense of its limitations. The primitive building of the founders became filled with tools and materials, of workmen exhausted by sweat, who asked everybody please not to molest them, exasperated by the sack of bones that followed them everywhere with its dull rattle. In that discomfort, breathing quicklime and tar, no one could see very well how from the bowels of the earth there was rising not only the largest house is the town, but the most hospitable and cool house that had ever existed in the region of the swamp. José Buendía, trying to surprise Divine Providence in the midst of the cataclysm, was the one who least understood it. The new house was almost finished when Úrsula drew him out of his chimerical world in order to inform him that she had an order to paint the front blue and not white as they had wanted. She showed him the official document. José Arcadio Buendía, without understanding what his wife was talking about, deciphered the signature. “Who is this fellow?” he asked: “The magistrate,” Úrsula answered disconsolately. They say he’s an authority sent by the government.” Don Apolinar Moscote, the magistrate, had arrived in Macondo very quietly. He put up at the Hotel Jacob—built by one of the first Arabs who came to swap knickknacks for macaws—and on the following day he rented a small room with a door on the street two blocks away from the Buendía house. He set up a table and a chair that he had bought from Jacob, nailed up on the wall the shield of the republic that he had brought with him, and on the door he painted the sign: Magistrate. His first order was for all the houses to be painted blue in celebration of the anniversary of national independence. José Arcadio Buendía, with the copy of the order in his hand, found him taking his nap in a hammock he had set up in the narrow office. “Did you write this paper?” he asked him. Don Apolinar Moscote, a mature man, timid, with a ruddy complexion, said yes. “By what right?” José Arcadio Buendía asked again. Don Apolinar Moscote picked up a paper from the drawer of the table and showed it to him. “I have been named magistrate of this town.” José Arcadio Buendía did not even look at the appointment. “In this town we do not give orders with pieces of paper,” he said without losing his calm. “And so that you know it once and for all, we don’t need any judge here because there’s nothing that needs judging.” Facing Don Apolinar Moscote, still without raising his voice, he gave a detailed account of how they had founded the village, of how they had distributed the land, opened the roads, and introduced the improvements that necessity required without having bothered the government and without anyone having bothered them. “We are so peaceful that none of us has died even of a natural death,” he said. “You can see that we still don’t have any cemetery.” No once was upset that the government had not helped them. On the contrary, they were happy that up until then it had let them grow in peace, and he hoped that it would continue leaving them that way, because they had not founded a town so that the first upstart who came along would tell them what to do. Don Apolinar had put on his denim jacket, white like his trousers, without losing at any moment the elegance of his gestures. “So that if you want to stay here like any other ordinary citizen, you’re quite welcome,” José Arcadio Buendía concluded. “But if you’ve come to cause disorder by making the people paint their houses blue, you can pick up your junk and go back where you came from. Because my house is going to be white, white, like a dove.” Don Apolinar Moscote turned pale. He took a step backward and tightened his jaws as he said with a certain affliction: “I must warn you that I’m armed.” José Arcadio Buendía did not know exactly when his hands regained the useful strength with which he used to pull down horses. He grabbed Don Apolinar Moscote by the lapels and lifted him up to the level of his eyes. “I’m doing this,” he said, “because I would rather carry you around alive and not have to keep carrying you around dead for the rest of my life.” In that way he carried him through the middle of the street, suspended by the lapels, until he put him down on his two feet on the swamp road. A week later he was back with six barefoot and ragged soldiers, armed with shotguns, and an oxcart in which his wife and seven daughters were traveling. Two other carts arrived later with the furniture, the baggage, and the household utensils. He settled his family in the Hotel Jacob, while he looked for a house, and he went back to open his office under the protection of the soldiers. The founders of Macondo, resolving to expel the invaders, went with their older sons to put themselves at the disposal of José Arcadio Buendía. But he was against it, as he explained, because it was not manly to make trouble for someone in front of his family, and Don Apolinar had returned with his wife and daughters. So he decided to resolve the situation in a pleasant way. Aureliano went with him. About that time he had begun to cultivate the black mustache with waxed tips and the somewhat stentorian voice that would characterize him in the war. Unarmed, without paying any attention to the guards, they went into the magistrate’s office. Don Apolinar Moscote did not lose his calm. He introduced them to two of his daughters who happened to be there: Amparo, sixteen, dark like her mother, and Remedios, only nine, a pretty little girl with lilycolored skin and green eyes. They were gracious and well-mannered. As soon as the men came in, before being introduced, they gave them chairs to sit on. But they both remained standing. “Very well, my friend,” José Arcadio Buendía said, “you may stay here, not because you have those bandits with shotguns at the door, but out of consideration for your wife and daughters.” Don Apolinar Moscote was upset, but José Arcadio Buendía did not give him time to reply. “We only make two conditions,” he went on. “The first: that everyone can paint his house the color he feels like. The second: that the soldiers leave at once. We will guarantee order for you.” The magistrate raised his right hand with all the fingers extended. “Your word of honor?” “The word of your enemy,” José Arcadio Buendía said. And he added in a bitter tone: “Because I must tell you one thing: you and I are still enemies.” The soldiers left that same afternoon. A few days later José Arcadio Buendía found a house for the magistrate’s family. Everybody was at peace except Aureliano. The image of Remedios, the magistrate’s younger daughter, who, because of her age, could have been his daughter, kept paining him in some part of his body. It was a physical sensation that almost bothered him when he walked, like a pebble in his shoe.  THE NEW HOUSE, white, like a dove, was inaugurated with a dance. Úrsula had got that idea from the afternoon when she saw Rebeca and Amaranta changed into adolescents, and it could almost have been said that the main reason behind the construction was a desire to have a proper place for the girls to receive visitors. In order that nothing would be lacking in splendor she worked like a galley slave as the repairs were under way, so that before they were finished she had ordered costly necessities for the decorations, the table service, and the marvelous invention that was to arouse the astonishment of the town and the jubilation of the young people: the pianola. They delivered it broken down, packed in several boxes that were unloaded along with the Viennese furniture, the Bohemian crystal, the table service from the Indies Company, the tablecloths from Holland, and a rich variety of lamps and candlesticks, hangings and drapes. The import house sent along at its own expense an Italian expert, Pietro Crespi, to assemble and tune the pianola, to instruct the purchasers in its functioning, and to teach them how to dance the latest music printed on its six paper rolls. Pietro Crespi was young and blond, the most handsome and well mannered man who had ever been seen in Macondo, so scrupulous in his dress that in spite of the suffocating heat he would work in his brocade vest and heavy coat of dark cloth. Soaked in sweat, keeping a reverent distance from the owners of the house, he spent several weeks shut up is the parlor with a dedication much like that of Aureliano in his silverwork. One morning, without opening the door, without calling anyone to witness the miracle, he placed the first roll in the pianola and the tormenting hammering and the constant noise of wooden lathings ceased in a silence that was startled at the order and neatness of the music. They all ran to the parlor. José Arcadio Buendía was as if struck by lightning, not because of the beauty of the melody, but because of the automatic working of the keys of the pianola, and he set up Melquíades’ camera with the hope of getting a daguerreotype of the invisible player. That day the Italian had lunch with them. Rebeca and Amaranta, serving the table, were intimidated by the way in which the angelic man with pale and ringless hands manipulated the utensils. In the living room, next to the parlor, Pietro Crespi taught them how to dance. He showed them the steps without touching them, keeping time with a metronome, under the friendly eye of Úrsula, who did not leave the room for a moment while her daughters had their lesson. Pietro Crespi wore special pants on those days, very elastic and tight, and dancing slippers, “You don’t have to worry so much,” José Arcadio Buendía told her. “The man’s a fairy.” But she did not leave off her vigilance until the apprenticeship was over and the Italian left Macondo. Then they began to organize the party. Úrsula drew up a strict guest list, in which the only ones invited were the descendants of the founders, except for the family of Pilar Ternera, who by then had had two more children by unknown fathers. It was truly a high-class list, except that it was determined by feelings of friendship, for those favored were not only the oldest friends of José Arcadio Buendía’s house since before they undertook the exodus and the founding of Macondo, but also their sons and grandsons, who were the constant companions of Aureliano and Arcadio since infancy, and their daughters, who were the only ones who visited the house to embroider with Rebeca and Amaranta. Don Apolinar Moscote, the benevolent ruler whose activity had been reduced to the maintenance from his scanty resources of two policemen armed with wooden clubs, was a figurehead. In older to support the household expenses his daughters had opened a sewing shop, where they made felt flowers as well as guava delicacies, and wrote love notes to order. But in spite of being modest and hard-working, the most beautiful girls in Iowa, and the most skilled at the new dances, they did not manage to be considered for the party.  While Úrsula and the girls unpacked furniture, polished silverware, and hung pictures of maidens in boats full of roses, which gave a breath of new life to the naked areas that the masons had built, José Arcadio Buendía stopped his pursuit of the image of God, convinced of His nonexistence, and he took the pianola apart in order to decipher its magical secret. Two days before the party, swamped in a shower of leftover keys and hammers, bungling in the midst of a mix-up of strings that would unroll in one direction and roll up again in the other, he succeeded in a fashion in putting the instrument back together. There had never been as many surprises and as much dashing about as in those days, but the new pitch lamps were lighted on the designated day and hour. The house was opened, still smelling of resin and damp whitewash, and the children and grandchildren of the founders saw the porch with ferns and begonias, the quiet rooms, the garden saturated with the fragrance of the roses, and they gathered together in the parlor, facing the unknown invention that had been covered with a white sheet. Those who were familiar with the piano, popular in other towns in the swamp, felt a little disheartened, but more bitter was Úrsula’s disappointment when she put in the first roll so that Amaranta and Rebeca could begin the dancing and the mechanism did not work. Melquíades, almost blind by then, crumbling with decrepitude, used the arts of his timeless wisdom in an attempt to fix it. Finally José Arcadio Buendía managed, by mistake, to move a device that was stuck and the music came out, first in a burst and then in a flow of mixed-up notes. Beating against the strings that had been put in without order or concert and had been tuned with temerity, the hammers let go. But the stubborn descendants of the twenty-one intrepid people who plowed through the mountains in search of the sea to the west avoided the reefs of the melodic mix-up and the dancing went on until dawn. Pietro Crespi came back to repair the pianola. Rebeca and Amaranta helped him put the strings in order and helped him with their laughter at the mix-up of the melodies. It was extremely pleasant and so chaste in its way that Úrsula ceased her vigilance. On the eve of his departure a farewell dance for him was improvised with the pianola and with Rebeca he put on a skillful demonstration of modern dance, Arcadio and Amaranta matched them in grace and skill. But the exhibition was interrupted because Pilar Ternera, who was at the door with the onlookers, had a fight, biting and hair pulling, with a woman who had dared to comment that Arcadio had a woman’s behind. Toward midnight Pietro Crespi took his leave with a sentimental little speech, and he promised to return very soon. Rebeca accompanied him to the door, and having closed up the house and put out the lamps, she went to her room to weep. It was an inconsolable weeping that lasted for several days, the cause of which was not known even by Amaranta. Her hermetism was not odd. Although she seemed expansive and cordial, she had a solitary character and an impenetrable heart. She was a splendid adolescent with long and firm bones, but she still insisted on using the small wooden rocking chair with which she had arrived at the house, reinforced many times and with the arms gone. No one had discovered that even at that age she still had the habit of sucking her finger. That was why she would not lose an opportunity to lock herself in the bathroom and had acquired the habit of sleeping with her face to the wall. On rainy afternoons, embroidering with a group of friends on the begonia porch, she would lose the thread of the conversation and a tear of nostalgia would salt her palate when she saw the strips of damp earth and the piles of mud that the earthworms had pushed up in the garden. Those secret tastes, defeated in the past by oranges and rhubarb, broke out into an irrepressible urge when she began to weep. She went back to eating earth. The first time she did it almost out of curiosity, sure that the bad taste would be the best cure for the temptation. And, in fact, she could not bear the earth in her mouth. But she persevered, overcome by the growing anxiety, and little by little she was getting back her ancestral appetite, the taste of primary minerals, the unbridled satisfaction of what was the original food. She would put handfuls of earth in her pockets, and ate them in small bits without being seen, with a confused feeling of pleasure and rage, as she instructed her girl friends in the most difficult needlepoint and spoke about other men, who did not deserve the sacrifice of having one eat the whitewash on the walls because of them. The handfuls of earth made the only man who deserved that show of degradation less remote and more certain, as if the ground that he walked on with his fine patent leather boots in another part of the world were transmitting to her the weight and the temperature of his blood in a mineral savor that left a harsh aftertaste in her mouth and a sediment of peace in her heart. One afternoon, for no reason, Amparo Moscote asked permission to see the house. Amaranta and Rebeca, disconcerted by the unexpected visit, attended her with a stiff formality. They showed her the remodeled mansion, they had her listen to the rolls on the pianola, and they offered her orange marmalade and crackers. Amparo gave a lesson in dignity, personal charm, and good manners that impressed Úrsula in the few moments that she was present during the visit. After two hours, when the conversation was beginning to wane, Amparo took advantage of Amaranta’s distraction and gave Rebeca a letter. She was able to see the name of the Estimable Señorita Rebeca Buendía, written in the same methodical hand, with the same green ink, and the same delicacy of words with which the instructions for the operation of the pianola were written, and she folded the letter with the tips of her fingers and hid it in her bosom, looking at Amparo Moscote with an expression of endless and unconditional gratitude and a silent promise of complicity unto death. The sudden friendship between Amparo Moscote and Rebeca Buendía awakened the hopes of Aureliano. The memory of little Remedios had not stopped tormenting him, but he had not found a chance to see her. When he would stroll through town with his closest friends, Magnífico Visbal and Gerineldo Márquez—the sons of the founders of the same names—he would look for her in the sewing shop with an anxious glance, but he saw only the older sisters. The presence of Amparo Moscote in the house was like a premonition. “She has to come with her,” Aureliano would say to himself in a low voice. “She has to come.” He repeated it so many times and with such conviction that one afternoon when he was putting together a little gold fish in the work shop, he had the certainty that she had answered his call. Indeed, a short time later he heard the childish voice, and when he looked up his heart froze with terror as he saw the girl at the door, dressed in pink organdy and wearing white boots. “You can’t go in there, Remedios, Amparo Moscote said from the hall. They’re working.” But Aureliano did not give her time to respond. He picked up the little fish by the chain that came through its mouth and said to her. “Come in.” Remedios went over and asked some questions about the fish that Aureliano could not answer because he was seized with a sudden attack of asthma. He wanted to stay beside that lily skin forever, beside those emerald eyes, close to that voice that called him “sir” with every question. showing the same respect that she gave her father. Melquíades was in the corner seated at the desk scribbling indecipherable signs. Aureliano hated him. All he could do was tell Remedios that he was going to give her the little fish and the girl was so startled by the offer that she left the workshop as fast as she could. That afternoon Aureliano lost the hidden patience with which he had waited for a chance to see her. He neglected his work. In several desperate efforts of concentration he willed her to appear but Remedios did not respond. He looked for her in her sisters’ shop, behind the window shades in her house, in her father’s office, but he found her only in the image that saturated his private and terrible solitude. He would spend whole hours with Rebeca in the parlor listening to the music on the pianola. She was listening to it because it was the music with which Pietro Crespi had taught them how to dance. Aureliano listened to it simply because everything, even music, reminded him of Remedios. The house became full of loves Aureliano expressed it in poetry that had no beginning or end. He would write it on the harsh pieces of parchment that Melquíades gave him, on the bathroom walls, on the skin of his arms, and in all of it Remedios would appear transfigured: Remedios in the  soporific air of two in the afternoon, Remedios in the soft breath of the roses, Remedios in the water-clock secrets of the moths, Remedios in the steaming morning bread, Remedios everywhere and Remedios forever. Rebeca waited for her love at four in the afternoon, embroidering by the window. She knew that the mailman’s mule arrived only every two weeks, but she always waited for him, convinced that he was going to arrive on some other day by mistake. It happened quite the opposite: once the mule did not come on the usual day. Mad with desperation, Rebeca got up in the middle of the night and ate handfuls of earth in the garden with a suicidal drive, weeping with pain and fury, chewing tender earthworms and chipping her teeth on snail shells. She vomited until dawn. She fell into a state of feverish prostration, lost consciousness, and her heart went into a shameless delirium. Úrsula, scandalized, forced the lock on her trunk and found at the bottom, tied together with pink ribbons, the sixteen perfumed letters and the skeletons of leaves and petals preserved in old books and the dried butterflies that turned to powder at the touch. Aureliano was the only one capable of understanding such desolation. That afternoon, while Úrsula was trying to rescue Rebeca from the slough of delirium, he went with Magnífico Visbal and Gerineldo Márquez to Catarino’s store. The establishment had been expanded with a gallery of wooden rooms where single women who smelled of dead flowers lived. A group made up of an accordion and drums played the songs of Francisco the Man, who had not been seen in Macondo for several years. The three friends drank fermented cane juice. Magnífico and Gerineldo, contemporaries of Aureliano but more skilled in the ways of the world, drank methodically with the women seated on their laps. One of the women, withered and with goldwork on her teeth, gave Aureliano a caress that made him shudder. He rejected her. He had discovered that the more he drank the more he thought about Remedios, but he could bear the torture of his recollections better. He did not know exactly when he began to float. He saw his friends and the women sailing in a radiant glow, without weight or mass, saying words that did not come out of their mouths and making mysterious signals that did not correspond to their expressions. Catarino put a hand on his shoulder and said to him: “It’s going on eleven.” Aureliano turned his head, saw the enormous disfigured face with a felt flower behind the ear, and then he lost his memory, as during the times of forgetfulness, and he recovered it on a strange dawn and in a room that was completely foreign, where Pilar Ternera stood in her slip, barefoot, her hair down, holding a lamp over him, startled with disbelief. “Aureliano!” Aureliano checked his feet and raised his head. He did not know how he had come there, but he knew what his aim was, because he had carried it hidden since infancy in an inviolable backwater of his heart. “I’ve come to sleep with you,” he said. His clothes were smeared with mud and vomit. Pilar Ternera, who lived alone at that time with her two younger children, did not ask him any questions. She took him to the bed. She cleaned his face with a damp cloth, took of his clothes, and then got completely undressed and lowered the mosquito netting so that her children would not see them if they woke up. She had become tired of waiting for the man who would stay, of the men who left, of the countless men who missed the road to her house, confused by the uncertainty of the cards. During the wait her skin had become wrinkled, her breasts had withered, the coals of her heart had gone out. She felt for Aureliano in the darkness, put her hand on his stomach and kissed him on the neck with a maternal tenderness. “My poor child,” she murmured. Aureliano shuddered. With a calm skill, without the slightest misstep, he left his accumulated grief behind and found Remedios changed into a swamp without horizons, smelling of a raw animal and recently ironed clothes. When he came to the surface he was weeping. First they were involuntary and broken sobs. Then he emptied himself out in an unleashed flow, feeling that something swollen and painful had burst inside of him. She waited, snatching his head  with the tips of her fingers, until his body got rid of the dark material that would not let him live. They Pilar Ternera asked him: “Who is it?” And Aureliano told her. She let out a laugh that in other times frightened the doves and that now did not even wake up the children. “You’ll have to raise her first,” she mocked, but underneath the mockery Aureliano found a reservoir of understanding. When he went out of the room, leaving behind not only his doubts about his virility but also the bitter weight that his heart had borne for so many months, Pilar Ternera made him a spontaneous promise. “I’m going to talk to the girl,” she told him, “and you’ll see what I’ll serve her on the tray.” She kept her promise. But it was a bad moment, because the house had lost its peace of former days. When she discovered Rebeca’s passion, which was impossible to keep secret because of her shouts, Amaranta suffered an attack of fever. She also suffered from the barb of a lonely love. Shut up in the bathroom, she would release herself from the torment of a hopeless passion by writing feverish letters, which she finally hid in the bottom of her trunk. Úrsula barely had the strength to take care of the two sick girls. She was unable, after prolonged and insidious interrogations, to ascertain the causes of Amaranta’s prostration. Finally, in another moment of inspiration, she forced the lock on the trunk and found the letters tied with a pink ribbon, swollen with fresh lilies and still wet with tears, addressed and never sent to Pietro Crespi. Weeping with rage, she cursed the day that it had occurred to her to buy the pianola, and she forbade the embroidery lessons and decreed a kind of mourning with no one dead which was to be prolonged until the daughters got over their hopes. Useless was the intervention of José Arcadio Buendía, who had modified his first impression of Pietro Crespi and admired his ability in the manipulation of musical machines. So that when Pilar Ternera told Aureliano that Remedios had decided on marriage, he could see that the news would only give his parents more trouble. Invited to the parlor for a formal interview, José Arcadio Buendía and Úrsula listened stonily to their son’s declaration. When he learned the name of the fiancée, however, José Arcadio Buendía grew red with indignation. “Love is a disease,” he thundered. “With so many pretty and decent girls around, the only thing that occurs to you is to get married to the daughter of our enemy.” But Úrsula agreed with the choice. She confessed her affection for the seven Moscote sisters. for their beauty, their ability for work, their modesty, and their good manners, and she celebrated her son’s prudence. Conquered by his wife’s enthusiasm, José Arcadio Buendía then laid down one condition: Rebeca, who was the one he wanted, would marry Pietro Crespi. Úrsula would take Amaranta on a trip to the capital of the province when she had time, so that contact with different people would alleviate her disappointment. Rebeca got her health back just as soon as she heard of the agreement, and she wrote her fiancé a jubilant letter that she submitted to her parents’ approval and put into the mail without the use of any intermediaries. Amaranta pretended to accept the decision and little by little she recovered from her fevers, but she promised herself that Rebeca would marry only over her dead body. The following Saturday José Arcadio Buendía put on his dark suit, his celluloid collar, and the deerskin boots that he had worn for the first time the night of the party, and went to ask for the hand of Remedios Moscote. The magistrate and his wife received him, pleased and worried at the same time, for they did not know the reason for the unexpected visit, and then they thought that he was confused about the name of the intended bride. In order to remove the mistake, the mother woke Remedios up and carried her into the living room, still drowsy from sleep. They asked her if it was true that she had decided to get married, and she answered, whimpering, that she only wanted them to let her sleep. José Arcadio Buendía, understanding the distress of the Moscotes, went to clear things up with Aureliano. When he returned, the Moscotes had put on formal clothing, had rearranged the furniture and put fresh flowers in the vases, and were waiting in the company of their older daughters. Overwhelmed by the unpleasantness of the occasion and the bothersome hard collar, José Arcadio Buendía confirmed the fact that Remedios, indeed, was the chosen one. “It  doesn’t make sense,” Don Apolinar Moscote said with consternation. “We have six other daughters, all unmarried, and at an age where they deserve it, who would be delighted to be the honorable wife of a gentleman as serious and hard-working as your son, and Aurelito lays his eyes precisely on the one who still wets her bed.” His wife, a well-preserved woman with afflicted eyelids and expression, scolded his mistake. When they finished the fruit punch, they willingly accepted Aureliano’s decision. Except that Señora Moscote begged the favor of speaking to Úrsula alone. Intrigued, protesting that they were involving her in men’s affairs, but really feeling deep emotion, Úrsula went to visit her the next day. A half hour later she returned with the news that Remedios had not reached puberty. Aureliano did not consider that a serious barrier. He had waited so long that he could wait as long as was necessary until his bride reached the age of conception. The newfound harmony was interrupted by the death of Melquíades. Although it was a foreseeable event, the circumstances were not. A few months after his return, a process of aging had taken place in him that was so rapid and critical that soon he was treated as one of those useless great-grandfathers who wander about the bedrooms like shades, dragging their feet, remembering better times aloud, and whom no one bothers about or remembers really until the morning they find them dead in their bed. At first José Arcadio Buendía helped him in his work, enthusiastic over the novelty of the daguerreotypes and the predictions of Nostradamus. But little by little he began abandoning him to his solitude, for communication was becoming Increasingly difficult. He was losing his sight and his hearing, he seemed to confuse the people he was speaking to with others he had known in remote epochs of mankind, and he would answer questions with a complex hodgepodge of languages. He would walk along groping in the air, although he passed between objects with an inexplicable fluidity, as if be were endowed with some instinct of direction based on an immediate prescience. One day he forgot to put in his false teeth, which at night he left in a glass of water beside his bed, and he never put them in again. When Úrsula undertook the enlargement of the house, she had them build him a special room next to Aureliano’s workshop, far from the noise and bustle of the house, with a window flooded with light and a bookcase where she herself put in order the books that were almost destroyed by dust and moths, the flaky stacks of paper covered with indecipherable signs, and the glass with his false teeth, where some aquatic plants with tiny yellow flowers had taken root. The new place seemed to please Melquíades, because he was never seen any more, not even in the dining room, He only went to Aureliano’s workshop, where he would spend hours on end scribbling his enigmatic literature on the parchments that he had brought with him and that seemed to have been made out of some dry material that crumpled like puff paste. There he ate the meals that Visitación brought him twice a day, although in the last days he lost his appetite and fed only on vegetables. He soon acquired the forlorn look that one sees in vegetarians. His skin became covered with a thin moss, similar to that which flourished on the antique vest that he never took off, and his breath exhaled the odor of a sleeping animal. Aureliano ended up forgetting about him, absorbed in the composition of his poems, but on one occasion he thought he understood something of what Melquíades was saying in his groping monologues, and he paid attention. In reality, the only thing that could be isolated in the rocky paragraphs was the insistent hammering on the word equinox, equinox, equinox, and the name of Alexander von Humboldt. Arcadio got a little closer to him when he began to help Aureliano in his silverwork. Melquíades answered that effort at communication at times by giving forth with phrases in Spanish that had very little to do with reality. One afternoon, however, he seemed to be illuminated by a sudden emotion. Years later, facing the firing squad, Arcadio would remember the trembling with which Melquíades made him listen to several pages of his impenetrable writing, which of course he did not understand, but which when read aloud were like encyclicals being chanted. Then he smiled for the first time in a long while and said in Spanish: “When I die, burn mercury in my room for three days.” Arcadio told that to José Arcadio Buendía and the latter tried to get more explicit information, but he received  only one answer: “I have found immortality.” When Melquíades’ breathing began to smell, Arcadio took him to bathe in the river on Thursday mornings. He seemed to get better. He would undress and get into the water with the boys, and his mysterious sense of orientation would allow him to avoid the deep and dangerous spots. “We come from the water,” he said on a certain occasion. Much time passed in that way without anyone’s seeing him in the house except on the night when he made a pathetic effort to fix the pianola, and when he would go to the river with Arcadio, carrying under his arm a gourd and a bar of palm oil soap wrapped in a towel. One Thursday before they called him to go to the river, Aureliano heard him say: “I have died of fever on the dunes of Singapore.” That day he went into the water at a bad spot and they did not find him until the following day, a few miles downstream, washed up on a bright bend in the river and with a solitary vulture sitting on his stomach. Over the scandalized protests of Úrsula, who wept with more grief than she had had for her own father, José Arcadio Buendía was opposed to their burying him. “He is immortal,” he said, “and he himself revealed the formula of his resurrection.” He brought out the forgotten water pipe and put a kettle of mercury to boil next to the body, which little by little was filling with blue bubbles. Don Apolinar Moscote ventured to remind him that an unburied drowned man was a danger to public health. “None of that, because he’s alive,” was the answer of José Arcadio Buendía, who finished the seventy-two hours with the mercurial incense as the body was already beginning to burst with a livid fluorescence, the soft whistles of which impregnated the house with a pestilential vapor. Only then did he permit them to bury him, not in any ordinary way, but with the honors reserved for Macondo’s greatest benefactor. It was the first burial and the bestattended one that was ever seen in the town, only surpassed, a century later, by Big Mama’s funeral carnival. They buried him in a grave dug in the center of the plot destined for the cemetery, with a stone on which they wrote the only thing they knew about him: MELQUÍADES. They gave him his nine nights of wake. In the tumult that gathered in the courtyard to drink coffee, tell jokes, and play cards. Amaranta found a chance to confess her love to Pietro Crespi, who a few weeks before had formalized his promise to Rebeca and had set up a store for musical instruments and mechanical toys in the same section where the Arabs had lingered in other times swapping knickknacks for macaws, and which the people called the Street of the Turks. The Italian, whose head covered with patent leather curls aroused in women an irrepressible need to sigh, dealt with Amaranta as with a capricious little girl who was not worth taking seriously. “I have a younger brother,” he told her. “He’s coming to help me in the store.” Amaranta felt humiliated and told Pietro Crespi with a virulent anger that she was prepared to stop her sister’s wedding even if her own dead body had to lie across the door. The Italian was so impressed by the dramatics of the threat that he could not resist the temptation to mention it to Rebeca. That was how Amaranta’s trip, always put off by Úrsula’s work, was arranged in less than a week. Amaranta put up no resistance, but when she kissed Rebeca good-bye she whispered in her ear: “Don’t get your hopes up. Even if they send me to the ends of the earth I’ll find some way of stopping you from getting married, even if I have to kill you.” With the absence of Úrsula, with the invisible presence of Melquíades, who continued his stealthy shuffling through the rooms, the house seemed enormous and empty. Rebeca took charge of domestic order, while the Indian woman took care of the bakery. At dusk, when Pietro Crespi would arrive, preceded by a cool breath of lavender and always bringing a toy as a gift, his fiancée would receive the visitor in the main parlor with doors and windows open to be safe from any suspicion. It was an unnecessary precaution, for the Italian had shown himself to be so respectful that he did not even touch the hand of the woman who was going to be his wife within the year. Those visits were filling the house with remarkable toys. Mechanical ballerinas, music boxes, acrobatic monkeys, trotting horses, clowns who played the tambourine: the rich and startling mechanical fauna that  Pietro Crespi brought dissipated José Arcadio Buendía’s affliction over the death of Melquíades and carried him back to his old days as an alchemist. He lived at that time in a paradise of disemboweled animals, of mechanisms that had been taken apart in an attempt to perfect them with a system of perpetual motion based upon the principles of the pendulum. Aureliano, for his part, had neglected the workshop in order to teach little Remedios to read and write. At first the child preferred her dolls to the man who would come every afternoon and who was responsible for her being separated from her toys in order to be bathed and dressed and seated in the parlor to receive the visitor. But Aureliano’s patience and devotion finally won her over, up to the point where she would spend many hours with him studying the meaning of the letters and sketching in a notebook with colored pencils little houses with cows in the corral and round suns with yellow rays that hid behind the hills. Only Rebeca was unhappy, because of Amaranta’s threat. She knew her sister’s character, the haughtiness of her spirit, and she was frightened by the virulence of her anger. She would spend whole hours sucking her finger in the bathroom, holding herself back with an exhausting iron will so as not to eat earth. In search of some relief for her uncertainty, she called Pilar Ternera to read her future. After a string of conventional vagaries, Pilar Ternera predicted: “You will not be happy as long as your parents remain unburied.” Rebeca shuddered. As in the memory of a dream she saw herself entering the house as a very small girl, with the trunk and the little rocker, and a bag whose contents she had never known. She remembered a bald gentleman dressed in linen and with his collar closed by a gold button, who had nothing to do with the king of hearts. She remembered a very young and beautiful woman with warm and perfumed hands, who had nothing in common with the jack of diamonds and his rheumatic hands, and who used to put flowers in her hair and take her out walking in the afternoon through a town with green streets. “I don’t understand,” she said. Pilar Ternera seemed disconcerted: “I don’t either, but that’s what the cards say.” Rebeca was so preoccupied with the enigma that she told it to José Arcadio Buendía, and he scolded her for believing in the predictions of the cards, but he undertook the silent task of searching closets and trunks, moving furniture and turning over beds and floorboards looking for the bag of bones. He remembered that he had not seen it since the time of the rebuilding. He secretly summoned the masons and one of them revealed that he had walled up the bag in some bedroom because it bothered him in his work. After several days of listening, with their ears against the walls, they perceived the deep cloc-cloc. They penetrated the wall and there were the bones in the intact bag. They buried it the same day in a grave without a stone next to that of Melquíades, and José Arcadio Buendía returned home free of a burden that for a moment had weighed on his conscience as much as the memory of Prudencio Aguilar. When he went through the kitchen he kissed Rebeca on the forehead. “Get those bad thoughts out of your head,” he told her. “You’re going to be happy.” The friendship with Rebeca opened up to Pilar Ternera the doors of the house, closed by Úrsula since the birth of Arcadio. She would arrive at any hour of the day, like a flock of goats, and would unleash her feverish energy in the hardest tasks. Sometimes she would go into the workshop and help Arcadio sensitize the daguerreotype plates with an efficiency and a tenderness that ended up by confusing him. That woman bothered him. The tan of her skin, her smell of smoke, the disorder of her laughter in the darkroom distracted his attention and made him bump into things. On a certain occasion Aureliano was there working on his silver, and Pilar Ternera leaned over the table to admire his laborious patience. Suddenly it happened. Aureliano made sure that Arcadio was in the darkroom before raising his eyes and meeting those of Pilar Ternera, whose thought was perfectly visible, as if exposed to the light of noon.  “Well,” Aureliano said. “Tell me what it is.” Pilar Ternera bit her lips with a sad smile. “That you’d be good in a war,” she said. “Where you put your eye, you put your bullet.” Aureliano relaxed with the proof of the omen. He went back to concentrate on his work as if nothing had happened, and his voice took on a restful strength. “I will recognize him,” he said. “He’ll bear my name.” José Arcadio Buendía finally got what he was looking for: he connected the mechanism of the clock to a mechanical ballerina, and the toy danced uninterruptedly to the rhythm of her own music for three days. That discovery excited him much more than any of his other harebrained undertakings. He stopped eating. He stopped sleeping. Only the vigilance and care of Rebeca kept him from being dragged off by his imagination into a state of perpetual delirium from which he would not recover. He would spend the nights walking around the room thinking aloud, searching for a way to apply the principles of the pendulum to oxcarts, to harrows, to everything that was useful when put into motion. The fever of insomnia fatigued him so much that one dawn he could not recognize the old man with white hair and uncertain gestures who came into his bedroom. It was Prudencio Aguilar. When he finally identified him, startled that the dead also aged, José Arcadio Buendía felt himself shaken by nostalgia. “Prudencio,” he exclaimed. “You’ve come from a long way off!” After many years of death the yearning for the living was so intense, the need for company so pressing, so terrifying the neatness of that other death which exists within death, that Prudencio Aguilar had ended up loving his worst enemy. He had spent a great deal of time looking for him. He asked the dead from Riohacha about him, the dead who came from the Upar Valley, those who came from the swamp, and no one could tell him because Macondo was a town that was unknown to the dead until Melquíades arrived and marked it with a small black dot on the motley maps of death. José Arcadio Buendía conversed with Prudencio Aguilar until dawn. A few hours later, worn out by the vigil, he went into Aureliano’s workshop and asked him: “What day is today?” Aureliano told him that it was Tuesday. “I was thinking the same thing,” José Arcadio Buendía said, “but suddenly I realized that it’s still Monday, like yesterday. Look at the sky, look at the walls, look at the begonias. Today is Monday too.” Used to his manias, Aureliano paid no attention to him. On the next day, Wednesday, José Arcadio Buendía went back to the workshop. “This is a disaster,” he said. “Look at the air, listen to the buzzing of the sun, the same as yesterday and the day before. Today is Monday too.” That night Pietro Crespi found him on the porch, weeping for Prudencio Aguilar, for Melquíades, for Rebeca’s parents, for his mother and father, for all of those he could remember and who were now alone in death. He gave him a mechanical bear that walked on its hind legs on a tightrope, but he could not distract him from his obsession. He asked him what had happened to the project he had explained to him a few days before about the possibility of building a pendulum machine that would help men to fly and he answered that it was impossible because a pendulum could lift anything into the air but it could not lift itself. On Thursday he appeared in the workshop again with the painful look of plowed ground. “The time machine has broken,” he almost sobbed, “and Úrsula and Amaranta so far away!” Aureliano scolded him like a child and he adopted a contrite air. He spent six hours examining things, trying to find a difference from their appearance on the previous day in the hope of discovering in them some change that would reveal the passage of time. He spent the whole night in bed with his eyes open, calling to Prudencio Aguilar, to Melquíades, to all the dead, so that they would share his distress. But no one came. On Friday. before anyone arose, he watched the appearance of nature again until he did not have the slightest doubt but that it was Monday. Then he grabbed the bar from a door and with the savage violence of his uncommon strength he smashed to dust the equipment in the alchemy laboratory, the daguerreotype room, the silver workshop, shouting like a man possessed in some high-sounding and fluent but completely incomprehensible language. He was about to finish off the rest of the house  when Aureliano asked the neighbors for help. Ten men were needed to get him down, fourteen to tie him up, twenty to drag him to the chestnut tree in the courtyard, where they left him tied up, barking in the strange language and giving off a green froth at the mouth. When Úrsula and Amaranta returned he was still tied to the trunk of the chestnut tree by his hands and feet, soaked with rain and in a state of total innocence. They spoke to him and he looked at them without recognizing them, saying things they did not understand. Úrsula untied his wrists and ankles, lacerated by the pressure of the rope, and left him tied only by the waist. Later on they built him a shelter of palm brandies to protect him from the sun and the rain. 
"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # print("old input_ids.shape:"+ str(input_ids.shape))
    
    # 限制输入长度为 input_length
    input_ids = input_ids[:, :input_len]
    # print("latest input_ids.shape:"+ str(input_ids.shape))
    
    # 将截断后的 prompt 解码回来
    true_str = tokenizer.batch_decode(input_ids)[0]
    prompt = true_str
  
    for i in range(num_prompts):
        # prompt = tokenizer.decode(prefix_token_ids +
        #                           [(offsets[i] + i + j) % tokenizer.vocab_size
        #                            for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate 
    with OPTIONAL burstiness.
    
    Args:
        input_requests: 
            A list of input requests, each represented as a tuple.
        request_rate: 
            The rate at which requests are generated (requests/s).
        burstiness (optional): 
            The burstiness factor of the request generation. 
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results 
            in more bursty requests, while a higher burstiness value 
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len -
                                                                 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if gootput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in gootput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(gootput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in gootput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(gootput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in gootput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(gootput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    gootput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0])
    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         prompt=test_prompt,
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         logprobs=logprobs,
                                         best_of=best_of,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = request
        request_func_input = RequestFuncInput(model=model_id,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              best_of=best_of,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        gootput_config_dict=gootput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if gootput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        metrics.request_goodput if gootput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    gootput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        gootput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in gootput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return gootput_config_dict


def parse_goodput(slo_pairs):
    gootput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            gootput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return gootput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    # model_id = args.model
    model_id = args.model.split('/')[-1]
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len, None)
                              for prompt, prompt_formatted, prompt_len,
                              output_len, _ in input_requests]

    elif args.dataset_name == "hf":
        input_requests = sample_hf_requests(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            random_seed=args.seed,
            fixed_output_len=args.hf_output_len,
        )

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    gootput_config_dict = check_goodput_args(args)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            gootput_config_dict=gootput_config_dict,
            max_concurrency=args.max_concurrency,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer.')

    args = parser.parse_args()
    main(args)
