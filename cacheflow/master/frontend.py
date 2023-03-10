from typing import List, Optional, Set, Tuple

from transformers import AutoTokenizer

from cacheflow.sampling_params import SamplingParams
from cacheflow.sequence import Sequence
from cacheflow.sequence import SequenceGroup
from cacheflow.utils import Counter


class Frontend:

    def __init__(
        self,
        model_name: str,
        block_size: int,
    ) -> None:
        self.block_size = block_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_group_counter = Counter()
        self.seq_counter = Counter()
        self.inputs: List[Tuple[SequenceGroup, SamplingParams]] = []

    def query(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_beam_search: bool = False,
        stop_token_ids: Set[int] = set(),
        max_num_steps: int = 16,  # From OpenAI API.
        num_logprobs: int = 0,
        context_window_size: Optional[int] = None,
    ) -> None:
        # Stop when we see an EOS token.
        stop_token_ids.add(self.tokenizer.eos_token_id)
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop_token_ids=stop_token_ids,
            max_num_steps=max_num_steps,
            num_logprobs=num_logprobs,
            context_window_size=context_window_size,
        )
        token_ids = self.tokenizer.encode(prompt)
        self._add_query(token_ids, sampling_params)

    def _add_query(
        self,
        token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> None:
        seqs: List[Sequence] = []
        for _ in range(sampling_params.n):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, token_ids, block_size=self.block_size)
            seqs.append(seq)

        group_id = next(self.seq_group_counter)
        seq_group = SequenceGroup(group_id, seqs)
        self.inputs.append((seq_group, sampling_params))

    def get_inputs(self) -> List[Tuple[SequenceGroup, SamplingParams]]:
        inputs = self.inputs
        self.inputs = []
        return inputs

    def print_response(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.seqs:
            token_ids = seq.get_token_ids()
            output = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f'Seq {seq.seq_id}: {output!r}')
