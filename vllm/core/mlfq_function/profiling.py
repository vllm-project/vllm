""" This profiler assumes that there are max_batch_size running requests when calling llm.engine.step()
Run the following commands to get the profiling database for facebook/opt-1.3b:

python fastserve/example/profile.py --model facebook/opt-1.3b --tokenizer facebook/opt-1.3b --beam_width 1 --file_path <file_path>
"""

from collections import namedtuple
import csv
import dataclasses
import pickle
from typing import List, Dict

ParallelConfig = namedtuple("ParallelConfig", ("pp", "tp"))
PromptConfig = namedtuple("PromptConfig", ("batch_size", "input_length", "beam_width"))

tp_config = [1, 2, 4, 8]
pp_config = [1, 2, 4, 8]
bs_config = [1, 2, 4, 8, 16]
in_len_config = [32, 64, 128, 256, 512]
bw_config = [1]


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""

    model_name: str
    # The latency of each iteration for a specific model under a
    # specific parallelism configuration
    # para_dict    type: Dict[parallel_config -> latency_dict]
    # latency_dict type: Dict[prompt_config -> latency_list]
    # latency_list type: List[float]
    para_dict: Dict

    def add_result(
        self,
        parallel_config: ParallelConfig,
        prompt_config: PromptConfig,
        latency_list: List[float],
    ):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = {}
        self.para_dict[parallel_config][prompt_config] = latency_list

    def get_latency_list(
        self,
        pp: int,
        tp: int,
        batch_size: int,
        beam_width: int,
        in_len: int,
    ):
        assert pp in pp_config and tp in tp_config
        para_config = ParallelConfig(pp, tp)

        assert para_config in self.para_dict
        result = self.para_dict[para_config]

        assert batch_size in bs_config and beam_width in bw_config
        upper_value = min(
            in_len_config,
            key=lambda x: x - in_len if x >= in_len else in_len_config[-1],
        )
        lower_value = min(
            in_len_config,
            key=lambda x: in_len - x if x <= in_len else in_len_config[-1],
        )

        upper_config = PromptConfig(batch_size, upper_value, beam_width)
        lower_config = PromptConfig(batch_size, lower_value, beam_width)

        upper_values = result[upper_config]
        lower_values = result[lower_config]

        if upper_value == lower_value:
            return lower_values
        elif in_len <= in_len_config[0]:
            return result[PromptConfig(batch_size, in_len_config[0], beam_width)]
        elif in_len >= in_len_config[-1]:
            return result[PromptConfig(batch_size, in_len_config[-1], beam_width)]
        else:
            # linear interpolation
            scaling = (in_len - lower_value) / (upper_value - lower_value)
            ret_val = [
                lower_values[i] + (upper_values[i] - lower_values[i]) * scaling
                for i in range(len(upper_values))
            ]
            return ret_val


class ProfilingDatabase:
    """Store the profiling results of all the models"""

    def __init__(self, database_filename: str, new_database: bool = False):
        # The file that backs up the profiling results.
        self.database_filename = database_filename
        # Dict[model_name -> ProfilingResult]
        self.results = {}
        if not new_database:
            with open(database_filename, "rb") as f:
                self.results = pickle.load(f)

    def get(self, model_name: str) -> ProfilingResult:
        return self.results.get(model_name)

    def update(self, result: ProfilingResult) -> None:
        self.results[result.model_name] = result

    def _retrive_data(
        self,
        row: Dict,
    ):
        """Retrive the profiling results from a row of the profiling CSV file."""
        parallel_config = ParallelConfig(int(row["pp"]), int(row["tp"]))
        prompt_config = PromptConfig(
            int(row["batch_size"]), int(row["input_length"]), int(row["beam_width"])
        )
        if row["iter_latencies(s)"] == "N/A" or "iter_latencies(s)" not in row:
            iter_latencies = None
        else:
            iter_latencies = list(
                map(float, row["iter_latencies(s)"].strip("[]").split(","))
            )
        return row["model_name"], parallel_config, prompt_config, iter_latencies

    def update_from_csv(self, file_name: str):
        # read lines
        with open(file_name, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                (
                    model_name,
                    parallel_config,
                    prompt_config,
                    iter_latencies,
                ) = self._extract_data(row)
                if iter_latencies is not None:
                    if model_name not in self.results:
                        self.results[model_name] = ProfilingResult(
                            model_name,
                            {parallel_config: {prompt_config: iter_latencies}},
                        )
                    else:
                        self.results[model_name].add_result(
                            parallel_config, prompt_config, iter_latencies
                        )

    def materialize(self):
        """Write the profiling results to the database file."""
        with open(self.database_filename, "wb") as f:
            pickle.dump(self.results, f)
