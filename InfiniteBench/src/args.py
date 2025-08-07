from argparse import ArgumentParser, Namespace
from InfiniteBench.src.eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="../results", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        choices=["gpt4", "yarn-mistral", "kimi", "claude2", "rwkv", "yi-6b-200k", "yi-34b-200k", "chatglm3"],
        default="gpt4",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use.")
    p.add_argument("--template_name", type=str, help="Override automatic template detection with a specific template name")
    return p.parse_args()