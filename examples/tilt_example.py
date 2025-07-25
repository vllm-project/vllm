import asyncio
import itertools
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import jsonlines
import PIL
from tqdm import tqdm

from vllm import AsyncLLMEngine, LLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.multimodal.tilt_processor import (Document, Page, Question,
                                            TiltPreprocessor)
from vllm.outputs import RequestOutput
from vllm.utils import FlexibleArgumentParser

os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


class Loader:
    OCR_TOOL_PRIORITY = {
        ("microsoft_cv", "basic"): 3,
        ("microsoft_cv", "natural"): 2,
        ("microsoft_cv", None): 1,
        ("ground_truth", None): 0,
    }

    def __init__(
        self,
        path: str | Path,
        splits: set[str] | None = None,
        limit_documents: int | None = None,
    ):
        self.path = Path(path)
        self.splits = splits
        self.limit_documents = limit_documents
        self._dummy_image = PIL.Image.new(mode="L",
                                          size=(768, 1086),
                                          color=255)

    def load_dataset(self) -> Iterator[tuple[Document, list[Question]]]:
        dataset = self._load_dataset()
        if self.limit_documents is not None:
            dataset = itertools.islice(dataset, self.limit_documents)
        return dataset

    def _load_dataset(self) -> Iterator[tuple[Document, list[Question]]]:
        with jsonlines.open(self.path / "document.jsonl", "r") as r:
            annotations = list(r)

        with jsonlines.open(self.path / "documents_content.jsonl", "r") as r:
            documents = list(r)

        document_contents = {}
        for doc in documents:
            doc_id = doc["name"]

            contents = None
            current_priority = -1
            for c in doc["contents"]:
                tool_name = c.get("tool_name", "")
                reading_order = c.get("tool_options",
                                      {}).get("reading_order", None)
                priority = self.OCR_TOOL_PRIORITY.get(
                    (tool_name, reading_order), -1)
                if priority > current_priority:
                    contents = c
                    current_priority = priority
            if contents is None:
                print(f"Skipping document {doc_id} due to missing contents")
                continue
            doc["contents"] = [contents]

            document_contents[doc_id] = doc

        for anno in annotations:
            if self.splits is not None and anno.get(
                    "split") not in self.splits:
                continue

            doc_id = anno["name"]
            if doc_id not in document_contents:
                continue

            doc = document_contents[doc_id]

            yield (anno, doc)

    def parse_document(self, anno: dict,
                       doc: dict) -> tuple[Document, list[Question]]:
        doc_id = doc["name"]
        contents = doc["contents"][0]

        if "tokens_layer" in contents:
            tokens_layer = contents["tokens_layer"]
        else:
            tokens_layer = contents["common_format"]
        tokens = tokens_layer["tokens"]
        token_bboxes = tokens_layer["positions"]
        page_struct = tokens_layer["structures"]["pages"]
        pages = []
        for i, (token_range, page_bbox) in enumerate(
                zip(page_struct["structure_value"], page_struct["positions"])):
            page_image_path = self.path / f"png/{doc_id}/{i}.png"
            if page_image_path.exists():
                page_image = PIL.Image.open(page_image_path)
            else:
                page_image = self._dummy_image
            pages.append(
                Page(
                    words=tokens[slice(*token_range)],
                    bboxes=token_bboxes[slice(*token_range)],
                    width=page_bbox[2],
                    height=page_bbox[3],
                    image=page_image,
                ))
        out_doc = Document(ident=doc_id, split=anno.get("split"), pages=pages)
        questions = [
            Question(feature_name=x["key"], text=x["key"])
            for x in anno["annotations"]
        ]
        return (out_doc, questions)


def _transform_output(key: str, output: RequestOutput) -> dict:
    generated_text = output.outputs[0].text
    values = [v.strip() for v in generated_text.split("|")]

    min_logprob = min(
        [list(lp.values())[0].logprob for lp in output.outputs[0].logprobs])
    score = math.exp(min_logprob)

    result = {
        "key": key,
        "values": [{
            "value": v
        } for v in values],
        "score": score,
    }
    return result


def run(args):
    loader = Loader(
        args.dataset,
        splits=(set(args.subset.split(",")) if args.subset != "" else None),
        limit_documents=args.limit_documents,
    )
    dataset = loader.load_dataset()

    # Create an LLM.
    load_start = time.perf_counter()
    llm = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))
    load_end = time.perf_counter()
    print(f"vLLM engine loading time {load_end - load_start}")

    tokenizer = llm.get_tokenizer()
    preprocessor = TiltPreprocessor.from_config(
        model_config=llm.model_config.hf_config,
        tokenizer=tokenizer.backend_tokenizer,
    )
    sampling_params = SamplingParams(
        temperature=0,
        logprobs=0,
        max_tokens=llm.model_config.hf_config.max_output_length,
    )

    i = 0
    request_mapping = {}
    for anno, doc_json in tqdm(dataset):
        document, questions = loader.parse_document(anno, doc_json)
        samples = preprocessor.preprocess(document, questions)
        for question, sample in zip(questions, samples):
            request_mapping[str(i)] = (document.ident, question.feature_name)
            llm.add_request(
                prompt=sample,
                request_id=f"{i}",
                params=sampling_params,
            )
            i += 1

    predictions = {}
    while llm.has_unfinished_requests():
        request_outputs = llm.step()
        for output in request_outputs:
            if output.finished:
                doc_id, key = request_mapping[output.request_id]
                if doc_id not in predictions:
                    predictions[doc_id] = {
                        "name": doc_id,
                        "annotations": [],
                    }
                predictions[doc_id]["annotations"].append(
                    _transform_output(key, output))

    # Print the outputs.
    with jsonlines.open(
            Path(args.output_dir or ".") /
            f"vllm_predictions_{datetime.now()}.jsonl", "w") as w:
        w.write_all(predictions.values())


async def _parse_documents(*, loader, preprocessor, generation_queue,
                           output_queue, parsing_complete_event):
    dataset = list(loader.load_dataset())

    i = 0
    for anno, doc_json in tqdm(dataset):
        document, questions = loader.parse_document(anno, doc_json)
        samples = preprocessor.preprocess(document, questions)
        await output_queue.put((
            "new_request",
            (document.ident, [question.feature_name
                              for question in questions]),
        ))
        for question, sample in zip(questions, samples):
            request_id = str(i)
            sample_id = (document.ident, question.feature_name)
            await generation_queue.put((request_id, sample_id, sample))
            i += 1
        await asyncio.sleep(0)

    parsing_complete_event.set()


async def _generate(*, llm, generation_queue, output_queue,
                    parsing_complete_event, sampling_params):
    while not (generation_queue.empty() and parsing_complete_event.is_set()):
        request = await generation_queue.get()
        request_id, sample_id, sample = request

        stream = llm.generate(
            prompt=sample,
            request_id=request_id,
            sampling_params=sampling_params,
        )
        final_output = None
        async for output in stream:
            if output.finished:
                final_output = output

        if final_output is None:
            raise RuntimeError(
                "Generation has not finished, but stream has ended.")

        result = _transform_output(sample_id[1], final_output)

        await output_queue.put(("prediction", (sample_id, result)))
        generation_queue.task_done()
        await asyncio.sleep(0)


async def _save_predictions(*, output_dir, output_queue):
    requested_document_predictions: dict[str, set[str]] = {}
    received_document_predictions: dict[str, set[str]] = defaultdict(set)
    predictions = {}
    with jsonlines.open(
            Path(args.output_dir or ".") /
            f"vllm_predictions_{datetime.now()}.jsonl",
            "w",
            flush=True,
    ) as w:
        while True:
            request = await output_queue.get()
            if request is None:
                break
            request_type, data = request
            match request_type:
                case "new_request":
                    document_id, questions = data
                    requested_document_predictions[document_id] = set(
                        questions)
                case "prediction":
                    sample_id, generated_result = data
                    document_id, question = sample_id

                    if document_id not in predictions:
                        predictions[document_id] = {
                            "name": document_id,
                            "annotations": [],
                        }
                    predictions[document_id]["annotations"].append(
                        generated_result)
                    received_document_predictions[document_id].add(question)
                    if (document_id in requested_document_predictions
                            and received_document_predictions[document_id]
                            == requested_document_predictions[document_id]):
                        w.write(predictions[document_id])
                        del predictions[document_id]
                        del requested_document_predictions[document_id]
                        del received_document_predictions[document_id]
            output_queue.task_done()
            await asyncio.sleep(0)


async def run_async(args):
    loader = Loader(
        args.dataset,
        splits=(set(args.subset.split(",")) if args.subset != "" else None),
        limit_documents=args.limit_documents,
    )

    # Create an LLM.
    load_start = time.perf_counter()
    llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))
    llm.start_background_loop()
    load_end = time.perf_counter()
    print(f"vLLM engine loading time {load_end - load_start}")

    tokenizer = await llm.get_tokenizer()
    model_config = await llm.get_model_config()
    preprocessor = TiltPreprocessor.from_config(
        model_config=model_config.hf_config,
        tokenizer=tokenizer.backend_tokenizer,
    )
    sampling_params = SamplingParams(
        temperature=0,
        logprobs=0,
        max_tokens=model_config.hf_config.max_output_length,
    )

    parsing_complete_event = asyncio.Event()
    generation_queue = asyncio.Queue(
        maxsize=args.async_parse_buffer_size or args.max_num_seqs * 3)
    output_queue = asyncio.Queue()

    all_tasks = []
    try:
        parse_task = asyncio.create_task(
            _parse_documents(
                loader=loader,
                preprocessor=preprocessor,
                generation_queue=generation_queue,
                output_queue=output_queue,
                parsing_complete_event=parsing_complete_event,
            ))
        all_tasks.append(parse_task)

        job_count = args.async_job_count or (args.max_num_seqs * 2)
        generation_tasks = [
            asyncio.create_task(
                _generate(
                    llm=llm,
                    generation_queue=generation_queue,
                    output_queue=output_queue,
                    parsing_complete_event=parsing_complete_event,
                    sampling_params=sampling_params,
                )) for _ in range(job_count)
        ]
        all_tasks.extend(generation_tasks)
        save_task = asyncio.create_task(
            _save_predictions(output_dir=args.output_dir,
                              output_queue=output_queue))
        all_tasks.append(save_task)
        await asyncio.gather(parse_task, *generation_tasks)
        await output_queue.put(None)  # Signal the save task to finish
        await asyncio.gather(save_task)
    except Exception as e:
        for task in all_tasks:
            task.cancel()
        raise e

    llm.shutdown_background_loop()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        "Run an inference on a DUE-format dataset using a TILT model")

    parser.add_argument(
        "--async",
        action="store_true",
        help="Run the inference asynchronously",
    )
    parser.add_argument(
        "--async-job-count",
        type=int,
        default=None,
        help="Number of parallel jobs to run for generation",
    )
    parser.add_argument(
        "--async-parse-buffer-size",
        type=int,
        default=None,
        help="Buffer size for parsing documents asynchronously",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="/datasets/DOCVQA/1_6/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--limit-documents",
        "-l",
        type=int,
        default=None,
        help="Limit the number of documents read from the dataset",
    )
    parser.add_argument(
        "--subset",
        "-s",
        type=str,
        default="test",
        help="Choose the subset of the dataset",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory, to which predictions will be saved",
    )

    parser = AsyncEngineArgs.add_cli_args(parser, async_args_only=False)
    parser.set_defaults(
        model="Snowflake/snowflake-arctic-tilt-v1.3",
        task="tilt_generate",
        scheduler_cls="vllm.tilt.scheduler.Scheduler",
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        max_num_seqs=16,
        enforce_eager=True,  # TODO: CUDA graphs for TILT Long
        disable_async_output_proc=True,  # Not implemented in TILT scheduler
        disable_log_requests=True,
    )
    args = parser.parse_args()
    if getattr(args, "async"):
        asyncio.run(run_async(args))
    else:
        run(args)
