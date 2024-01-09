import argparse
from typing import List
import logging
import os
import time
from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.master.server import (Server, add_server_arguments,
                                     initialize_ray_cluster)
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

def main(args: argparse.Namespace):
    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_ray_cluster(
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    # Create a server.
    server = Server(
        model=args.model,
        model_path=args.model_path,
        use_dummy_weights=args.use_dummy_weights,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_sequences=args.max_num_sequences,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
        gpu_memory=get_gpu_memory(),
        cpu_memory=get_cpu_memory(),
    )

    # Create a frontend.
    frontend = SimpleFrontend(
        model_name=args.model,
        block_size=args.block_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    block_size = frontend.block_size
    # Test the following inputs.
     # Add instruction first.
    prefix = 'Please translate these following English sentence(s) to German sentence(s):\n'

    # 8 fixed common examples of about the same length
    # that was generated independently from the WMT16 dataset.
    examples = [
        ('The new shopping center has a wide variety of stores, including clothing, electronics, and a supermarket.'
         ' => Das neue Einkaufszentrum bietet eine große Vielfalt an Geschäften, einschließlich Bekleidung, Elektronik und einem Supermarkt.'),
        # ('For a healthier lifestyle, try incorporating regular exercise, a balanced diet, and stress-reducing activities into your routine.'
        #  ' => Für einen gesünderen Lebensstil versuchen Sie, regelmäßige Bewegung, eine ausgewogene Ernährung und stressreduzierende Aktivitäten in Ihre Routine einzubauen.'),
        # ('The library will be hosting a series of workshops on various topics, such as creative writing.'
        #  ' => Die Bibliothek veranstaltet eine Reihe von Workshops zu verschiedenen Themen wie kreativem Schreiben.'),
        # ('The museum offers guided tours every day at 11:00 am and 4:00 pm, and admission is free on Sundays.'
        #  ' => Das Museum bietet jeden Tag um 11:00 Uhr und 16:00 Uhr Führungen an, und der Eintritt ist sonntags kostenlos.'),
        # ('If you experience any technical difficulties during the conference, please don\'t hesitate to contact the support team.'
        #  ' => Wenn Sie während der Konferenz technische Schwierigkeiten haben, zögern Sie bitte nicht, das Support-Team zu kontaktieren.'),
        # ('The local farmer\'s market offers fresh fruits, vegetables, and other produce directly from the farms every Saturday morning.'
        #  ' => Der örtliche Bauernmarkt bietet jeden Samstagmorgen frische Früchte, Gemüse und andere landwirtschaftliche Produkte direkt von den Höfen an.'),
        # ('Remember to set your clocks one hour forward for daylight saving time this weekend to enjoy longer days and more sunlight.'
        #  ' => Denken Sie daran, Ihre Uhren am Wochenende für die Sommerzeit eine Stunde vorzustellen, um längere Tage und mehr Sonnenlicht zu genießen.'),
        # ('The restaurant offers a diverse menu featuring international cuisine, including Italian, French, and Japanese dishes.'
        #  ' => Das Restaurant bietet eine vielfältige Speisekarte mit internationaler Küche, einschließlich italienischer, französischer und japanischer Gerichte'),
    ]
    prefix += '\n'.join(examples) + '\n'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    prefix_len = len(prefix_tokens)
    remainder_tokens = []
    if prefix_len % block_size != 0:
        remainder_tokens = prefix_tokens[-(prefix_len % block_size):]
        prefix_tokens = prefix_tokens[:-(prefix_len % block_size)]
        prefix_len = len(prefix_tokens)
    print(prefix)

    # Register prefix.
    logger.info('Registering prefix.')

    frontend._add_query(prefix_tokens, SamplingParams.from_dict({}))
    # logger.info('frontend._add_query.')
    server.register_prefix(frontend.get_inputs()[0][0])
    server.step()

    # Test the following inputs.
    test_inputs = [
       ( 'cheese =>'),
       ( 'I love you =>'),
    ]



    # # Warm up.
    # logger.info('Warming up.')
    # num_warmup_requests = 8
    # warmup_input_len = 8
    # warmup_output_len = 32
    # warmup_sampling_params = SamplingParams(
    #     n=1,
    #     temperature=1.0,
    #     top_p=0.99,
    #     max_num_steps=warmup_output_len,
    #     use_beam_search=False,
    #     stop_token_ids=set(),
    #     num_logprobs=0,
    #     context_window_size=None,
    #     prefix_id=None,
    # )
    # for _ in range(num_warmup_requests):
    #     frontend._add_query([0] * warmup_input_len, warmup_sampling_params)
    # server.add_sequence_groups(frontend.get_inputs())
    # while True:
    #     updated_seq_groups = server.step()
    #     if not server.has_unfinished_requests():
    #         break

    # Start benchmarking.
    logger.info('Start benchmarking.')

    request_sampling_params_dict = {
        'temperature': 0.0,
        'top_p': 1.0,
        'use_beam_search': False,
        'stop_token_ids': set(),
        'num_logprobs': 0,
        'context_window_size': None,
        'prefix_id': 0, # FIXME
    }

    latencies = []
    server.scheduler.reset_stats()
    while True:
        if test_inputs:
            text= test_inputs.pop(0)
            request_sampling_params = SamplingParams.from_dict(request_sampling_params_dict)
            request_sampling_params = frontend.add_eos_token(request_sampling_params)
            frontend.query(text, request_sampling_params)
        server.add_sequence_groups(frontend.get_inputs())
        start_time = time.time()
        updated_seq_groups = server.step()
        for seq_group in updated_seq_groups:
            if seq_group.is_finished():
                end_time = time.time()
                frontend.print_response(seq_group)
                latency = end_time - start_time
                print(latency)
                latencies.append(latency)
        if not (server.has_unfinished_requests() or test_inputs):
            break
    print(f'Avg latency: {np.mean(latencies)} seconds')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow simple server.')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    main(args)
