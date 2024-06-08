import torch
import triton

from tests.lora.test_sgmv_triton import MAX_TEST_POWER, setup_
from vllm.model_executor.layers.lora import sgmv_triton as sgmv


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['S'],  # argument names to use as an x-axis for the plot
        x_vals=[16 * 2**i for i in range(3, 6)] +
        [4096],  # different possible values for `x_name`
        line_arg=
        'R',  # argument name which corresponds to a different line in the plot
        line_vals=[64, None],  # possible values for `line_arg``
        line_names=['Rank=64', f'Random Rank up to {2**MAX_TEST_POWER}'
                    ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name=
        "sgmv",  # name for the plot. Used as file name for saving the plot too.
        args={},
    ))
def benchmark_repeats_expand(S, R, repeats_per_lora=1):
    weights, ranks, indices, repeats, _, R, dtype = setup_(
        S, R, 4096, dtype=torch.bfloat16, repeats_per_lora=repeats_per_lora)

    buffer = torch.randn((S, R), device='cuda', dtype=torch.float32)
    out = torch.randn((S, 4096), device='cuda', dtype=dtype)
    ms = triton.testing.do_bench(lambda: sgmv.sgmv_expand(buffer,
                                                          weights,
                                                          out,
                                                          ranks,
                                                          indices,
                                                          repeats,
                                                          repeats_per_lora,
                                                          out_col_offset=0),
                                 warmup=500,
                                 rep=4000)
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['S'],  # argument names to use as an x-axis for the plot
        x_vals=[16 * 2**i for i in range(3, 6)] +
        [4096],  # different possible values for `x_name`
        line_arg=
        'R',  # argument name which corresponds to a different line in the plot
        line_vals=[64, None],  # possible values for `line_arg``
        line_names=['Rank=64', f'Random Rank up to {2**MAX_TEST_POWER}'
                    ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name=
        "sgmv",  # name for the plot. Used as file name for saving the plot too.
        args={},
    ))
def benchmark_repeats_shrink(S, R, repeats_per_lora=1):
    weights, ranks, indices, repeats, _, R, dtype = setup_(
        S, R, 4096, dtype=torch.bfloat16, repeats_per_lora=repeats_per_lora)

    x = torch.rand((S, 4096), device='cuda', dtype=dtype)
    out = torch.zeros((S, R), device='cuda', dtype=torch.float32)
    ms = triton.testing.do_bench(lambda: sgmv.sgmv_shrink(
        x, weights, out, ranks, indices, repeats, repeats_per_lora),
                                 warmup=500,
                                 rep=4000)
    return ms


if __name__ == '__main__':
    # NOTE: the random rank benchmark is random ranks up to 2^MAX_TEST_POWER,
    # not random up to the rank specified,
    # so it doesn't change when you change the rank you're testing
    print('These benchmarks can vary a decent amount sometimes. ', end='')
    print('They should be consistent across increasing seq length ')
    print('(slower but with strong superlinear scaling), ', end='')
    print('consistently faster with random ranks up to the same rank and')
    print('faster with increasing repeats per lora.')
    print('Times are in ms.')
    print('-' * 40)
    print('Expand | repeats [1]')
    benchmark_repeats_expand.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=1)
    print('-' * 40)
    print('Shrink | repeats [1]')
    benchmark_repeats_shrink.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=1)

    print('-' * 40)
    print('Expand | repeats [8]')
    benchmark_repeats_expand.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=8)
    print('-' * 40)
    print('Shrink | repeats [8]')
    benchmark_repeats_shrink.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=8)

    # # set repeats >= 16 for plaid mode
    # # (tl.dot is applicable which makes it fast)
    print('-' * 40)
    print('Expand | repeats [16]')
    benchmark_repeats_expand.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=16)
    print('-' * 40)
    print('Shrink | repeats [16]')
    benchmark_repeats_shrink.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=16)

    print('-' * 40)
    print('Expand | repeats [32]')
    benchmark_repeats_expand.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=32)
    print('-' * 40)
    print('Shrink | repeats [32]')
    benchmark_repeats_shrink.run(show_plots=False,
                                 print_data=True,
                                 repeats_per_lora=32)
    print('-' * 40)
