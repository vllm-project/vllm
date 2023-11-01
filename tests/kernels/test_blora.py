import pytest
from vllm.model_executor.parallel_utils.layers import BLoraColumnParallelLinear, BLoraRowParallelLinear
from peft.utils.other import transpose
from peft.tuners.lora import Linear
import torch
import torch.nn.functional as F



LORA_DROP_OUTS = [0.0, 0.5]
INPUT_SIZE = [8, 16, 16]
OUTPUT_SIZE = [16, 16, 32]
ADAPTER_NAMES = [["lora1", "lora2"], ["lora1", "lora2", "lora3"]]
LORA_ALPHA = [4, 6, 8, 10]
BIAS = [True, False]
SEEDS = [0]
R = [1, 2, 4, 8]

REDUCE_RESULTS = [True, False]

IS_INPUT_PARALLEL = [True, False]

class RefBLinear(Linear):
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            assert x.size(0) == len(self.batch_lora_ids)

            batch = list(zip(x, self.batch_lora_ids))
            # rewrite as for loop
            lora_out = torch.zeros_like(result)
            for i, (xi, lora_id) in enumerate(batch):
                if lora_id in self.lora_A.keys():
                    lora_out[i] = self.scaling[lora_id] * self.lora_B[lora_id](
                        self.lora_A[lora_id](self.lora_dropout[lora_id](xi))
                    )
            result += lora_out

        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)
        return result, self.bias


@pytest.mark.parametrize("adapter_names", ADAPTER_NAMES)
@pytest.mark.parametrize("output_size", OUTPUT_SIZE)
@pytest.mark.parametrize("input_size", INPUT_SIZE)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("r", R)
@pytest.mark.parametrize("lora_alpha", LORA_ALPHA)
@pytest.mark.parametrize("lora_drop_out", LORA_DROP_OUTS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_column_blora(
    adapter_names: list[str],
    input_size: int,
    output_size: int,
    bias: bool,
    r: int,
    lora_alpha: int,
    lora_drop_out: float,
    seed: int,
    init_lora_weights = True,
):
    ref_blinear = None
    column_blora = None
    skip_bias = not bias
    # create model
    for i in range(len(adapter_names)):
        adapter_name = adapter_names[i]
        if i == 0:
            ref_blinear = RefBLinear(adapter_name=adapter_name,
                                     in_features=input_size,
                                     out_features=output_size,
                                     r=r,
                                     lora_alpha=lora_alpha,
                                     bias=bias)
            column_blora = BLoraColumnParallelLinear(input_size=input_size,
                                                     output_size=output_size,
                                                     adapter_name=adapter_name,
                                                     bias=bias,
                                                     skip_bias_add=skip_bias,
                                                     r=r,
                                                     lora_alpha=lora_alpha,
                                                     lora_dropout=lora_drop_out)
        else:
            ref_blinear.update_layer(adapter_name=adapter_name,
                                     r=r,
                                     lora_alpha=lora_alpha,
                                     lora_dropout=lora_drop_out,
                                     init_lora_weights=init_lora_weights)
            column_blora.update_layer(adapter_name=adapter_name,
                                      r=r,
                                      lora_alpha=lora_alpha,
                                      lora_dropout=lora_drop_out,
                                      init_lora_weights=init_lora_weights)
    ref_blinear.cuda()
    column_blora.cuda()
    # prepare inputs
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scale = float(input_size**-0.5)
    x = torch.empty(len(adapter_names), input_size, device="cuda")
    x.uniform_(-scale, scale)
    setattr(ref_blinear, "batch_lora_ids", adapter_names)
    setattr(column_blora, "batch_lora_ids", adapter_names)
    batch_token_length = []
    for i in range(len(adapter_names)):
        batch_token_length.append(1)
    setattr(column_blora, "batch_token_lengths", batch_token_length)

    # align weights
    column_blora.weight.copy_(ref_blinear.weight)
    assert torch.allclose(column_blora.weight,
                          ref_blinear.weight,
                          atol=1e-8, rtol=1e-8)
    if column_blora.bias is not None:
        column_blora.bias.copy_(ref_blinear.bias)
        assert torch.allclose(column_blora.bias, ref_blinear.bias,
                              atol=1e-8, rtol=1e-8)

    for lora_id, adapter in column_blora.lora_A.items():
        adapter.weight.copy_(ref_blinear.lora_A[lora_id].weight)
        assert torch.allclose(adapter.weight,
                              ref_blinear.lora_A[lora_id].weight,
                              atol=1e-8,
                              rtol=1e-8)

    #test inputs
    ref_output, _ = ref_blinear.forward(x)
    col_output, _ = column_blora.forward(x)

    assert torch.allclose(ref_output, col_output, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("adapter_names", ADAPTER_NAMES)
@pytest.mark.parametrize("output_size", OUTPUT_SIZE)
@pytest.mark.parametrize("input_size", INPUT_SIZE)
@pytest.mark.parametrize("reduce_results", REDUCE_RESULTS)
@pytest.mark.parametrize("r", R)
@pytest.mark.parametrize("lora_alpha", LORA_ALPHA)
@pytest.mark.parametrize("lora_drop_out", LORA_DROP_OUTS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("input_parallel", IS_INPUT_PARALLEL)
@torch.inference_mode()
def test_row_blora(
    adapter_names: list[str],
    input_size: int,
    output_size: int,
    reduce_results: bool,
    r: int,
    lora_alpha: int,
    lora_drop_out: float,
    seed: int,
    input_parallel: bool,
    init_lora_weights = True,
):
    ref_blinear = None
    row_blora = None
    bias = reduce_results
    skip_bias = not bias

    # create model
    for i in range(len(adapter_names)):
        adapter_name = adapter_names[i]
        if i == 0:
            ref_blinear = RefBLinear(adapter_name=adapter_name,
                                     in_features=input_size,
                                     out_features=output_size,
                                     r=r,
                                     lora_alpha=lora_alpha,
                                     bias=bias)
            row_blora = BLoraRowParallelLinear(input_size=input_size,
                                               output_size=output_size,
                                               adapter_name=adapter_name,
                                               bias=bias,
                                               skip_bias_add=skip_bias,
                                               r=r,
                                               lora_alpha=lora_alpha,
                                               lora_dropout=lora_drop_out,
                                               is_input_parallel=input_parallel,
                                               reduce_results=reduce_results)
        else:
            ref_blinear.update_layer(adapter_name=adapter_name,
                                     r=r,
                                     lora_alpha=lora_alpha,
                                     lora_dropout=lora_drop_out,
                                     init_lora_weights=init_lora_weights)
            row_blora.update_layer(adapter_name=adapter_name,
                                   r=r,
                                   lora_alpha=lora_alpha,
                                   lora_dropout=lora_drop_out,
                                   init_lora_weights=init_lora_weights)
    ref_blinear.cuda()
    row_blora.cuda()
    # prepare inputs
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scale = float(input_size**-0.5)
    x = torch.empty(len(adapter_names), input_size, device="cuda")
    x.uniform_(-scale, scale)
    setattr(ref_blinear, "batch_lora_ids", adapter_names)
    setattr(row_blora, "batch_lora_ids", adapter_names)
    batch_token_length = []
    for i in range(len(adapter_names)):
        batch_token_length.append(1)
    setattr(row_blora, "batch_token_lengths", batch_token_length)

    # align weights
    row_blora.weight.copy_(ref_blinear.weight)
    assert torch.allclose(row_blora.weight, ref_blinear.weight,
                           atol=1e-8, rtol=1e-8)
    if row_blora.bias is not None:
        row_blora.bias.copy_(ref_blinear.bias)
        assert torch.allclose(row_blora.bias, ref_blinear.bias
                              , atol=1e-8, rtol=1e-8)
    for lora_id, adapter in row_blora.lora_A.items():
        adapter.weight.copy_(ref_blinear.lora_A[lora_id].weight)
        adapter_weight = adapter.weight
        ref_weight = ref_blinear.lora_A[lora_id].weight
        assert torch.allclose(adapter_weight, ref_weight, atol=1e-8, rtol=1e-8)

    #test inputs
    ref_output, _ = ref_blinear.forward(x)
    row_output, _ = row_blora.forward(x)

    assert torch.allclose(ref_output, row_output, atol=1e-8, rtol=1e-8)
