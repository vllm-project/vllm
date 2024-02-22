import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
from torch._inductor.virtualized import V

vllm_lib = torch.library.Library("vllm", "DEF")

# Available starting PT 2.2
class NoneLayout(ir.IRNode):
    def __init__(self, device):
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        return 0

    def as_fixed(self):
        return self


# Available starting PT 2.2
class MutationOutput(ir.ExternKernel):
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def __init__(self, layout, input, parent):
        super().__init__(None, layout, [input, parent], ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def is_no_op(self):
        return True

    def has_side_effects(self):
        return True

    def get_alias_names(self):
        return [self.inputs[0].get_name()]


class VllmCudaKernel(ir.FallbackKernel):
    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    @classmethod
    def create(cls, kernel, *args, mutated_inputs=[], **kwargs) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        packed = cls(
            NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
        )
        # Mark inplace inputs as mutated
        for kernel_input in mutated_inputs:
            V.graph.mark_buffer_mutated(kernel_input.get_name())
            MutationOutput(kernel_input.layout, kernel_input, packed)

def register_vllm_lowering(op, mutating_inputs):
    lowering.fallbacks.add(op)

    @lowering.register_lowering(op, type_promotion_kind=None)
    def op_lowering(
        *args,
        **kwargs,
    ):
        VllmCudaKernel.create(
            op.default,
            *args,
            **kwargs,
            mutated_inputs=mutating_inputs,
        )
        returns = [args[i] for i in mutating_inputs]
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


# lib.define(
#     "paged_attention_v2(Tensor out, Tensor exp_sums, Tensor max_logits, Tensor tmp_out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
# )


# @torch.library.impl(lib, "paged_attention_v2", "Meta")
# def _paged_attention_v2_meta(
#     out,
#     exp_sums,
#     max_logits,
#     tmp_out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     return out.contiguous()


# @torch.library.impl(lib, "paged_attention_v2", "CUDA")
# def _paged_attention_v2(
#     out,
#     exp_sums,
#     max_logits,
#     tmp_out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     out = out.contiguous()
#     exp_sums = exp_sums.contiguous()
#     max_logits = max_logits.contiguous()
#     tmp_out = tmp_out.contiguous()
#     query = query.contiguous()
#     key_cache = key_cache.contiguous()
#     value_cache = value_cache.contiguous()
#     block_tables = block_tables.contiguous()
#     context_lens = context_lens.contiguous()

#     attn_ops.paged_attention_v2(
#         out,
#         exp_sums,
#         max_logits,
#         tmp_out,
#         query,
#         key_cache,
#         value_cache,
#         num_kv_heads,
#         scale,
#         block_tables,
#         context_lens,
#         block_size,
#         max_context_len,
#         alibi_slopes,
#     )
#     return out


# lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v2)


# @lowering.register_lowering(
#     torch.ops.paged_attention.paged_attention_v2, type_promotion_kind=None
# )
# def _paged_attention_v2_lowering(
#     out,
#     exp_sums,
#     max_logits,
#     tmp_out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     VllmCudaKernel.create(
#         torch.ops.paged_attention.paged_attention_v2.default,
#         out,
#         exp_sums,
#         max_logits,
#         tmp_out,
#         query,
#         key_cache,
#         value_cache,
#         num_kv_heads,
#         scale,
#         block_tables,
#         context_lens,
#         block_size,
#         max_context_len,
#         alibi_slopes,
#         mutated_inputs=[out],
#     )
#     return out


# lib.define(
#     "paged_attention_v1(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
# )


# @torch.library.impl(lib, "paged_attention_v1", "Meta")
# def _paged_attention_v1_meta(
#     out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     return out.contiguous()


# @torch.library.impl(lib, "paged_attention_v1", "CUDA")
# def _paged_attention_v1(
#     out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     out = out.contiguous()
#     query = query.contiguous()
#     key_cache = key_cache.contiguous()
#     value_cache = value_cache.contiguous()
#     block_tables = block_tables.contiguous()
#     context_lens = context_lens.contiguous()

#     attn_ops.paged_attention_v1(
#         out,
#         query,
#         key_cache,
#         value_cache,
#         num_kv_heads,
#         scale,
#         block_tables,
#         context_lens,
#         block_size,
#         max_context_len,
#         alibi_slopes,
#     )
#     return out


# lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v1)


# @lowering.register_lowering(
#     torch.ops.paged_attention.paged_attention_v1, type_promotion_kind=None
# )
# def _paged_attention_v1_lowering(
#     out,
#     query,
#     key_cache,
#     value_cache,
#     num_kv_heads,
#     scale,
#     block_tables,
#     context_lens,
#     block_size,
#     max_context_len,
#     alibi_slopes=None,
# ):
#     PagedAttnKernel.create(
#         torch.ops.paged_attention.paged_attention_v1.default,
#         out,
#         query,
#         key_cache,
#         value_cache,
#         num_kv_heads,
#         scale,
#         block_tables,
#         context_lens,
#         block_size,
#         max_context_len,
#         alibi_slopes,
#         mutated_inputs=[out],
#     )
#     return out
