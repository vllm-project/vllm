# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full sparse-layer execution using the adapted ATOM kernel."""

from __future__ import annotations

import flydsl.compiler as flyc
import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerMetadata
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.platforms import current_platform

from ..mono.layer.config import HIDDEN, MAX_INDEX_BLOCKS, MAX_TOKENS, O_K
from ..mono.layer.kernels.post_attn import K4_ABI, build_post_attn_kernel
from ..mono.layer.kernels.pre_attn import K1_ARGS
from ..mono.layer.kernels.pre_attn import SCRATCH_BYTES as PRE_BYTES
from ..mono.layer.layout import SCRATCH, SCRATCH_BYTES, sym_layout
from ..mono.layer.step import (
    ERROR_BYTES,
    FENCE_BYTES,
    begin_fence,
    check_diagnostics,
    expand_rows,
)
from ..mono.layer.weights import LayerWeights
from ..mono.residency import active_workgroups_per_cu
from ..mono.runner import _writable_flydsl_cache
from .executor import MonoHealth, mark_peers_failed, tp_agree

logger = init_logger(__name__)


class LayerExecutor:
    uses_caches = True

    def __init__(self, layers, weights, widths, tp):
        _writable_flydsl_cache()
        group = get_tp_group()
        self.rank, self.tp = group.rank_in_group, tp
        self.layers = {layer.layer_id: layer for layer in layers}
        self.widths = widths
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.weights = {}
        tables: dict[int, torch.Tensor] = {}
        failure = None
        try:
            for layer in layers:
                self.weights[layer.layer_id] = LayerWeights.from_layer(layer, tables)
        except Exception as error:
            failure = error
        if not tp_agree(failure is None):
            raise RuntimeError(
                "full-layer weight packing failed on a TP rank"
            ) from failure

        self.peer_size = sym_layout(tp)["_bytes"]
        self.peer = PeerBuffer(
            ERROR_BYTES + self.peer_size + FENCE_BYTES,
            group.cpu_group,
            self.rank,
            tp,
            self.device,
        )
        self.peer.bytes.zero_()
        self.peers = self.peer.addresses + ERROR_BYTES
        self.sym = self.peer.local + ERROR_BYTES
        self.peer_data = self.peer.bytes[ERROR_BYTES : ERROR_BYTES + self.peer_size]
        self.fault_word = self.peer.bytes[:16].view(torch.int32)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.pre = torch.zeros(PRE_BYTES, dtype=torch.uint8, device=self.device)
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=self.device)
        self.h = torch.empty(
            MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        self.q = torch.empty(MAX_TOKENS, O_K, dtype=torch.bfloat16, device=self.device)
        self.iq = torch.empty(MAX_TOKENS, 128, dtype=torch.bfloat16, device=self.device)
        self.out = [torch.empty_like(self.h) for _ in range(2)]
        self.residual = [torch.empty_like(self.h) for _ in range(2)]
        self.main_table = torch.empty(
            MAX_TOKENS, MAX_INDEX_BLOCKS, dtype=torch.int32, device=self.device
        )
        self.index_table = torch.empty_like(self.main_table)
        self.seq_lens = torch.empty(MAX_TOKENS, dtype=torch.int32, device=self.device)
        self.slots = torch.empty(MAX_TOKENS, dtype=torch.int64, device=self.device)
        self.index_slots = torch.empty_like(self.slots)
        self.batch_ids = torch.empty_like(self.seq_lens)
        self.bindings = {}
        self.k1_args = {}
        self.compiled = {}
        self.steps = self.outputs = 0
        self.launches = dict.fromkeys(self.layers, 0)
        first = layers[0].self_attn
        sparse = getattr(first.indexer, "impl", first.indexer)
        self.init_blocks, self.local_blocks = sparse.init_blocks, sparse.local_blocks
        self.query_len = 1
        self._slots = None

        failure = None
        try:
            cus = current_platform.num_compute_units(self.device.index)
            for width in widths:
                launch = build_post_attn_kernel(
                    tp,
                    first.scaling,
                    layers[0].input_layernorm.variance_epsilon,
                    2.0,
                    1.0,
                    7.0,
                    self.init_blocks,
                    self.local_blocks,
                    tokens=width,
                    fuse_k1=True,
                    debug=True,
                )
                compiled = flyc.compile(launch, *K4_ABI.zeros())
                from ..mono.layer.support.device.ops import kernel_symbol

                name = kernel_symbol(
                    "minimax_m3_mono_layer",
                    tp=tp,
                    s=width,
                    ib=self.init_blocks,
                    lb=self.local_blocks,
                    ih=1,
                    io=0,
                    tl=0,
                    it=1,
                )
                resident = active_workgroups_per_cu(compiled, name, 512)
                if resident * cus < 256:
                    raise RuntimeError(f"full-layer width {width} is not resident")
                self.compiled[width] = compiled
            torch.accelerator.synchronize()
        except Exception as error:
            failure = error
        if not tp_agree(failure is None):
            self.peer.close()
            raise RuntimeError(
                "full-layer compilation failed on a TP rank"
            ) from failure
        self._kernels = {
            layer_id: self._layer_kernel(layer_id) for layer_id in self.layers
        }
        logger.info(
            "MiniMax-M3 full-layer kernel: widths %s, bounded waits, step-end TP fence",
            widths,
        )

    def bind(self, layer_id, binding):
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_sparse_block_page_stride,
        )

        if (
            minimax_m3_sparse_block_page_stride(binding.key_pages, binding.value_pages)
            != 16
        ):
            raise ValueError("full-layer decode requires interleaved K/V cache blocks")
        self.bindings[layer_id] = binding
        w = self.weights[layer_id]
        pointers = dict(
            ar=0,
            g_in=w.input_norm.data_ptr(),
            w_qkv=w.qkv.data_ptr(),
            s_qkv=w.qkv_scale.data_ptr(),
            g_q=w.q_norm.data_ptr(),
            g_k=w.k_norm.data_ptr(),
            g_iq=w.index_q_norm.data_ptr(),
            g_ik=w.index_k_norm.data_ptr(),
            cos_sin=w.cos_sin.data_ptr(),
            index_cache=binding.index_key.data_ptr(),
            iq_out=self.iq.data_ptr(),
            scratch=self.pre.data_ptr(),
        )
        self.k1_args[layer_id] = torch.tensor(
            [pointers[name] for name in K1_ARGS], dtype=torch.int64, device=self.device
        )

    def begin_step(self, slots):
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        first = next(iter(self.layers.values())).self_attn
        main = metadata[first.layer_name]
        index = metadata[first.indexer.index_cache.prefix]
        assert isinstance(main, MiniMaxM3SparseMetadata)
        assert isinstance(index, MiniMaxM3IndexerMetadata)
        assert main.decode is not None and index.decode is not None
        self.query_len = main.decode.decode_query_len
        self._slots = slots
        rows = slots.numel()
        expand_rows[(rows, MAX_INDEX_BLOCKS // 256)](
            main.decode.block_table,
            index.decode.block_table,
            main.decode.seq_lens,
            slots,
            index.slot_mapping,
            self.main_table,
            self.index_table,
            self.seq_lens,
            self.slots,
            self.index_slots,
            self.batch_ids,
            MAIN_STRIDE=main.decode.block_table.stride(0),
            INDEX_STRIDE=index.decode.block_table.stride(0),
            WIDTH=MAX_INDEX_BLOCKS,
            Q=self.query_len,
            REQUESTS=main.num_decodes,
            BLOCK=256,
        )
        self.pre.zero_()
        self.scratch.zero_()
        self.peer_data.zero_()
        begin_fence(
            self.peers.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            self.peer_size,
            self.tp,
            stream=torch.cuda.current_stream(),
        )

    def _layer_kernel(self, layer_id):
        w = self.weights[layer_id]
        f = w.ffn
        parity = (layer_id - min(self.layers)) % 2
        final = layer_id == max(self.layers)

        def run(positions, hidden_states, residual):
            rows = hidden_states.shape[0]
            binding = self.bindings[layer_id]
            values = dict(
                h_in=self.h.data_ptr(),
                q=self.q.data_ptr(),
                block_table=self.main_table.data_ptr(),
                seq_lens=self.seq_lens.data_ptr(),
                k_cache=binding.key_pages.data_ptr(),
                v_cache=binding.value_pages.data_ptr(),
                k_scale=binding.k_scale.data_ptr(),
                v_scale=binding.v_scale.data_ptr(),
                w_o=w.output.data_ptr(),
                s_o=w.output_scale.data_ptr(),
                g_post=f.post_norm.data_ptr(),
                w_gate=w.router_bf16.data_ptr(),
                bias=f.router_bias.data_ptr(),
                w13=f.w13.data_ptr(),
                s13=f.s13.data_ptr(),
                w2=f.w2.data_ptr(),
                s2=f.s2.data_ptr(),
                h_mid=self.residual[parity].data_ptr(),
                ar_out=self.out[parity].data_ptr(),
                scratch=self.scratch.data_ptr(),
                sym=self.sym,
                peers=self.peers.data_ptr(),
                rank=self.rank,
                layer=layer_id,
                bt_width=MAX_INDEX_BLOCKS,
                q_len=self.query_len,
                tl=0,
                k1_args=self.k1_args[layer_id].data_ptr(),
                positions=positions.data_ptr(),
                slot_mapping=self.slots.data_ptr(),
                res=residual.data_ptr(),
                batch_ids=self.batch_ids.data_ptr(),
                index_block_table=self.index_table.data_ptr(),
                index_bt_width=MAX_INDEX_BLOCKS,
                index_slot_mapping=self.index_slots.data_ptr(),
                ar_in=hidden_states.data_ptr(),
                mode=1,
            )
            self.compiled[rows](*K4_ABI.pack(values), torch.cuda.current_stream())
            self.launches[layer_id] += 1
            if final:
                diagnostic = self.scratch[SCRATCH["diag"] :].view(torch.int32)
                check_diagnostics[(1,)](
                    diagnostic, self.peers, self.rank, self.epoch, TP=self.tp
                )
                # Every rank must publish its diagnostics before any rank can
                # pass the model's error-word check and return a sampled token.
                begin_fence(
                    self.peers.data_ptr(),
                    self.rank,
                    self.epoch.data_ptr(),
                    self.peer_size,
                    self.tp,
                    stream=torch.cuda.current_stream(),
                )
            return self.out[parity][:rows], self.residual[parity][:rows]

        return run

    def layer(self, layer_id):
        return self._kernels.get(layer_id)

    def health(self):
        words = self.fault_word.tolist()
        return MonoHealth(bool(words[0]), *words[1:])

    def abort(self):
        mark_peers_failed(self.peer, self.rank)

    def after_step(self):
        self.steps += 1

    def output_step(self):
        self.outputs += 1
        if self.outputs % 1024 == 0 and self.rank == 0:
            logger.info("MiniMax-M3 full-layer kernel: %d output steps", self.outputs)

    def close(self):
        self.peer.close()
