# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Tuple, Union

import torch
from torchdiffeq import odeint

from vllm.transformers_utils.qwen2_code2wav_dit.model.dit import DiT
from vllm.transformers_utils.qwen2_code2wav_dit.model.t2w_cfm import CodecCFM
from vllm.transformers_utils.qwen2_code2wav_dit.model.utils import (
    exists, load_checkpoint)
from vllm.transformers_utils.qwen2_code2wav_dit.modeling_qwen2_code2wav import (
    Qwen2Code2wavBigvgan)


class CudaGraphRunner:

    def __init__(self, fn, device):
        """
        initialize CUDA Graph Wrapper.

        Args:
            original_fast_forward (callable): original fast_forward method.
            device (torch.device): CUDA device.
        """
        torch._dynamo.config.cache_size_limit = 64
        self.fn_compile = torch.compile(
            fn,
            mode="default",
            fullgraph=False,
        )
        self.cuda_graph: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.input_buffers: Dict[Tuple[int, int], Dict[str,
                                                       torch.Tensor]] = dict()
        self.output_buffers: Dict[Tuple[int, int], torch.Tensor] = dict()

        self.device = device

        # Create customized CUDA stream for Cuda Graph capture
        self.capture_stream = torch.cuda.Stream(device=self.device)

    def capture_cuda_graph(self, x, cond, spk, text, time, mask):
        """
        Capture CUDA Graph。

        Args:
            x (torch.Tensor): nosied input audio.
            cond (torch.Tensor): masked cond audio.
            spk (torch.Tensor): spk embedding.
            text (torch.Tensor): text.
            time (torch.Tensor): time step.
            mask (torch.Tensor or None): mask.
        """
        # Move the input data to the specified device and detach it from the computation graph.

        size = (text.size(0), text.size(1))

        with torch.no_grad():
            if size not in self.input_buffers:
                self.input_buffers[size] = {
                    "x":
                    x.to(self.device, non_blocking=True).clone().detach(),
                    "cond":
                    cond.to(self.device, non_blocking=True).clone().detach(),
                    "spk":
                    spk.to(self.device, non_blocking=True).clone().detach(),
                    "text":
                    text.to(self.device, non_blocking=True).clone().detach(),
                    "time":
                    time.to(self.device, non_blocking=True).clone().detach(),
                    "mask":
                    mask.to(self.device, non_blocking=True).clone().detach()
                    if mask is not None else None,
                }

        # Determine the output shape through a single forward pass and pre-allocate the output buffer.
        with torch.no_grad():
            if size not in self.output_buffers:
                generated = self.fn_compile(
                    x=self.input_buffers[size]["x"],
                    cond=self.input_buffers[size]["cond"],
                    spk=self.input_buffers[size]["spk"],
                    text=self.input_buffers[size]["text"],
                    time=self.input_buffers[size]["time"],
                    mask=self.input_buffers[size]["mask"],
                )
                self.output_buffers[size] = torch.empty_like(
                    generated, device=self.device)

        # Ensure that all previous operations are complete.
        torch.cuda.synchronize(self.device)

        # Begin to capture CUDA Graph
        self.cuda_graph[size] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph[size],
                              stream=self.capture_stream):
            # Perform the forward pass and copy the results to the pre-allocated buffer.
            generated = self.fn_compile(
                x=self.input_buffers[size]["x"],
                cond=self.input_buffers[size]["cond"],
                spk=self.input_buffers[size]["spk"],
                text=self.input_buffers[size]["text"],
                time=self.input_buffers[size]["time"],
                mask=self.input_buffers[size]["mask"],
            )
            self.output_buffers[size].copy_(generated)

        # Make sure capture complete
        torch.cuda.synchronize(self.device)

    def __call__(self, x, cond, spk, text, time, mask=None):
        """
        Args:
            x (torch.Tensor): nosied input audio.
            cond (torch.Tensor): masked cond audio.
            spk (torch.Tensor): spk embedding.
            text (torch.Tensor): text.
            time (torch.Tensor): time step.
            mask (torch.Tensor or None): mask.

        Returns:
            torch.Tensor: generated
        """

        size = (text.size(0), text.size(1))

        if size not in self.cuda_graph:
            # Capture the CUDA Graph on the first call.
            self.capture_cuda_graph(x, cond, spk, text, time, mask)

        # Update input to buffer.
        self.input_buffers[size]["x"].copy_(
            x.to(self.device, non_blocking=True))
        self.input_buffers[size]['cond'].copy_(
            cond.to(self.device, non_blocking=True))
        self.input_buffers[size]["spk"].copy_(
            spk.to(self.device, non_blocking=True))
        self.input_buffers[size]["text"].copy_(
            text.to(self.device, non_blocking=True))
        self.input_buffers[size]["time"].copy_(
            time.to(self.device, non_blocking=True))
        if self.input_buffers[size]["mask"] is not None and mask is not None:
            self.input_buffers[size]["mask"].copy_(
                mask.to(self.device, non_blocking=True))
        elif mask is None:
            self.input_buffers[size]["mask"] = None
        # Replay CUDA Graph
        self.cuda_graph[size].replay()
        return self.output_buffers[size]


class BatchCodecCFM(CodecCFM):

    @torch.no_grad()
    def fast_block_sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        codec: int["b nc dc"],
        ref_mel: float["b n d"],  # noqa: F722
        y0: float["b n d"],
        lens: int[b] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]]
        | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        max_duration = y0.shape[1]
        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()
            ref_mel = ref_mel.half()
            y0 = y0.half()
            # print(next(self.parameters()).dtype)

        # raw wave

        cond = cond.unsqueeze(1).repeat(1, max_duration, 1)
        batch, cond_seq_len, device = *ref_mel.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch, ),
                              cond_seq_len,
                              device=device,
                              dtype=torch.long)

        mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            out_put = self.transformer.fast_forward(
                x=x,
                text=codec,
                spk=cond,
                cond=ref_mel,
                time=t,
                mask=mask,
            )
            pred, null_pred = torch.chunk(out_put, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        t_start = 0
        t = torch.linspace(t_start,
                           1,
                           steps,
                           device=self.device,
                           dtype=ref_mel.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, ref_mel, out)
        return out, trajectory


class Qwen2Code2wavDit(torch.nn.Module):

    def __init__(self, ckpt, frequency: str = "50hz", device='cpu'):
        super().__init__()
        self.frequency = frequency
        self.device = device
        self.dit = DiT(
            dim=1024,
            depth=22 if frequency == '50hz' else 32,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            use_codec=True,
            repeats=2 if frequency == '50hz' else 4,
            attn_processor='stream_block_sr'
            if frequency == '50hz' else 'stream_block_8_L_4',
            text_num_embeds=8193 if frequency == '50hz' else 32769,
            mel_dim=80,
        )
        self.mel_spec_kwargs = dict(
            target_sample_rate=16000,
            n_mel_channels=80,
            hop_length=160,
        )
        self.odeint_kwargs = dict(method="euler", )
        self.cfm_model = BatchCodecCFM(
            transformer=self.dit,
            mel_spec_kwargs=self.mel_spec_kwargs,
            odeint_kwargs=self.odeint_kwargs,
        ).to(device)
        self.cfm_model = load_checkpoint(self.cfm_model,
                                         ckpt,
                                         device,
                                         use_ema=True)

    def sample(self,
               cond,
               ref_mel,
               codec,
               steps=10,
               cfg_strength=0.5,
               sway_sampling_coef=-1.0):
        y_all = torch.randn([1, 30000, 80],
                            device=self.device,
                            dtype=ref_mel.dtype)
        expect_y_len = codec.shape[1] * (2 if self.frequency == '50hz' else 4)
        y0 = y_all[:, :expect_y_len]
        with torch.inference_mode():
            generated, _ = self.cfm_model.sample(
                cond=cond,
                ref_mel=ref_mel,
                codec=codec,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                y0=y0)
        generated = generated.to(torch.float32)
        generated_mel_spec = generated.permute(0, 2, 1)
        return generated_mel_spec

    def fast_block_sample(
        self,
        cond,
        codec,
        ref_mel,
        y0,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
    ):
        return self.cfm_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )


class Qwen2Code2wav(torch.nn.Module):

    def __init__(self,
                 dit_ckpt,
                 bigvgan_ckpt,
                 steps: int = 10,
                 bs_mel: int = 24,
                 odeint_method: str = "euler",
                 odeint_method_relaxed: bool = False,
                 batched_chunk: int = 3,
                 frequency: str = "50hz",
                 device='cpu',
                 with_weight_norm: bool = True):
        super().__init__()
        self.frequency = frequency
        self.code2wav_dit_model = Qwen2Code2wavDit(ckpt=dit_ckpt,
                                                   frequency=frequency,
                                                   device=device)
        self.code2wav_bigvgan_model = Qwen2Code2wavBigvgan(
            ckpt=bigvgan_ckpt,
            frequency=frequency,
            device=device,
            with_weight_norm=with_weight_norm)
        self.device = device

        # odeint method: use ruler for first and last chunk to optimize performance
        self.odeint_method_relaxed = odeint_method_relaxed

        # cfm model: override the odeint method
        self.code2wav_dit_model.cfm_model.odeint_kwargs[
            "method"] = odeint_method

        # dit autocast
        self.code2wav_dit_model.dit.fast_forward = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )(self.code2wav_dit_model.dit.fast_forward)

        self.dit_forward = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_compiled = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_compiled_first_chunk = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_cudagraph_first = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_cudagraph_intermediate = self.code2wav_dit_model.dit.fast_forward
        self.dit_forward_cudagraph_last = self.code2wav_dit_model.dit.fast_forward

        self.torch_compile_first_chunk = False

        self.factor = 2 if frequency == '50hz' else 4
        self.steps = steps
        self.bs_mel = bs_mel
        self.bs_codec = bs_mel // self.factor
        self.past_cache_size = bs_mel * self.factor
        self.future_cache_size = bs_mel * 1
        self.chunk_size = bs_mel * batched_chunk
        self.future_size = 20 if self.frequency == '50hz' else 13
        self.past_size = 20 if self.frequency == '50hz' else 51

        text_embed = self.code2wav_dit_model.dit.text_embed
        if hasattr(text_embed, 'codec_embed'):
            self.codec_embed_size = text_embed.codec_embed.weight.size(0)
        elif hasattr(text_embed, 'text_embed'):
            self.codec_embed_size = text_embed.text_embed.weight.size(0)
        else:
            self.codec_embed_size = -1

    @contextlib.contextmanager
    def relax_odeint_method(self, relax: bool = False):
        if relax and self.odeint_method_relaxed:
            odeint_method = self.code2wav_dit_model.cfm_model.odeint_kwargs[
                "method"]
            self.code2wav_dit_model.cfm_model.odeint_kwargs["method"] = "euler"
        yield
        if relax and self.odeint_method_relaxed:
            self.code2wav_dit_model.cfm_model.odeint_kwargs[
                "method"] = odeint_method

    def enable_torch_compile(self, compile_first_chunk: bool = False):
        self.torch_compile_first_chunk = compile_first_chunk

        self.dit_forward_compiled = torch.compile(
            self.code2wav_dit_model.dit.fast_forward,
            # mode="default",
            mode="reduce-overhead",
            fullgraph=False,
        )
        self.dit_forward_cudagraph_first = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device)
        self.dit_forward_cudagraph_intermediate = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device)
        self.dit_forward_cudagraph_last = CudaGraphRunner(
            self.code2wav_dit_model.dit.fast_forward, self.device)

    @torch.inference_mode()
    def forward(self, cond, ref_mel, codec):
        generated_mel = self.code2wav_dit_model.sample(cond, ref_mel, codec)
        generated_mel = generated_mel.permute(0, 2, 1)
        waveform = self.code2wav_bigvgan_model(generated_mel)
        return waveform

    @torch.inference_mode()
    def process_chunk_dit_batch(
        self,
        cond,
        ref_mel,
        codec,
        y0,
        steps,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.codec_embed_size > 0:
            codec[codec >= self.codec_embed_size] = 0

        self.code2wav_dit_model.dit.fast_forward = self.dit_forward_cudagraph_intermediate
        generated, _ = self.code2wav_dit_model.fast_block_sample(
            cond=cond,
            codec=codec,
            ref_mel=ref_mel,
            y0=y0,
            steps=steps,
            cfg_strength=0.5,
            sway_sampling_coef=-1.0,
        )
        return generated

    @torch.inference_mode()
    def process_chunk_bigvgan_batch(self, mel_batch):
        return self.code2wav_bigvgan_model(mel_batch)

    @torch.inference_mode()
    def process_little_chunk(
        self,
        cond,
        ref_mel,
        codec_all,
        y_all,
        i,
        steps,
        prev_generated: torch.Tensor,
        finished: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask to prevent codec from being out of range (the eos token)
        if self.codec_embed_size > 0:
            codec_all[codec_all >= self.codec_embed_size] = 0

        return None, self.forward(cond, ref_mel, codec_all)

    @torch.inference_mode()
    def process_chunk(
        self,
        cond,
        ref_mel,
        codec_all,
        y_all,
        i,
        steps,
        prev_generated: Union[torch.Tensor, List[torch.Tensor]],
        finished: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        start_index = max(i * self.chunk_size - self.past_cache_size, 0)
        end_index = min((i + 1) * self.chunk_size + self.future_cache_size,
                        codec_all.shape[1] * self.factor)
        y0 = y_all[:, start_index:end_index].reshape(1, -1, 80).contiguous()
        codec = codec_all[:, start_index // self.factor:end_index //
                          self.factor].reshape(1, -1).contiguous()

        # mask to prevent codec from being out of range (the eos token)
        if self.codec_embed_size > 0:
            codec[codec >= self.codec_embed_size] = 0

        # N.B. when using cuda graph ("reduce-overhead" mode), don't compile
        # shape for the first and the last chunk, as it will affect the performance
        # for normal chunks.
        #
        # The reason is not clear yet. The default torch.compile() mode is not affected.
        if i == 0:
            if self.torch_compile_first_chunk:
                self.code2wav_dit_model.dit.fast_forward = self.dit_forward_compiled
            else:
                self.code2wav_dit_model.dit.fast_forward = self.dit_forward_cudagraph_first
        elif finished:
            self.code2wav_dit_model.dit.fast_forward = self.dit_forward_cudagraph_last
        else:
            self.code2wav_dit_model.dit.fast_forward = self.dit_forward_cudagraph_intermediate

        with self.relax_odeint_method(relax=i == 0 or finished):
            generated, _ = self.code2wav_dit_model.fast_block_sample(
                cond=cond,
                codec=codec,
                ref_mel=ref_mel,
                y0=y0,
                steps=steps,
                cfg_strength=0.5,
                sway_sampling_coef=-1.0,
            )

        if self.frequency == "50hz":
            return self.process_chunk_for_50hz(
                i,
                start_index,
                end_index,
                finished,
                prev_generated,
                generated,
            )
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

    def process_chunk_for_50hz(
        self,
        i: int,
        start_index: int,
        end_index: int,
        finished: bool,
        prev_generated: torch.Tensor,
        generated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if start_index == 0:
            generated = generated.to(torch.float32)[:, :self.chunk_size, :]
            mel = generated
        elif finished:
            generated = generated.to(torch.float32)[:,
                                                    self.past_cache_size:, :]
            mel = torch.cat(
                [prev_generated[:, -self.future_size * 2:, :], generated],
                dim=1)
        else:
            generated = generated.to(
                torch.float32)[:,
                               self.past_cache_size:-self.future_cache_size, :]
            mel = torch.cat(
                [prev_generated[:, -self.future_size * 2:, :], generated],
                dim=1)

        audio = self.code2wav_bigvgan_model(mel)
        if i == 0:
            audio_output = audio[:-self.future_size * 240]
        elif finished:
            audio_output = audio[self.future_size * 240:]
        else:
            audio_output = audio[self.future_size * 240:-self.future_size *
                                 240]
        return generated, audio_output
