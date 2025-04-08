"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Callable
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from vllm.transformers_utils.qwen2_code2wav_dit.model.modules import MelSpec
from vllm.transformers_utils.qwen2_code2wav_dit.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)


class CodecCFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        upsample_rate: int = 2,
    ):
        super().__init__()

       

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # self.spk_proj = nn.Linear(192, 80)

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        self.upsample_rate = upsample_rate
    def logit_normal_sample(self, batch, dtype, device, m=0.0, s=1.0 ):

        u = torch.randn((batch,),dtype=dtype,device=device) * s + m  # u ~ N(m, s^2)
        samples = torch.sigmoid(u)  # logistic(u) = 1 / (1 + exp(-u))
        
        return samples
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        codec: int["b nc dc"],
        ref_mel: float["b n d"],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        y0: float["b n d"] | None = None,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        max_duration = codec.shape[1] * self.transformer.repeats
        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()
            ref_mel = ref_mel.half()
            if y0 is not None:
                y0 = y0.half()
        # raw wave
        cond = cond.unsqueeze(1).repeat(1,max_duration, 1)
        # if cond.ndim == 2:
        #     cond = self.mel_spec(cond)
        #     cond = cond.permute(0, 2, 1)
        #     assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *ref_mel.shape[:2], codec.device
        assert batch == 1, "only support batch size = 1 currently"
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # cond_mask = lens_to_mask(lens)
        # ref_mel = F.pad(ref_mel, (0, 0, 0, cond_seq_len), value=0.0)
        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        # cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, ref_mel, torch.zeros_like(ref_mel)
        # )  # allow direct control (cut cond audio) with lens passed in

        mask = None

        # test for no ref audio
        if no_ref_audio:
            ref_mel = torch.zeros_like(ref_mel)
            cond = torch.zeros_like(cond)
        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, ref_mel, torch.zeros_like(ref_mel))

            # predict flow
            # print(x.dtype,cond.dtype,ref_mel.dtype)
            pred = self.transformer(
                x=x, spk=cond,cond=ref_mel, text=codec, time=t, mask=mask, drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, spk=cond,cond=ref_mel , text=codec, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        if y0 is None:
            y0 = torch.randn([1, max_duration, self.num_channels], device=self.device, dtype=cond.dtype)
        
        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, cond, out)
        return out, trajectory
    
    @torch.no_grad()
    def block_sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        codec: int["b nc dc"],
        ref_mel: float["b n d"],  # noqa: F722
        y0: float["b n d"],
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
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

        # if cond.ndim == 2:
        #     cond = self.mel_spec(cond)
        #     cond = cond.permute(0, 2, 1)
        #     assert cond.shape[-1] == self.num_channels
        cond = cond.unsqueeze(1).repeat(1,max_duration, 1)
        batch, cond_seq_len, device = *ref_mel.shape[:2], cond.device
        assert batch == 1, "only support batch size = 1 currently"
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # cond_mask = lens_to_mask(lens)
        # ref_mel = F.pad(ref_mel, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        # cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, ref_mel, torch.zeros_like(ref_mel)
        # )  # allow direct control (cut cond audio) with lens passed in
        # ref_mel = F.pad(ref_mel, (0, 0, 0, cond_seq_len), value=0.0)
        mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, ref_mel, torch.zeros_like(ref_mel))

            # predict flow
            # import pdb;pdb.set_trace()
            pred = self.transformer(
                x=x, cond=ref_mel, spk=cond, text=codec, time=t, mask=mask, drop_audio_cond=False, drop_text=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=ref_mel, spk=cond, text=codec, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # y0 = torch.randn([1, max_duration, self.num_channels], device=self.device, dtype=step_cond.dtype)
        
        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=ref_mel.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, ref_mel, out)
        return out, trajectory

    @torch.no_grad()
    def fast_block_sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        codec: int["b nc dc"],
        ref_mel: float["b n d"],  # noqa: F722
        y0: float["b n d"],
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
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

        # if cond.ndim == 2:
        #     cond = self.mel_spec(cond)
        #     cond = cond.permute(0, 2, 1)
        #     assert cond.shape[-1] == self.num_channels
        cond = cond.unsqueeze(1).repeat(1,max_duration, 1)
        batch, cond_seq_len, device = *ref_mel.shape[:2], cond.device
        assert batch == 1, "only support batch size = 1 currently"
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # cond_mask = lens_to_mask(lens)
        # ref_mel = F.pad(ref_mel, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        # cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        # cond_mask = cond_mask.unsqueeze(-1)
        # step_cond = torch.where(
        #     cond_mask, ref_mel, torch.zeros_like(ref_mel)
        # )  # allow direct control (cut cond audio) with lens passed in
        # ref_mel = F.pad(ref_mel, (0, 0, 0, cond_seq_len), value=0.0)
        mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, ref_mel, torch.zeros_like(ref_mel))

            # predict flow
            # print(x.dtype,cond.dtype,ref_mel.dtype)
            out_put = self.transformer.fast_forward(
                x=x,
                text=codec,
                spk=cond,
                cond=ref_mel,
                time=t,
                mask=mask,
            )
            pred,null_pred = torch.chunk(out_put,2,dim=0)
            # pred = self.transformer(
            #     x=x, spk=cond,cond=ref_mel, text=codec, time=t, mask=mask, drop_audio_cond=False, drop_text=False
            # )
            # if cfg_strength < 1e-5:
            #     return pred

            # null_pred = self.transformer(
            #     x=x, spk=cond,cond=ref_mel , text=codec, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            # )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # y0 = torch.randn([1, max_duration, self.num_channels], device=self.device, dtype=step_cond.dtype)
        
        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=ref_mel.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        # out = torch.where(cond_mask, ref_mel, out)
        return out, trajectory
        
    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        codec: int["b nc dc"],
        lens: int["b"] | None = None,  # noqa: F821
        spk: int["b nc"] | None = None,
        ref_mel: float["b n d"] | None = None,
        noise_scheduler: str | None = None,
        use_log_norm: bool = True,
    ):

        batch, seq_len, dtype, device, sigma = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # # get a random span to mask out for training conditionally
        # frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        # rand_span_mask = mask_from_frac_lengths(lens, 0.2)

        # if exists(mask):
        #     rand_span_mask &= mask

        # mel is x1
        x1 = inp
        # cond = self.spk_proj(cond)
        spk = spk.unsqueeze(1).repeat(1, inp.size(1), 1)
        # x0 is gaussian noise
        x0 = torch.randn_like(x1)
        # cond = torch.zeros_like(x1)
        # cond_mask  = torch.zeros_like(cond,dtype=torch.bool)
        # for i,j in enumerate(lens):
        #     if random.random() < 0.6:
        #         continue
        #     index = random.randint(0,int(0.6*j))
        #     length = random.randint(0,int(0.3*j))
        #     cond[i,index:index+length,:] = x1[i,index:index+length,:]
        #     cond_mask[i,index:index+length,:] = True
        
        # import pdb;pdb.set_trace()
        # time step
        if use_log_norm:
            time = self.logit_normal_sample(batch, dtype=dtype, device=self.device)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        
        # transformer and cfg training with a drop rate
        drop_audio_cond = random.random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random.random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=phi, cond=ref_mel,spk=spk, text=codec, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        # mask = cond_mask&mask.unsqueeze(-1)
        loss = loss[mask]

        return loss.mean(), ref_mel, pred
