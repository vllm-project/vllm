# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests per-shard FP8 scaling at real model dimensions (#40934, #38527, #36350)."""

import torch
import unittest

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_INTRINSIC_ERROR = 0.03
RESOLUTION_FLOOR = 200


def scaled_fp8_quant(tensor, scale=None):
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = amax / FP8_E4M3_MAX
    q = (tensor / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = q.to(torch.float8_e4m3fn)
    return q, scale


def fp8_dequant(q, scale):
    return q.to(torch.float32) * scale


def single_scale_quant(weight):
    q, s = scaled_fp8_quant(weight)
    return fp8_dequant(q, s)


def per_shard_quant(weight, shard_widths):
    out = torch.empty_like(weight, dtype=torch.float32)
    offset = 0
    for w in shard_widths:
        shard = weight[offset:offset + w]
        q, s = scaled_fp8_quant(shard)
        out[offset:offset + w] = fp8_dequant(q, s)
        offset += w
    return out


def shard_metrics(fused, shard_widths, quant_fn, quant_args=()):
    if quant_args:
        recon = quant_fn(fused, *quant_args)
    else:
        recon = quant_fn(fused)
    q_full, _ = scaled_fp8_quant(fused)

    errors = []
    distinct = []
    offset = 0
    for w in shard_widths:
        s = slice(offset, offset + w)
        orig = fused[s]
        err = (recon[s] - orig).abs() / (orig.abs() + 1e-12)
        errors.append(err.mean().item())
        distinct.append(q_full[s].to(torch.float32).unique().numel())
        offset += w
    return errors, distinct


# Qwen3.6-27B: Q=512, K=512, V=6144, Z=6144, input=5120
def make_qwen36_27b_in_proj_qkvz(mag_q=1.0, mag_z=1.0, seed=42):
    torch.manual_seed(seed)
    widths = [512, 512, 6144, 6144]
    input_dim = 5120
    shards = [
        torch.randn(512, input_dim) * mag_q,
        torch.randn(512, input_dim) * mag_q,
        torch.randn(6144, input_dim) * mag_z,
        torch.randn(6144, input_dim) * mag_z,
    ]
    return torch.cat(shards, dim=0), widths, input_dim


# Qwen3.5-4B: Q=2048, K=2048, V=4096, Z=4096, input=2560
def make_qwen35_4b_in_proj_qkvz(mag_q=1.0, mag_z=1.0, seed=42):
    torch.manual_seed(seed)
    widths = [2048, 2048, 4096, 4096]
    input_dim = 2560
    shards = [
        torch.randn(2048, input_dim) * mag_q,
        torch.randn(2048, input_dim) * mag_q,
        torch.randn(4096, input_dim) * mag_z,
        torch.randn(4096, input_dim) * mag_z,
    ]
    return torch.cat(shards, dim=0), widths, input_dim


# #38527: Qwen3.5-35B outputs "!!!!!!!!!!" (same model dimensions, offline FP8)

class TestIssue38527_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_z_gate_resolution_at_20x_gap(self):
        """Z gate loses FP8 resolution when Q and K are 20x larger."""
        fused, widths, _ = make_qwen36_27b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        _, distinct = shard_metrics(fused, widths, single_scale_quant)
        z_distinct = distinct[3]
        self.assertGreater(z_distinct, RESOLUTION_FLOOR,
            f"Z gate has {z_distinct} FP8 levels (need {RESOLUTION_FLOOR})")

    @unittest.expectedFailure
    def test_z_gate_resolution_at_100x_gap(self):
        fused, widths, _ = make_qwen36_27b_in_proj_qkvz(mag_q=1.0, mag_z=0.01)
        _, distinct = shard_metrics(fused, widths, single_scale_quant)
        z_distinct = distinct[3]
        self.assertGreater(z_distinct, RESOLUTION_FLOOR,
            f"Z gate has only {z_distinct} FP8 levels")


class TestIssue38527_PerShard_PASSES(unittest.TestCase):

    def test_z_gate_resolution_at_20x_gap(self):
        fused, widths, _ = make_qwen36_27b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        q_z, _ = scaled_fp8_quant(fused[widths[0]+widths[1]+widths[2]:])
        z_distinct = q_z.to(torch.float32).unique().numel()
        self.assertGreater(z_distinct, RESOLUTION_FLOOR,
            f"Z gate has {z_distinct} FP8 levels")

    def test_z_gate_resolution_at_100x_gap(self):
        fused, widths, _ = make_qwen36_27b_in_proj_qkvz(mag_q=1.0, mag_z=0.01)
        q_z, _ = scaled_fp8_quant(fused[widths[0]+widths[1]+widths[2]:])
        z_distinct = q_z.to(torch.float32).unique().numel()
        self.assertGreater(z_distinct, RESOLUTION_FLOOR,
            f"Z gate has only {z_distinct} FP8 levels")


# #40934: --quantization fp8 crashes on Qwen3.5 hybrid (GDN)

class TestIssue40934_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_qkvz_resolution_preserved(self):
        """V and Z shards lose resolution when Q and K are 20x larger."""
        fused, widths, _ = make_qwen35_4b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        _, distinct = shard_metrics(fused, widths, single_scale_quant)
        for i, (d, name) in enumerate(zip(distinct, ['Q', 'K', 'V', 'Z'])):
            self.assertGreater(d, RESOLUTION_FLOOR,
                f"{name} shard has {d} FP8 levels")


class TestIssue40934_PerShard_PASSES(unittest.TestCase):

    def test_qkvz_resolution_preserved(self):
        fused, widths, _ = make_qwen35_4b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        offset = 0
        for w, name in zip(widths, ['Q', 'K', 'V', 'Z']):
            shard = fused[offset:offset + w]
            q, _ = scaled_fp8_quant(shard)
            d = q.to(torch.float32).unique().numel()
            self.assertGreater(d, RESOLUTION_FLOOR,
                f"{name} shard has {d} FP8 levels")
            offset += w


# #36350: Qwen3.5 4B fails on Intel XPU, FP8 precision component

class TestIssue36350_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_4b_vz_resolution_preserved(self):
        """V and Z shards lose resolution at 20x magnitude gap."""
        fused, widths, _ = make_qwen35_4b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        _, distinct = shard_metrics(fused, widths, single_scale_quant)
        v_distinct, z_distinct = distinct[2], distinct[3]
        self.assertGreater(min(v_distinct, z_distinct), RESOLUTION_FLOOR,
            f"V={v_distinct} Z={z_distinct} FP8 levels")


class TestIssue36350_PerShard_PASSES(unittest.TestCase):

    def test_4b_vz_resolution_preserved(self):
        fused, widths, _ = make_qwen35_4b_in_proj_qkvz(mag_q=1.0, mag_z=0.05)
        offset = widths[0] + widths[1]
        for w, name in zip(widths[2:], ['V', 'Z']):
            shard = fused[offset:offset + w]
            q, _ = scaled_fp8_quant(shard)
            d = q.to(torch.float32).unique().numel()
            self.assertGreater(d, RESOLUTION_FLOOR,
                f"{name} shard has {d} FP8 levels")
            offset += w


# SwiGLU gate_up_proj at Qwen3.6-27B dimensions (17408+17408 x 5120)

class TestSwiGLU_27B_SingleScale_FAILS(unittest.TestCase):

    @unittest.expectedFailure
    def test_gate_precision_at_40x_gap(self):
        """Gate shard resolution collapses when up weights are 40x larger."""
        torch.manual_seed(42)
        gate = torch.randn(17408, 5120) * 0.05
        up = torch.randn(17408, 5120) * 2.0
        fused = torch.cat([gate, up], dim=0)
        q, _ = scaled_fp8_quant(fused)
        gate_distinct = q[:17408].to(torch.float32).unique().numel()
        self.assertGreater(gate_distinct, RESOLUTION_FLOOR,
            f"Gate has {gate_distinct} FP8 levels")


class TestSwiGLU_27B_PerShard_PASSES(unittest.TestCase):

    def test_gate_precision_at_40x_gap(self):
        torch.manual_seed(42)
        gate = torch.randn(17408, 5120) * 0.05
        q, _ = scaled_fp8_quant(gate)
        gate_distinct = q.to(torch.float32).unique().numel()
        self.assertGreater(gate_distinct, RESOLUTION_FLOOR,
            f"Gate has {gate_distinct} FP8 levels")
