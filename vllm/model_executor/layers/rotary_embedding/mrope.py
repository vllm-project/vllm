# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Optional, Union

import numpy as np
import torch
from transformers import PretrainedConfig

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .base import RotaryEmbedding
from .common import apply_rotary_emb_dispatch


@triton.jit
def _triton_qwen2vl_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    # instead of (3, bsz, seq_len, head_dim)
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)

    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    # Updated stride calculation for half head_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets,
                       mask=first_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets,
                       mask=first_k_mask,
                       other=0).to(sin_row.dtype)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets,
                       mask=second_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets,
                       mask=second_k_mask,
                       other=0).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # Since cos and sin are now half-size,
    # we use the same cos_row and sin_row for both halves
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2VL mrope kernel.

    Args:
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
    """
    n_row, n_q_head_head_dim = q.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # ensure tensors passed into the kernel are contiguous.
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_qwen2vl_mrope_forward[(n_row, )](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
    )
    return q, k


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style, dtype)

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

        self.use_triton = current_platform.is_cuda_alike()

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MRope forward.

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        if self.use_triton:
            return self.forward_cuda(positions, query, key)
        else:
            return self.forward_native(positions, query, key)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section

            cos = torch.cat([
                m[i]
                for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)
            sin = torch.cat([
                m[i]
                for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin,
                                              self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin,
                                            self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            assert self.mrope_section

            q, k = triton_mrope(
                query,
                key,
                cos,
                sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
            )

            return q.reshape(query_shape), k.reshape(key_shape)

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin,
                                              self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin,
                                            self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    @classmethod
    def get_input_positions(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
        video_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
        second_per_grid_ts: Optional[list[float]],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[list[list[int]], int]:
        """Get mrope input positions and delta value."""

        image_grid_thw = [] if image_grid_thw is None else image_grid_thw
        video_grid_thw = [] if video_grid_thw is None else video_grid_thw
        second_per_grid_ts = [] if second_per_grid_ts is None else \
            second_per_grid_ts

        llm_positions, mrope_position_delta = \
            cls.get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )

        return llm_positions.tolist(), mrope_position_delta

    @classmethod
    def get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        from vllm.transformers_utils.config import thinker_uses_mrope
        if thinker_uses_mrope(hf_config):
            return cls._omni_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        elif hf_config.model_type in ["glm4v", "glm4v_moe"]:
            return cls._glm4v_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        else:
            return cls._vl_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
            )

    @classmethod
    def _glm4v_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value for GLM4V."""

        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        llm_pos_ids_list: list = []

        if not (image_grid_thw is None and video_grid_thw is None):
            if isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = image_grid_thw.tolist()

            input_token_type: list[str] = []
            video_check_flg = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False

                if (token == image_token_id) and (video_check_flg is False):
                    input_token_type.append("image")
                elif (token == image_token_id) and (video_check_flg is True):
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")

            input_type_group: list[tuple[str, int, int]] = []
            for key, group_iter in itertools.groupby(
                    enumerate(input_token_type), lambda x: x[1]):
                group_list = list(group_iter)
                start_index = group_list[0][0]
                end_index = group_list[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            video_frame_num = 1
            mm_data_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                if modality_type == "image":
                    t, h, w = (
                        image_grid_thw[mm_data_idx][0],
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = \
                        t, h // spatial_merge_size, w // spatial_merge_size

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + st_idx)
                    mm_data_idx += 1

                elif modality_type == "video":
                    t, h, w = (
                        video_frame_num,
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = \
                        t, h // spatial_merge_size, w // spatial_merge_size

                    for t_idx in range(llm_grid_t):
                        t_index = torch.tensor(t_idx).view(-1, 1).expand(
                            -1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(
                            1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(
                            1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(
                            torch.stack([t_index, h_index, w_index]) + st_idx)

                    mm_data_idx += 1
                    video_frame_num += 1

                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) +
                        st_idx)
                    video_frame_num = 1

        else:
            text_len = len(input_tokens)
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:seq_len]
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        return llm_positions, mrope_position_delta

    @classmethod
    def _vl_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = (torch.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w) * video_second_per_grid_t *
                       tokens_per_second).long().flatten()

            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @classmethod
    def _omni_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value (Qwen2.5-Omni version).

        Differences from MRotaryEmbedding:
            1. Add audio support (and related `audio_feature_lengths`).
            2. Add `use_audio_in_video` option to read audio from video inputs.
                In this case, audio and vision position ids will be split into
                chunks and interleaved.

        Example:

            (V_i are vision position ids, A_i are audio position ids)

            |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
            |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...
        """

        # TODO(fyabc): refactor and share more code with
        #  _vl_get_input_positions_tensor.

        thinker_config = hf_config.thinker_config
        audio_token_id = thinker_config.audio_token_index
        image_token_id = thinker_config.image_token_index
        video_token_id = thinker_config.video_token_index
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        vision_start_token_id = thinker_config.vision_start_token_id
        vision_end_token_id = thinker_config.vision_end_token_id
        seconds_per_chunk = thinker_config.seconds_per_chunk
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(thinker_config.vision_config,
                                    "tokens_per_second", 25)

        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)

        src_item = input_tokens
        audio_seqlens = audio_feature_lengths
        if not second_per_grid_ts:
            second_per_grid_ts = [1] * video_grid_thw.shape[0]
        audio_idx = 0
        video_idx = 0
        image_idx = 0
        new_src_item: list[int] = []
        llm_pos_ids_list: list[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            if src_item[idx] not in [
                    audio_token_id, video_token_id, image_token_id
            ]:
                if use_audio_in_video and idx > 0:
                    if src_item[idx] == vision_end_token_id and \
                        src_item[idx - 1] == audio_end_token_id:
                        # processing the <|audio_eos|> before <|vision_eos|>
                        start_idx -= 1
                    elif src_item[idx] == audio_start_token_id and \
                        src_item[idx - 1] == vision_start_token_id:
                        # processing the <|audio_bos|> after <|vision_eos|>
                        start_idx -= 1
                new_src_item.append(src_item[idx])
                llm_pos_ids = torch.tensor([start_idx],
                                           dtype=torch.long).expand(3, -1)
                llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == audio_token_id:
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1)
                new_src_item.extend([audio_token_id] * place_num)
                llm_pos_ids = torch.arange(place_num).expand(3, -1) + start_idx
                llm_pos_ids_list.append(llm_pos_ids)
                audio_idx += 1
            elif src_item[idx] == image_token_id:
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = cls._get_llm_pos_ids_for_vision(
                    start_idx, image_idx, spatial_merge_size, t_index, grid_hs,
                    grid_ws)
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = image_grid_thw[image_idx].prod() // (
                    spatial_merge_size**2)
                new_src_item.extend([image_token_id] * vision_seqlen)
                image_idx += 1
            elif src_item[idx] == video_token_id and not use_audio_in_video:
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) *
                           second_per_grid_ts[video_idx] *
                           tokens_per_second).long()
                llm_pos_ids = cls._get_llm_pos_ids_for_vision(
                    start_idx, video_idx, spatial_merge_size, t_index, grid_hs,
                    grid_ws)
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2)
                new_src_item.extend([video_token_id] * vision_seqlen)
                video_idx += 1
            else:
                # read audio from video
                assert audio_seqlens is not None
                audio_seqlen = audio_seqlens[audio_idx]
                vision_seqlen = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2)
                grid_t = video_grid_thw[video_idx][0]
                grid_h = video_grid_thw[video_idx][1]
                grid_w = video_grid_thw[video_idx][2]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
                t_index = (torch.arange(grid_t) *
                           second_per_grid_ts[video_idx] *
                           tokens_per_second).long()
                t_index_split_chunk = cls._split_list_into_ranges(
                    t_index, t_ntoken_per_chunk)
                place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
                pure_audio_len = place_num - 2
                added_audio_len = 0
                audio_llm_pos_ids_list: list[torch.Tensor] = []
                for t_chunk in t_index_split_chunk:
                    vision_ntoken_per_chunk = len(
                        t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
                    new_src_item.extend([video_token_id] *
                                        vision_ntoken_per_chunk)
                    vision_llm_pos_ids_list = cls._get_llm_pos_ids_for_vision(
                        start_idx, video_idx, spatial_merge_size, t_chunk,
                        grid_hs, grid_ws).split(1, dim=1)
                    llm_pos_ids_list.extend(vision_llm_pos_ids_list)
                    new_src_item.extend(
                        min(t_ntoken_per_chunk, pure_audio_len -
                            added_audio_len) * [audio_token_id])
                    audio_start_idx = start_idx if len(
                        audio_llm_pos_ids_list
                    ) == 0 else audio_llm_pos_ids_list[-1][0].item() + 1
                    if min(t_ntoken_per_chunk,
                           pure_audio_len - added_audio_len) > 0:
                        audio_llm_pos_ids_list = (torch.arange(
                            min(t_ntoken_per_chunk, pure_audio_len -
                                added_audio_len)).expand(3, -1) +
                                                  audio_start_idx).split(1,
                                                                         dim=1)
                    else:
                        audio_llm_pos_ids_list = []
                    added_audio_len += min(t_ntoken_per_chunk,
                                           pure_audio_len - added_audio_len)
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                if added_audio_len < pure_audio_len:
                    new_src_item.extend(
                        (pure_audio_len - added_audio_len) * [audio_token_id])
                    audio_llm_pos_ids_list = (
                        torch.arange(pure_audio_len - added_audio_len).expand(
                            3, -1) + llm_pos_ids_list[-1].max() + 1).split(
                                1, dim=1)
                    llm_pos_ids_list.extend(audio_llm_pos_ids_list)
                audio_idx += 1
                video_idx += 1
            # move to the next token
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = torch.cat(llm_pos_ids_list,
                                         dim=1).max() + 1 - len(src_item)
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def _get_llm_pos_ids_for_vision(
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
    ) -> torch.Tensor:
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (torch.arange(llm_grid_h).view(1, -1, 1).expand(
            len(t_index), -1, llm_grid_w).flatten())
        w_index = (torch.arange(llm_grid_w).view(1, 1, -1).expand(
            len(t_index), llm_grid_h, -1).flatten())
        t_index_tensor = torch.Tensor(t_index).to(llm_grid_h.device).view(
            -1, 1).expand(-1, llm_grid_h * llm_grid_w).long().flatten()
        _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    @staticmethod
    def _split_list_into_ranges(lst: torch.Tensor,
                                interval: int) -> list[list[int]]:
        ranges: list[list[int]] = [[]
                                   for _ in range((max(lst) // interval) + 1)]
        for num in lst:
            index = num // interval
            ranges[index].append(num)
        return ranges

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(context_len + mrope_position_delta,
                      seq_len + mrope_position_delta)) for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(out: np.ndarray, out_offset: int,
                                        mrope_position_delta: int,
                                        context_len: int, num_new_tokens: int):

        values = np.arange(mrope_position_delta + context_len,
                           mrope_position_delta + context_len + num_new_tokens,
                           dtype=out.dtype)
        out[:, out_offset:out_offset + num_new_tokens] = values

    @classmethod
    def omni_get_updates_use_audio_in_video(
        cls,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: Union[list[int], torch.Tensor],
        video_second_per_grid_t: float,
    ) -> list[int]:
        """Get video prompt updates when `use_audio_in_video` is True.

        In this case, audio and vision update ids will be split into
        chunks and interleaved (details in `_omni_get_input_positions_tensor`).

        <|video_bos|><|VIDEO|><|video_eos|> =>
        <|video_bos|><|audio_bos|>(... chunks ...)<|audio_eos|><|video_eos|>
        """

        audio_token_id = thinker_config.audio_token_index
        video_token_id = thinker_config.video_token_index
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        seconds_per_chunk = thinker_config.seconds_per_chunk
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(thinker_config.vision_config,
                                    "tokens_per_second", 25)

        grid_t = video_grid_thw[0]
        grid_h = video_grid_thw[1]
        grid_w = video_grid_thw[2]
        t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
        t_index = (torch.arange(grid_t) * video_second_per_grid_t *
                   tokens_per_second).long()
        t_index_split_chunk = cls._split_list_into_ranges(
            t_index, t_ntoken_per_chunk)

        updates = [audio_start_token_id]
        added_audio_len = 0
        for t_chunk in t_index_split_chunk:
            vision_ntoken_per_chunk = len(t_chunk) * grid_h * grid_w // (
                spatial_merge_size**2)
            updates.extend([video_token_id] * vision_ntoken_per_chunk)

            audio_chunk_size = min(t_ntoken_per_chunk,
                                   audio_len - added_audio_len)
            updates.extend(audio_chunk_size * [audio_token_id])
            added_audio_len += audio_chunk_size
        if added_audio_len < audio_len:
            updates.extend((audio_len - added_audio_len) * [audio_token_id])
        updates.extend([audio_end_token_id])

        return updates
