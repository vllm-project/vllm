from typing import List, Optional, Tuple, Union

import numba
import numpy as np
import torch
from transformers import PretrainedConfig

def mrope_get_input_positions_and_delta(
    input_tokens: Union[list[int], np.ndarray],
    hf_config: PretrainedConfig,
    image_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
    video_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
    second_per_grid_ts: Optional[list[float]],
    context_len: int = 0,
    seq_len: Optional[int] = None,
    audio_feature_lengths: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    use_numba: bool = True,
) -> tuple[torch.Tensor, int]:
    from vllm.transformers_utils.config import thinker_uses_mrope
    is_omni = thinker_uses_mrope(hf_config)

    if (image_grid_thw is None or len(image_grid_thw) == 0) and \
        (video_grid_thw is None or len(video_grid_thw) == 0) and \
        (audio_feature_lengths is None or len(audio_feature_lengths) == 0):
        # text-only prompt
        input_positions = torch.arange(len(input_tokens)).expand(3, -1)
        mrope_position_delta = 0
    else:
        if use_numba:
            input_tokens = np.asarray(input_tokens, dtype=np.int64)

            if image_grid_thw is None or len(image_grid_thw) == 0:
                image_grid_thw = np.empty((0, 3), dtype=np.int64)
            elif isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = image_grid_thw.numpy()
            else:
                image_grid_thw = np.array(image_grid_thw, dtype=np.int64)

            if video_grid_thw is None or len(video_grid_thw) == 0:
                video_grid_thw = np.empty((0, 3), dtype=np.int64)
            elif isinstance(video_grid_thw, torch.Tensor):
                video_grid_thw = video_grid_thw.numpy()
            else:
                video_grid_thw = np.array(video_grid_thw, dtype=np.int64)

            if second_per_grid_ts is None:
                second_per_grid_ts = np.empty((0, ), dtype=np.float64)
            else:
                second_per_grid_ts = np.array(second_per_grid_ts, dtype=np.float64)
            
            if len(second_per_grid_ts) < len(video_grid_thw):
                raise ValueError("second_per_grid_ts is shorter than video_grid_thw")
                
            if is_omni:
                if audio_feature_lengths is None:
                    audio_feature_lengths = np.empty((0, ), dtype=np.int64)
                else:
                    audio_feature_lengths = np.array(audio_feature_lengths, dtype=np.int64)

                thinker_config = hf_config.thinker_config
                input_positions, mrope_position_delta = omni_get_input_positions_numba(
                    input_tokens=input_tokens,
                    image_token_id=int(thinker_config.image_token_index),
                    video_token_id=int(thinker_config.video_token_index),
                    audio_token_id=int(thinker_config.audio_token_index),
                    vision_start_token_id=int(thinker_config.vision_start_token_id),
                    vision_end_token_id=int(thinker_config.vision_end_token_id),
                    audio_start_token_id=int(thinker_config.audio_start_token_id),
                    audio_end_token_id=int(thinker_config.audio_end_token_id),
                    spatial_merge_size=int(thinker_config.vision_config.spatial_merge_size),
                    tokens_per_second=float(thinker_config.vision_config.tokens_per_second),
                    seconds_per_chunk=float(thinker_config.seconds_per_chunk),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )
            else:
                input_positions, mrope_position_delta = vl_get_input_positions_numba(
                    input_tokens=input_tokens,
                    image_token_id=int(hf_config.image_token_id),
                    video_token_id=int(hf_config.video_token_id),
                    spatial_merge_size=int(hf_config.vision_config.spatial_merge_size),
                    tokens_per_second=float(getattr(hf_config.vision_config, "tokens_per_second", 1.0)),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                )
            
            input_positions = torch.from_numpy(input_positions)
            if context_len != 0 or seq_len is not None:
                input_positions = input_positions[:, context_len:seq_len]
        else:
            if isinstance(input_tokens, np.ndarray):
                input_tokens = input_tokens.tolist()

            if is_omni:
                input_positions, mrope_position_delta = omni_get_input_positions_torch(
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
            else:
                input_positions, mrope_position_delta = vl_get_input_positions_torch(
                    input_tokens=input_tokens,
                    hf_config=hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    context_len=context_len,
                    seq_len=seq_len,
                )

    return input_positions, mrope_position_delta

def mrope_get_next_input_positions(
    mrope_position_delta: int,
    context_len: int,
    seq_len: int,
) -> List[List[int]]:
    return [
        list(
            range(context_len + mrope_position_delta,
                    seq_len + mrope_position_delta)) for _ in range(3)
    ]

def mrope_get_next_input_positions_tensor(
    mrope_position_delta: int,
    context_len: int,
    seq_len: int,
) -> torch.Tensor:
    return torch.arange(
        mrope_position_delta + context_len,
        mrope_position_delta + seq_len,
    ).expand(3, -1)

def vl_get_input_positions_torch(
    input_tokens: list[int],
    hf_config: PretrainedConfig,
    image_grid_thw: Union[list[list[int]], torch.Tensor],
    video_grid_thw: Union[list[list[int]], torch.Tensor],
    second_per_grid_ts: list[float],
    context_len: int = 0,
    seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Get mrope input positions and delta value for Qwen2/2.5-VL

    This is the original PyTorch implementation
    """

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
    if context_len != 0 or seq_len is not None:
        llm_positions = llm_positions[:, context_len:seq_len]

    return llm_positions, mrope_position_delta

@numba.jit(nopython=True)
def vl_get_input_positions_numba(
    input_tokens: np.ndarray,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float,
    image_grid_thw: np.ndarray,
    video_grid_thw: np.ndarray,
    second_per_grid_ts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Get mrope input positions and delta value for Qwen2/2.5-VL

    This is the optimized numba implementation
    """

    mrope_pos = np.empty((3, len(input_tokens)), dtype=np.int64)

    cur_t = -1

    cur_image_idx = -1
    cur_video_idx = -1

    i = 0
    while i < len(input_tokens):
        token_id = input_tokens[i]
        if token_id == image_token_id:
            cur_image_idx += 1
            assert cur_image_idx < len(image_grid_thw), "mrope image_grid_thw index out of range"

            i, cur_t = _emit_image_positions(
                mrope_pos,
                i=i,
                image_grid_thw=image_grid_thw[cur_image_idx],
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
            )
        elif token_id == video_token_id:
            cur_video_idx += 1
            assert cur_video_idx < len(video_grid_thw), "mrope video_grid_thw index out of range"

            i, cur_t = _emit_video_positions(
                mrope_pos,
                i=i,
                video_grid_thw=video_grid_thw[cur_video_idx],
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
                second_per_grid_t=second_per_grid_ts[cur_video_idx],
            )
        else:
            cur_t += 1
            i = _emit_standalone_token(
                mrope_pos,
                i=i,
                t=cur_t,
            )

    mrope_position_delta = cur_t + 1 - len(input_tokens)
    return mrope_pos, mrope_position_delta

@numba.jit(nopython=True, inline="always")
def _emit_frame_positions(
    mrope_pos: np.ndarray,
    i: int,
    num_h: int,
    num_w: int,
    cur_t: int,
    start_hw: int,
) -> int:
    for h in range(num_h):
        for w in range(num_w):
            assert i < mrope_pos.shape[1], "incomplete mrope positions"
            mrope_pos[0, i] = cur_t
            mrope_pos[1, i] = start_hw + h
            mrope_pos[2, i] = start_hw + w
            i += 1
            
    return i

@numba.jit(nopython=True)
def _emit_image_positions(
    mrope_pos: np.ndarray,
    i: int,
    image_grid_thw: np.ndarray,
    start_t: int,
    spatial_merge_size: int,
) -> tuple[int, int]:
    num_h = image_grid_thw[1] // spatial_merge_size
    num_w = image_grid_thw[2] // spatial_merge_size
    for t in range(start_t, start_t + image_grid_thw[0]):
        i = _emit_frame_positions(
            mrope_pos,
            i=i,
            num_h=num_h,
            num_w=num_w,
            cur_t=t,
            start_hw=start_t,
        )

    cur_t = start_t + max(image_grid_thw[0], num_h, num_w) - 1
    return i, cur_t

@numba.jit(nopython=True)
def _emit_video_positions(
    mrope_pos: np.ndarray,
    i: int,
    video_grid_thw: np.ndarray,
    start_t: int,
    spatial_merge_size: int,
    tokens_per_second: int,
    second_per_grid_t: float,
) -> tuple[int, int]:
    num_h = video_grid_thw[1] // spatial_merge_size
    num_w = video_grid_thw[2] // spatial_merge_size

    tokens_per_grid_t = tokens_per_second * second_per_grid_t

    for t in range(video_grid_thw[0]):
        i = _emit_frame_positions(
            mrope_pos,
            i=i,
            num_h=num_h,
            num_w=num_w,
            cur_t=start_t + int(t * tokens_per_grid_t),
            start_hw=start_t,
        )
    
    cur_t = start_t + max(video_grid_thw[0], num_h, num_w) - 1
    return i, cur_t

@numba.jit(nopython=True)
def _emit_video_with_audio(
    mrope_pos: np.ndarray,
    i: int,
    video_grid_thw: np.ndarray,
    start_t: int,
    spatial_merge_size: int,
    tokens_per_second: int,
    second_per_grid_t: float,
    seconds_per_chunk: float,
    audio_feature_length: int,
):
    video_num_h = video_grid_thw[1] // spatial_merge_size
    video_num_w = video_grid_thw[2] // spatial_merge_size

    tokens_per_grid_t = tokens_per_second * second_per_grid_t

    audio_token_num = _calc_audio_token_num(audio_feature_length)
    added_audio_token_num = 0

    t_ntoken_per_chunk = int(seconds_per_chunk * second_per_grid_t)
    next_chunk_t = start_t + t_ntoken_per_chunk
    
    for t in range(video_grid_thw[0]):
        video_t = start_t + int(t * tokens_per_grid_t)

        # audio tokens
        if video_t >= next_chunk_t:
            next_chunk_t += t_ntoken_per_chunk
            if added_audio_token_num < audio_token_num:
                chunked_audio_token_num = min(
                    t_ntoken_per_chunk,
                    audio_token_num - added_audio_token_num)
                i, _ = _emit_standalone_tokens(
                    mrope_pos,
                    i=i,
                    start_t=start_t + added_audio_token_num,
                    num_tokens=chunked_audio_token_num,
                )
                added_audio_token_num += chunked_audio_token_num
            
        # video tokens
        i = _emit_frame_positions(
            mrope_pos,
            i=i,
            num_h=video_num_h,
            num_w=video_num_w,
            cur_t=video_t,
            start_hw=start_t,
        )
    
    # remaining audio tokens
    if added_audio_token_num < audio_token_num:
        i, _ = _emit_standalone_tokens(
            mrope_pos,
            i=i,
            start_t=start_t + added_audio_token_num,
            num_tokens=audio_token_num - added_audio_token_num,
        )
    
    cur_t = max(mrope_pos[0, i - 1], mrope_pos[1, i - 1], mrope_pos[2, i - 1])
    return i, cur_t

@numba.jit(nopython=True, inline="always")
def _emit_standalone_token(
    mrope_pos: np.ndarray,
    i: int,
    t: int,
) -> int:
    mrope_pos[0, i] = t
    mrope_pos[1, i] = t
    mrope_pos[2, i] = t
    return i + 1

@numba.jit(nopython=True, inline="always")
def _emit_standalone_tokens(
    mrope_pos: np.ndarray,
    i: int,
    start_t: int,
    num_tokens: int,
) -> tuple[int, int]:
    for t in range(start_t, start_t + num_tokens):
        assert i < mrope_pos.shape[1], "incomplete mrope positions"
        i = _emit_standalone_token(
            mrope_pos,
            i=i,
            t=t,
        )

    return i, start_t + num_tokens - 1

@numba.jit(nopython=True, inline="always")
def _calc_audio_token_num(audio_feature_length: int):
    return (((audio_feature_length - 1) // 2 + 1 - 2) // 2 + 1)

def omni_get_input_positions_torch(
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
            llm_pos_ids = _get_llm_pos_ids_for_vision(
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
            llm_pos_ids = _get_llm_pos_ids_for_vision(
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
            t_index_split_chunk = _split_list_into_ranges(
                t_index, t_ntoken_per_chunk)
            place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
            pure_audio_len = place_num - 2
            added_audio_len = 0
            audio_llm_pos_ids_list: List[torch.Tensor] = []
            for t_chunk in t_index_split_chunk:
                vision_ntoken_per_chunk = len(
                    t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
                new_src_item.extend([video_token_id] *
                                    vision_ntoken_per_chunk)
                vision_llm_pos_ids_list = _get_llm_pos_ids_for_vision(
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
                                                audio_start_idx).split(
                                                    1, dim=1)
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

@numba.jit(nopython=True)
def omni_get_input_positions_numba(
    input_tokens: np.ndarray,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    audio_start_token_id: int,
    audio_end_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float,
    seconds_per_chunk: float,
    image_grid_thw: np.ndarray,
    video_grid_thw: np.ndarray,
    second_per_grid_ts: np.ndarray,
    audio_feature_lengths: np.ndarray,
    use_audio_in_video: bool,
) -> tuple[np.ndarray, int]:
    mrope_pos = np.empty((3, len(input_tokens)), dtype=np.int64)

    cur_t = -1

    cur_image_idx = -1
    cur_video_idx = -1
    cur_audio_idx = -1

    i = 0
    while i < len(input_tokens):
        token_id = input_tokens[i]
        if token_id == image_token_id:
            cur_image_idx += 1
            assert cur_image_idx < len(image_grid_thw), "mrope image_grid_thw index out of range"

            i, cur_t = _emit_image_positions(
                mrope_pos,
                image_grid_thw=image_grid_thw[cur_image_idx],
                i=i,
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
            )
        elif token_id == video_token_id and use_audio_in_video:
            # audio and vision position ids splitted into chunks and interleaved.
            # 
            # |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
            # |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...

            cur_video_idx += 1
            assert cur_video_idx < len(video_grid_thw), "mrope video_grid_thw index out of range"

            cur_audio_idx += 1
            assert cur_audio_idx < len(audio_feature_lengths), "mrope audio_feature_lengths index out of range"
            
            i, cur_t = _emit_video_with_audio(
                mrope_pos,
                i=i,
                video_grid_thw=video_grid_thw[cur_video_idx],
                start_t = cur_t + 1,
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
                second_per_grid_t=second_per_grid_ts[cur_video_idx],
                seconds_per_chunk=seconds_per_chunk,
                audio_feature_length=audio_feature_lengths[cur_audio_idx],
            )
        elif token_id == video_token_id:
            cur_video_idx += 1
            assert cur_video_idx < len(video_grid_thw), "mrope video_grid_thw index out of range"

            i, cur_t = _emit_video_positions(
                mrope_pos,
                i=i,
                video_grid_thw=video_grid_thw[cur_video_idx],
                start_t=cur_t + 1,
                spatial_merge_size=spatial_merge_size,
                tokens_per_second=tokens_per_second,
                second_per_grid_t=second_per_grid_ts[cur_video_idx],
            )
        elif token_id == audio_token_id:
            cur_audio_idx += 1
            assert cur_audio_idx < len(audio_feature_lengths), "mrope audio_feature_lengths index out of range"

            i, cur_t = _emit_standalone_tokens(
                mrope_pos,
                i=i,
                start_t=cur_t + 1,
                num_tokens=_calc_audio_token_num(audio_feature_lengths[cur_audio_idx]),
            )
        elif token_id == audio_start_token_id \
            and use_audio_in_video \
            and i > 0 \
            and input_tokens[i - 1] == vision_start_token_id:
            # handling the <|audio_bos|> after <|vision_bos|>
            i = _emit_standalone_token(
                mrope_pos,
                i=i,
                t=cur_t,
            )
        elif token_id == vision_end_token_id \
            and use_audio_in_video \
            and i > 0 \
            and input_tokens[i - 1] == audio_end_token_id:
            # handling the <|vision_eos|> after <|audio_eos|>
            i = _emit_standalone_token(
                mrope_pos,
                i=i,
                t=cur_t,
            )
        else:
            cur_t += 1
            i = _emit_standalone_token(
                mrope_pos,
                i=i,
                t=cur_t,
            )

    mrope_position_delta = cur_t + 1 - len(input_tokens)
    return mrope_pos, mrope_position_delta

def omni_get_updates_use_audio_in_video(
    thinker_config: PretrainedConfig,
    audio_len: int,
    video_grid_thw: Union[List[int], torch.Tensor],
    video_second_per_grid_t: float,
) -> List[int]:
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
    t_index_split_chunk = _split_list_into_ranges(
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

def _get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: List[int],
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

def _split_list_into_ranges(lst: torch.Tensor,
                            interval: int) -> List[List[int]]:
    ranges: List[List[int]] = [[]
                                for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges
