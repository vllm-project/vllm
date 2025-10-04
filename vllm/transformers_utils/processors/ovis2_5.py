# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cached_property
from typing import Optional, Union

import numpy as np
import PIL
import torch
from transformers import AutoProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (ProcessingKwargs, ProcessorMixin,
                                           Unpack)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.mediapipe import MediaPipe
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import imgtype as it
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import (
    media_ext_reader_op_impl, media_ext_reader_op_tensor_info
)
from habana_frameworks.mediapipe.plugins.iterator_pytorch import (
    MediaGenericPytorchIterator
)
# Handle MediaPipe pipe_manager destructor
from habana_frameworks.mediapipe.backend.cal import (
    pipe_manager, cpp_pipe_manager_list
)

from queue import Queue
import os
import io
import time
import atexit
from dataclasses import dataclass

from vllm.logger import init_logger
logger = init_logger(__name__)


def _patched_close(self):
    """Patched close method that handles None cpp_pipe_manager_list during shutdown"""
    try:
        # Check if cpp_pipe_manager_list exists and is not None
        if cpp_pipe_manager_list is not None and self._pm_ in cpp_pipe_manager_list:
            cpp_pipe_manager_list.remove(self._pm_)
    except (TypeError, AttributeError):
        # Handle case where cpp_pipe_manager_list is None or not iterable
        pass

    # Clean up the pipe manager
    if self._pm_ is not None:
        self._pm_.close()
        self._pm_ = None

pipe_manager.close = _patched_close

# Queue shared between external reader and mediapipe call
shared_q = Queue()


class MediaPytorchIterator(MediaGenericPytorchIterator):
    def __init__(self, mediapipe):
        super().__init__(mediapipe=mediapipe, device="hpu", fw_type="PYT_FW")


class external_reader(media_ext_reader_op_impl):
    def __init__(self, params, fw_params):
        self.batch_size = fw_params.batch_size
        self.max_file = ""
        self.num_batches = 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        img_list = shared_q.get()
        for i in range(len(img_list)):
            # NOTE: this padding is needed because of HW alignmnet requirment
            rem = len(img_list[i]) % 64
            pad = (64 - rem) % 64
            if pad:
                img_list[i] = np.pad(img_list[i],
                                    (0, pad),
                                    'constant')
        return img_list

    def get_media_output_type(self):
        return ro.BUFFER_LIST

    def get_largest_file(self):
        return self.max_file

    def gen_output_info(self):
        out_info = []
        o = media_ext_reader_op_tensor_info(
            dt.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        return out_info


class hpuMediaPipe(MediaPipe):
    def __init__(self, device, queue_depth, batch_size,
                 num_threads, op_device,
                 img_height, img_width):
        super(
            hpuMediaPipe,
            self).__init__(
            device,
            queue_depth,
            batch_size,
            num_threads,
            self.__class__.__name__)

        mediapipe_seed = int(time.time_ns() % (2**31 - 1))

        self.input = fn.MediaExtReaderOp(impl=external_reader,
                                         num_outputs=1,
                                         seed=mediapipe_seed,
                                         device=op_device)
        self.decode = fn.ImageDecoder(
            device="hpu", output_format=it.RGB_I, resize=[img_width, img_height])

        self.mean_node = fn.MediaConst(
            data=np.array([127.5, 127.5, 127.5], dtype=dt.FLOAT32),
            shape=[1, 1, 3],
            dtype=dt.FLOAT32
        )

        self.std_node = fn.MediaConst(
            data=np.array([1/127.5, 1/127.5, 1/127.5], dtype=dt.FLOAT32),
            shape=[1, 1, 3],
            dtype=dt.FLOAT32
        )

        self.cmn = fn.CropMirrorNorm(crop_w=img_width, crop_h=img_height, dtype=dt.FLOAT32, device="hpu")

        self.transpose = fn.Transpose(
            device="hpu",
            tensorDim=4,
            permutation=[1, 2, 0, 3] #NCHW
        )

    def definegraph(self):
        images = self.input()
        images = self.decode(images)
        mean = self.mean_node()
        std = self.std_node()
        images = self.cmn(images, mean, std)
        images = self.transpose(images)

        return images


# -----------------------------------------------------------------------------
# MediaPipe manager (persist pipes/iterators)
# -----------------------------------------------------------------------------
@dataclass
class _PipeState:
    pipe:   hpuMediaPipe | None = None
    it:     MediaGenericPytorchIterator | None = None
    bsz:    int | None = None
    H:      int | None = None
    W:      int | None = None


class MediaPipeTiler:
    """Owns and reuses MediaPipe pipes/iterators for main path"""
    def __init__(self) -> None:
        self._main  = _PipeState()

    def _rebuild(self, st: _PipeState, *, bsz: int, H: int, W: int) -> None:
        if st.pipe is not None:
            try:
                st.pipe.close()
            except Exception:
                pass
        pipe = hpuMediaPipe("legacy", 0, bsz, 1, "cpu", H, W)
        pipe.build()
        st.pipe, st.it, st.bsz, st.H, st.W = pipe, iter(MediaPytorchIterator(pipe)), bsz, H, W

    def ensure_main(self, *, bsz: int, H: int, W: int) -> tuple[hpuMediaPipe, MediaGenericPytorchIterator]:
        st = self._main
        if st.pipe is None or st.bsz != bsz or st.H != H or st.W != W:
            self._rebuild(st, bsz=bsz, H=H, W=W)
        return st.pipe, st.it  # type: ignore[return-value]

    def reset_iter(self) -> None:
        st = self._main
        if st.pipe is not None:
            st.it = iter(MediaPytorchIterator(st.pipe))

    def close_all(self) -> None:
        st = self._main
        try:
            if st.pipe is not None:
                st.pipe.close()
        except Exception:
            pass
        finally:
            st.pipe = None
            st.it = None
            st.bsz = st.H = st.W = None


_MP = MediaPipeTiler()
atexit.register(_MP.close_all)

def get_image_info(data):
    # Get image info using PIL without decoding
    try:
        with Image.open(io.BytesIO(data)) as img:
            return {
                'format': img.format,
                'size': img.size,
                'mode': img.mode
            }
    except Exception as e:
        raise ValueError(f"Input image bitstream is not in supported format: {str(e)}")


__all__ = ['Ovis2_5Processor']
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 1792 * 1792


class Ovis2_5ProcessorKwargs(ProcessingKwargs,
                             total=False):  # type: ignore[call-arg]
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            'convert_to_rgb': True,
            'min_pixels': MIN_PIXELS,
            'max_pixels': MAX_PIXELS,
        },
        "videos_kwargs": {
            'convert_to_rgb': True,
            'min_pixels': MIN_PIXELS,
            'max_pixels': MAX_PIXELS,
        }
    }


class Ovis2_5Processor(ProcessorMixin):
    r"""
    Constructs a Ovis processor which wraps a Ovis image processor
    and a Qwen2 tokenizer into a single processor.
    [`OvisProcessor`] offers all the functionalities of
    [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`].
    See the [`~OvisProcessor.__call__`] and [`~OvisProcessor.decode`]
    for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will
            be used to convert lists of messages in a chat into
            a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_pad_token"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_pad_token=None,
        patch_size=16,
        hidden_stride=2,
        temporal_patch_size=1,
        **kwargs,
    ):
        self.image_token = IMAGE_TOKEN
        self.video_token = VIDEO_TOKEN
        self.image_pad_token = "<|image_pad|>"

        self.patch_size = patch_size
        self.hidden_stride = hidden_stride
        self.temporal_patch_size = temporal_patch_size
        super().__init__(image_processor,
                         tokenizer,
                         chat_template=chat_template)

    @cached_property
    def extra_special_tokens(self):
        image_pad_token_id = self.tokenizer.get_vocab()[self.image_pad_token]
        extra_special_tokens = {
            "image_token": -200,
            "video_token": -201,
            "visual_atom": -300,
            "image_start": -301,
            "image_end": -302,
            "video_start": -303,
            "video_end": -304,
            'image_pad': image_pad_token_id,
        }
        return extra_special_tokens

    def __call__(
        self,
        images: ImageInput = None,
        videos: Union[np.ndarray, list[ImageInput]] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput],
                    list[PreTokenizedInput]] = None,
        **kwargs: Unpack[Ovis2_5ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s)
        and image(s). This method forwards the `text`and `kwargs` arguments
        to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text`
        is not `None` to encode the text. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`.
            Args:
                images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`,
                    `list[PIL.Image.Image]`, `list[np.ndarray]`,
                    `list[torch.Tensor]`):
                    The image or batch of images to be prepared.
                    Each image can be a PIL image, NumPy array or PyTorch
                    tensor. Both channels-first and channels-last formats
                    are supported.
                text (`str`, `list[str]`, `list[list[str]]`):
                    The sequence or batch of sequences to be encoded.
                    Each sequence can be a string or a list of strings
                    (pretokenized string). If the sequences are provided as
                    list of strings (pretokenized), you must set
                    `is_split_into_words=True` (to lift the ambiguity with
                    a batch of sequences).
                videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`,
                    `list[torch.Tensor]`):
                    The image or batch of videos to be prepared. Each video
                    can be a 4D NumPy array or PyTorch tensor, or a nested
                    list of 3D frames. Both channels-first and channels-last
                    formats are supported.
                return_tensors (`str` or [`~utils.TensorType`], *optional*):
                    If set, will return tensors of a particular framework.
                    Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
            Returns:
                [`BatchFeature`]: A [`BatchFeature`] with the following fields:
                - **input_ids** -- list of token ids to be fed to a model.
                  Returned when `text` is not `None`.
                - **attention_mask** -- list of indices specifying which tokens
                  should be attended to by the model (when
                  `return_attention_mask=True` or if *"attention_mask"*
                  is in `self.model_input_names` and if `text` is not `None`).
                - **pixel_values** -- Pixel values to be fed to a model.
                  Returned when `images` is not `None`.
                - **pixel_values_videos** -- Pixel values of videos to be fed to
                  a model. Returned when `videos` is not `None`.
                - **image_grid_thw** -- list of image 3D grid in LLM. Returned
                  when `images` is not `None`.
                - **video_grid_thw** -- list of video 3D grid in LLM. Returned
                  when `videos` is not `None`.
                - **second_per_grid_ts** -- list of video seconds per time grid.
                  Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Ovis2_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Process all images first
        visual_features = {}
        output = BatchFeature()
        if images is not None:
            processed_images = []
            image_placeholders_list = []
            grids = []
            # Process each image
            for image in images if isinstance(images, list) else [images]:
                pixel_values, image_placeholders, grid = (
                    self.preprocess_multidata(
                        images=image, **output_kwargs["images_kwargs"]))
                processed_images.append(pixel_values)
                image_placeholders_list.append(image_placeholders)
                grids.append(grid)

            # assign all processed images
            if processed_images:
                visual_features["image_placeholders"] = image_placeholders_list
            output["pixel_values"] = processed_images
            output["grids"] = grids

        if videos is not None:
            processed_videos = []
            videos_placeholders_list = []
            grids = []
            # Process each video
            for video in videos if isinstance(videos, list) else [videos]:
                pixel_values, video_placeholders, grid = (
                    self.preprocess_multidata(
                        video=video, **output_kwargs["videos_kwargs"]))
                processed_videos.append(pixel_values)
                videos_placeholders_list.append(video_placeholders)
                grids.append(grid)
            # assign all processed videos
            if processed_videos:
                visual_features[
                    "video_placeholders"] = videos_placeholders_list
            output["video_pixel_values"] = processed_videos
            output["video_grids"] = grids

        # Process text input
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            tokenized_batched_text = self._tokenize_with_visual_symbol(text)
            image_token_id = self.get_token_value("image_token")
            video_token_id = self.get_token_value("video_token")
            replaced_ids_list = []
            image_idx = 0
            video_idx = 0
            for ids_tensor in tokenized_batched_text:
                has_image_tokens = (image_token_id in ids_tensor
                                    and "image_placeholders" in visual_features
                                    and image_idx < len(
                                        visual_features["image_placeholders"]))
                has_video_tokens = (video_token_id in ids_tensor
                                    and "video_placeholders" in visual_features
                                    and video_idx < len(
                                        visual_features["video_placeholders"]))
                if has_image_tokens or has_video_tokens:
                    # Convert to list for easier manipulation
                    ids_list = ids_tensor.tolist()
                    new_ids = []

                    # Replace placeholders
                    for token_id in ids_list:
                        if token_id == image_token_id:
                            new_ids.extend(
                                visual_features["image_placeholders"]
                                [image_idx])
                            image_idx += 1
                        elif token_id == video_token_id:
                            new_ids.extend(
                                visual_features["video_placeholders"]
                                [video_idx])
                            video_idx += 1
                        else:
                            new_ids.append(token_id)
                    # Convert back to tensor
                    ids_tensor = torch.tensor(new_ids, dtype=torch.long)
                replaced_ids_list.append(ids_tensor)
            if replaced_ids_list:
                replaced_and_tokenized_ids = torch.stack(replaced_ids_list)
            else:
                replaced_and_tokenized_ids = torch.tensor([], dtype=torch.long)
            output["input_ids"] = replaced_and_tokenized_ids

            return output
        # If only images were provided
        return BatchFeature(data=visual_features)

    def _tokenize_with_visual_symbol(self,
                                     text_list: list[str]) -> torch.LongTensor:
        batch_token_ids = []
        for text in text_list:
            token_ids = []
            video_token_id = self.get_token_value("video_token")
            image_token_id = self.get_token_value("image_token")
            video_split_texts = text.split(self.video_token)

            for j, video_segment in enumerate(video_split_texts):
                image_split_texts = video_segment.split(self.image_token)
                text_chunks = [
                    self.tokenizer(chunk, add_special_tokens=False).input_ids
                    for chunk in image_split_texts
                ]
                segment_tokens = []
                for i, chunk in enumerate(text_chunks):
                    segment_tokens.extend(chunk)
                    if i < len(text_chunks) - 1:
                        segment_tokens.append(image_token_id)
                token_ids.extend(segment_tokens)
                if j < len(video_split_texts) - 1:
                    token_ids.append(video_token_id)

            batch_token_ids.append(token_ids)
        return torch.tensor(batch_token_ids, dtype=torch.long)

    # Copied from qwen2_vl
    def smart_resize(self,
                     height: int,
                     width: int,
                     factor: int = 28,
                     min_pixels: int = MIN_PIXELS,
                     max_pixels: int = MAX_PIXELS):
        """Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range
            ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if height < factor or width < factor:
            print(f"height:{height} or width:{width} must be "
                  f"larger than factor:{factor}")
            if height < width:
                width = round(factor / height * width)
                height = factor
            else:
                height = round(factor / width * height)
                width = factor

        elif max(height, width) / min(height, width) > 200:
            print(f"absolute aspect ratio must be smaller than 200, "
                  f"got {max(height, width) / min(height, width)}")
            if height > width:
                height = 200 * width
            else:
                width = 200 * height

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def get_token_value(self, tok):
        return self.extra_special_tokens[tok]

    def construct_visual_indicators(self, grid, is_video: bool = False):
        if is_video:
            start_token = self.get_token_value('video_start')
            end_token = self.get_token_value('video_end')
        else:
            start_token = self.get_token_value('image_start')
            end_token = self.get_token_value('image_end')

        image_placeholders = [start_token, self.get_token_value('visual_atom')]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(
                        self.get_token_value('visual_atom'))

        image_placeholders.append(end_token)
        return image_placeholders

    def construct_visual_placeholders(self, grid, is_video: bool = False):
        visual_placeholders = self.construct_visual_indicators((1, 1),
                                                               is_video)

        image_atom_token_id = self.get_token_value('visual_atom')
        # Extract the padding token ID from tokenizer
        image_padding_token_id = self.get_token_value('image_pad')

        num_image_atoms = grid[0] * grid[1] * grid[2]
        num_image_atoms //= self.hidden_stride**2
        num_image_atoms //= self.temporal_patch_size

        # Create a new list with padding tokens inserted
        padded_placeholder_tokens = []
        for token in visual_placeholders:
            if token == image_atom_token_id:
                padded_placeholder_tokens.extend([image_padding_token_id] *
                                                 num_image_atoms)
            else:
                padded_placeholder_tokens.append(image_padding_token_id)
        return padded_placeholder_tokens

    def preprocess_multidata(
        self,
        images: Optional[Union[PIL.Image.Image, list[PIL.Image.Image]]] = None,
        video: Optional[Union[list[PIL.Image.Image], np.ndarray]] = None,
        convert_to_rgb: Optional[bool] = True,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        return_tensors: Optional[str] = 'pt',
    ):
        is_video = False
        if images is not None:
            if not isinstance(images, list):
                images = [images]
        elif video is not None:
            is_video = True
            # type of vidoe in dummy_mm_data is np.ndarray
            if isinstance(video, np.ndarray):
                images = []
                for i in range(video.shape[0]):
                    image = PIL.Image.fromarray(video[i].astype(np.uint8))
                    images.append(image)
            elif isinstance(video, list):
                images = video
        min_pixels = min(max_pixels if max_pixels is not None else MAX_PIXELS,
                         min_pixels if min_pixels is not None else MIN_PIXELS)

        use_hpu_mp = (os.getenv("VLLM_USE_MEDIA_PIPELINE", "false").lower() in ("1", "true", "yes"))
        is_bytes_batch = isinstance(images, list) and images and isinstance(images[0], (bytes, bytearray))

        if use_hpu_mp and is_bytes_batch:
            queue_depth = 0
            num_threads = 1

            # Probe original size from first image (without decoding on CPU)
            with PIL.Image.open(io.BytesIO(images[0])) as _probe:
                width, height = _probe.size
            # Ovis policy: factor = patch_size * hidden_stride
            resized_height, resized_width = self.smart_resize(
                height, width,
                factor=self.patch_size * self.hidden_stride,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            batch_size = len(images)

            # if batch, H, W is changed, create new one
            main_pipe, main_iter = _MP.ensure_main(bsz=batch_size, H=resized_height, W=resized_width)

            img_list = np.empty(shape=[batch_size, ], dtype=object)
            for i in range(batch_size):
                img_list[i] = np.frombuffer(images[i], dtype=np.uint8)

            shared_q.put(img_list)
            try:
                processed_images = next(main_iter)[0]  # (B, 3, H, W) on hpu
            except StopIteration:
                _MP.reset_iter()
                _, main_iter = _MP.ensure_main(bsz=batch_size, H=resized_height, W=resized_width)
                processed_images = next(main_iter)[0]
            finally:
                shared_q.task_done()

            ps = self.patch_size
            hs = self.hidden_stride

            # processed: (B, C, H, W)  — resize + normalized
            B, C, H, W = processed_images.shape

            # check sizes
            assert H % ps == 0 and W % ps == 0, f"(H,W)=({H},{W}) not divisible by ps={ps}"
            Ty, Tx = H // ps, W // ps
            assert Ty % hs == 0 and Tx % hs == 0, f"(Ty,Tx)=({Ty},{Tx}) not divisible by hidden_stride={hs}"
            Gy, Gx = Ty // hs, Tx // hs

            # (B,C,H,W) -> (B,C,Ty,ps,Tx,ps) -> (B,C,Gy,hs,ps,Gx,hs,ps)
            x = processed_images.contiguous().reshape(B, C, Ty, ps, Tx, ps) \
                                    .reshape(B, C, Gy, hs, ps, Gx, hs, ps)

            # stride-aware order: (B, Gy, Gx, hs, hs, C, ps, ps)
            tiles = x.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()

            # (B*Ty*Tx, C*ps*ps)
            flatten_patches = tiles.reshape(B * Ty * Tx, C * ps * ps).contiguous()

            # grids/placeholder
            grid_t = 1
            grids = torch.tensor([[grid_t, Ty, Tx]] * B, device=flatten_patches.device)
            visual_placeholders = [
                self.construct_visual_placeholders([grid_t, Ty, Tx], is_video=False)
                for _ in range(B)
            ]

            return flatten_patches, visual_placeholders, grids

        images = [
            image.convert("RGB")
            if convert_to_rgb and image.mode != 'RGB' else image
            for image in images
        ]

        width, height = images[0].size
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=self.patch_size * self.hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            new_size = dict(height=resized_height, width=resized_width)
            image_pt = self.image_processor.preprocess(
                image, size=new_size, return_tensors="np")['pixel_values'][0]

            processed_images.append(image_pt)

        patches = np.array(processed_images)
        if patches.shape[0] % self.temporal_patch_size != 0:
            num_to_pad = self.temporal_patch_size - (patches.shape[0] %
                                                     self.temporal_patch_size)
            repeats = np.repeat(patches[-1][np.newaxis], num_to_pad, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
            grid_w // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size *
            self.patch_size * self.patch_size)

        visual_placeholders = self.construct_visual_placeholders(
            [grid_t, grid_h, grid_w], is_video)
        return torch.tensor(
            flatten_patches), visual_placeholders, torch.tensor(
                [[grid_t, grid_h, grid_w]])


AutoProcessor.register("Ovis2_5Processor", Ovis2_5Processor)
