"""TILT preprocessing pipeline."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias, TypeVar, cast

import torch
from PIL.Image import Image
from tokenizers import Tokenizer
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from vllm.inputs import TokensPrompt
from vllm.inputs.data import TiltPrompt
from vllm.multimodal.ocr_document import OcrDocument
from vllm.transformers_utils.configs.tilt import TiltConfig

logger = logging.getLogger(__name__)

# this is the width of an ISO paper format scaled to fit in a 1x1 square
NORMALIZED_ISO_WIDTH = 1.0 / math.sqrt(2)

T = TypeVar("T")
Rect: TypeAlias = tuple[int, int, int, int]


@dataclass
class Page:
    words: list[str]
    bboxes: list[Rect]
    width: float
    height: float
    image: Image


@dataclass
class ProcessedPage:
    # these are not necessarily in 1-1 correspondence with Page, e.g. filtering can occur
    token_ids: torch.Tensor
    bboxes: torch.Tensor
    # we need only height:
    #  1. after preprocessing width is constant
    #  2. we don't use width, only height
    height: float
    token_slice: slice
    roi_bboxes: torch.Tensor
    image: torch.Tensor | None
    page_ids: torch.Tensor


@dataclass
class Document:
    ident: str
    pages: list[Page]
    split: str | None = None


@dataclass
class TensorizedDocument:
    token_ids: torch.Tensor
    bboxes: torch.Tensor
    roi_bboxes: torch.Tensor
    pages: list[slice]
    images: list[torch.Tensor | None]
    page_token_ids: torch.Tensor


@dataclass
class Question:
    feature_name: str  # key in a features structure (e.g. from due annotations)
    text: str  # actual text of a formed question (in QA task usually the same as feature_name)


@dataclass
class Prefix:
    token_ids: torch.Tensor
    bboxes: torch.Tensor


def _clone_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """
    Clone a tokenizer.

    This is necessary, as the options for the tokenizer are passed through methods which mutate it. Sharing
    the same tokenizer is therefore very fragile.
    """
    serialized = tokenizer.to_str()
    return Tokenizer.from_str(serialized)


class TiltDocumentPreprocessor:

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
        image_limit: int,
        image_width: int = 768,
        max_image_height: int = 2048,
        crop_bboxes: bool = True,
        filter_empty_tokens: bool = True,
    ) -> None:
        self._image_limit = image_limit
        self._image_width = image_width
        self._max_length = max_length
        self._tokenizer = _clone_tokenizer(tokenizer)
        self._image_processor = TiltImageProcessor(image_width,
                                                   max_image_height)
        self._crop_bboxes = crop_bboxes
        self._filter_empty_tokens = filter_empty_tokens

    def process(self, document: Document) -> TensorizedDocument:
        processed_pages = self.process_pages(document.pages)
        if not processed_pages:
            raise ValueError(
                "An empty document should have been caught earlier and never get to this point."
            )
        return TensorizedDocument(
            token_ids=torch.cat([page.token_ids for page in processed_pages]),
            bboxes=torch.cat([page.bboxes for page in processed_pages]),
            roi_bboxes=torch.cat([page.roi_bboxes
                                  for page in processed_pages]),
            pages=[page.token_slice for page in processed_pages],
            images=[page.image for page in processed_pages],
            page_token_ids=torch.cat(
                [page.page_ids for page in processed_pages]),
        )

    def process_pages(self, pages: list[Page]) -> list[ProcessedPage]:
        processed_pages = []
        preceding_tokens = 0
        vertical_offset = 0.0  # offset used to translate bboxes
        for page_id, page in enumerate(pages):
            self._tokenizer.enable_truncation(self._max_length -
                                              preceding_tokens)
            discard_image = page_id >= self._image_limit
            processed_page = self.process_page(
                page, vertical_offset, preceding_tokens, discard_image,
                page_id + 1)  # page + 1, as 0 is considered to be padding
            processed_pages.append(processed_page)

            vertical_offset += processed_page.height
            preceding_tokens += len(processed_page.token_ids)
            if preceding_tokens >= self._max_length:
                break

        return processed_pages

    def process_page(
        self,
        page: Page,
        vertical_offset: float,
        preceding_tokens: int,
        discard_image: bool,
        page_id: int,
    ) -> ProcessedPage:
        if self._filter_empty_tokens:
            words, bbox_list = self.filter_empty(page.words, page.bboxes)
        else:
            words, bbox_list = page.words, page.bboxes

        # tensorize bboxes; in MLM we might want to modify them before tensorization
        if bbox_list:
            bboxes = torch.tensor(bbox_list, dtype=torch.float32)
        else:
            bboxes = torch.empty(0, 4)

        # Cropping and normalization
        if self._crop_bboxes:
            bboxes = self.crop_bboxes(bboxes, page.height, page.width)
        bboxes, normalized_page_height = self.normalize_bboxes(
            bboxes, page.height, page.width)

        # tokenize words and interpolate bboxes accordingly
        token_ids, tokens_to_words, token_offsets = self.subtokenize(words)
        bboxes = self.interpolate_bboxes(bboxes, tokens_to_words,
                                         token_offsets)

        image = self._image_processor.process(
            page.image) if not discard_image else None

        # TODO if there is no image, we could also discard the clipping bboxes.
        #  Need to recompute ROI slices though, so it adds a bit more complexity,
        #  as the relation token_ids -> roi_bboxes is no longer identity.
        # Note: in the current use cases, we don't discard images, i.e. image_limit >= page_limit,
        #  so we won't get anything from optimizing this
        roi_bboxes = self.compute_roi_bboxes(bboxes)
        roi_bboxes = torch.cat(
            [torch.full((roi_bboxes.shape[0], 1), fill_value=-1), roi_bboxes],
            axis=1)
        roi_bboxes[:, 0] = page_id - 1

        # translate bboxes so that pages are laid out vertically one under another
        bboxes = self.translate_bboxes(bboxes, vertical_offset)
        token_slice = slice(preceding_tokens,
                            preceding_tokens + len(token_ids))
        return ProcessedPage(
            token_ids,
            bboxes,
            normalized_page_height,
            token_slice,
            roi_bboxes,
            image,
            torch.ones_like(token_ids) * page_id,
        )

    def normalize_bboxes(self, bboxes: torch.Tensor, page_height: float,
                         page_width: float) -> tuple[torch.Tensor, float]:
        scale = NORMALIZED_ISO_WIDTH / page_width
        bboxes = bboxes * scale
        normalized_page_height = page_height * scale
        return bboxes, normalized_page_height

    def filter_empty(
            self, words: Sequence[str],
            bboxes: Sequence[Rect]) -> tuple[Sequence[str], Sequence[Rect]]:
        pairs = [(word, bbox) for word, bbox in zip(words, bboxes, strict=True)
                 if word.strip()]
        if len(pairs) == 0:
            return [], []
        return cast(tuple[Sequence[str], Sequence[Rect]], tuple(zip(*pairs)))

    def crop_bboxes(self, bboxes: torch.Tensor, page_height: float,
                    page_width: float) -> torch.Tensor:
        cropping_values = torch.tensor(
            [page_width, page_height, page_width, page_height],
            dtype=torch.float32)
        return torch.clamp(bboxes, min=torch.tensor(0), max=cropping_values)

    def subtokenize(
        self, words: Sequence[str]
    ) -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
        """
        Subtokenize the words and interpolate bboxes.

        Returns:
            a tensor of token ids, list of word indices for each token, list of lengths for each token
        """
        # 1. We don't need to fix offsets, as we don't use original strings anymore!
        # 2. What if we omit fixing the final token if it's incomplete? Let's try! This happens only in the last page.

        encoding = self._tokenizer.encode(words,
                                          is_pretokenized=True,
                                          add_special_tokens=False)
        token_ids = torch.tensor(encoding.ids, dtype=torch.long)
        return token_ids, encoding.word_ids, encoding.offsets

    def interpolate_bboxes(
        self,
        bboxes: torch.Tensor,
        tokens_to_words: Sequence[int],
        token_offsets: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Subdivide word bboxes into bboxes of constituent subtokens, proportionally to token lengths.

        Args:
            bboxes (Tensor): shape [N, 4], bboxes of original words
            tokens_to_words (list[int]): a list that maps subtoken index to its word index
            token_offsets (list[tuple(int, int)]): a list of (start, end) offsets for each subtoken
        """
        right_weights, left_weights, word_weights = [], [], [
            0.0
        ] * bboxes.shape[0]
        # Loop over the tokens once, computing the *fixed* token lengths and updating the word weights
        # The fix is needed as the tokenizer generates inconsistent offsets if word is tokenized
        # into beginning-of-word token and the rest-of-word token, i.e. "word" -> ["_", "word"].
        if len(token_offsets) != 0:
            for _ind, ((start, end),
                       (next_start, _)) in enumerate(pairwise(token_offsets)):
                word = tokens_to_words[_ind]
                if word == tokens_to_words[_ind + 1] and end > next_start:
                    current_length = next_start - start
                else:
                    current_length = end - start
                right_weights.append(word_weights[tokens_to_words[_ind]])
                word_weights[tokens_to_words[_ind]] += current_length
                left_weights.append(word_weights[tokens_to_words[_ind]])
            # The last token is not handled in the loop above, so we need to handle it separately.
            right_weights.append(word_weights[tokens_to_words[-1]])
            word_weights[tokens_to_words[-1]] += (token_offsets[-1][1] -
                                                  token_offsets[-1][0])
            left_weights.append(word_weights[tokens_to_words[-1]])

        weights = torch.clamp(torch.tensor(word_weights, dtype=torch.float32),
                              min=1.0)
        adjusted_widths = (bboxes[:, 2] - bboxes[:, 0]) / weights
        relative_cutpoints = torch.tensor([right_weights, left_weights],
                                          dtype=torch.float32)
        relative_cutpoints *= adjusted_widths[tokens_to_words]

        expanded_bboxes = bboxes[tokens_to_words]
        return torch.stack(
            [
                expanded_bboxes[:, 0] + relative_cutpoints[0],
                expanded_bboxes[:, 1],
                expanded_bboxes[:, 0] + relative_cutpoints[1],
                expanded_bboxes[:, 3],
            ],
            dim=1,
        )

    def compute_roi_bboxes(self,
                           normalized_bboxes: torch.Tensor) -> torch.Tensor:
        """Scale the bboxes to image coordinates, to be used for token image embeddings."""
        scale = self._image_width / NORMALIZED_ISO_WIDTH
        return normalized_bboxes * scale

    def translate_bboxes(self, bboxes: torch.Tensor,
                         offset: float) -> torch.Tensor:
        return bboxes + torch.tensor([0, offset, 0, offset],
                                     dtype=torch.float32)


class TiltImageProcessor:
    """
    Processor of images for TILT.

    Args:
        target_width: the target width to which images are resized
        max_height: the maximum height to which images are cropped after resizing
    """

    # [0.2989*R + 0.57*G + 0.113*B] / 255 (this corresponds to human vision color sensitivity)
    LIGHT_SENSITIVITIES = torch.tensor([0.001172, 0.002302, 0.000447])

    def __init__(
        self,
        target_width: int,
        max_height: int,
    ) -> None:
        self.target_width = target_width
        self.max_height = max_height

    def resize(self, image: torch.Tensor) -> torch.Tensor:
        _, image_height, image_width = image.shape
        if image_width == self.target_width:
            return image
        target_height = image_height * self.target_width / image_width
        target_height = max(1, round(target_height))
        return cast(
            torch.Tensor,
            resize(
                image,
                size=[target_height, self.target_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        )

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :self.max_height, :]

    def ensure_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 1:
            return image / 255.0
        else:
            _, width, height = image.shape
            tmp = self.LIGHT_SENSITIVITIES @ image.view(3, -1)
            return tmp.view(1, width, height)

    def process(self, image: Image) -> torch.Tensor:
        image = pil_to_tensor(image).float()
        image = self.resize(image)
        image = self.crop(image)
        image = self.ensure_grayscale(image)
        return image


class TiltQuestionProcessor:
    INITIAL_BBOX = torch.tensor([-0.01, -0.01, -0.01, -0.01],
                                dtype=torch.float32)
    STEP = torch.tensor([[0.05, 0, 0.05, 0]], dtype=torch.float32)

    def __init__(
        self,
        tokenizer: Tokenizer,
        prefix_separator: str = " : ",
        max_length: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._prefix_separator = prefix_separator
        self._max_length = max_length

    def process(self, question: Question) -> Prefix:
        prefix = question.text + self._prefix_separator
        encoding = self._tokenizer.encode(prefix,
                                          is_pretokenized=False,
                                          add_special_tokens=False)
        if self._max_length and len(encoding.ids) > self._max_length:
            # TODO: if we limit the length of question (in words, we don't have the tokenizer earlier),
            #  we can replace this exception with truncation
            msg = (
                f"Tokenized question has length {len(encoding.ids)} "
                f"which is more than maximum permitted length {self._max_length}."
            )
            raise RuntimeError(msg)
        token_ids = torch.tensor(encoding.ids, dtype=torch.long)
        bboxes = self.compute_bboxes(len(token_ids))
        return Prefix(token_ids, bboxes)

    def compute_bboxes(self, length: int) -> torch.Tensor:
        steps = torch.arange(0.0, length, dtype=torch.float32)
        return self.INITIAL_BBOX[None, :] + steps[:, None] * self.STEP


class TiltPreprocessor:

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 4096,
        max_question_length: int | None = 256,
        image_limit: int = 250,
        image_width: int = 768,
        max_image_height: int = 2048,
        crop_bboxes: bool = True,
        filter_empty_tokens: bool = True,
        prefix_separator: str = " : ",
    ):
        self.question_proc = TiltQuestionProcessor(
            tokenizer=tokenizer,
            max_length=max_question_length,
            prefix_separator=prefix_separator,
        )
        self.document_proc = TiltDocumentPreprocessor(
            tokenizer=tokenizer,
            max_length=max_length,
            image_limit=image_limit,
            image_width=image_width,
            max_image_height=max_image_height,
            crop_bboxes=crop_bboxes,
            filter_empty_tokens=filter_empty_tokens,
        )

    @classmethod
    def from_config(cls, model_config: TiltConfig,
                    tokenizer: Tokenizer) -> TiltPreprocessor:
        return cls(
            tokenizer=tokenizer,
            max_length=model_config.max_seq_length,
            max_question_length=model_config.max_question_length,
            image_limit=model_config.image_limit,
            image_width=model_config.image_width,
            max_image_height=model_config.max_image_height,
            crop_bboxes=model_config.crop_bboxes,
            prefix_separator=model_config.prefix_separator,
        )

    def preprocess(self, document: Document, questions: list[Question]):
        prefixes = [self.question_proc.process(q) for q in questions]
        processed_doc = self.document_proc.process(document)

        result = []
        for prefix in prefixes:
            result.append(
                TiltPrompt(
                    encoder_prefix_prompt=TokensPrompt(
                        prompt_token_ids=prefix.token_ids.tolist(),
                        multi_modal_data={
                            "ocr_document":
                            OcrDocument(
                                token_bboxes=prefix.bboxes,
                                images=[],
                                roi_bboxes=torch.empty(0, 5),
                                roi_token_indices=None,
                                pages=(torch.ones_like(prefix.token_ids) *
                                       processed_doc.page_token_ids[0].item()),
                                token_count=prefix.token_ids.shape[-1],
                                page_count=0,
                            ),
                        },
                    ),
                    encoder_prompt=TokensPrompt(
                        prompt_token_ids=processed_doc.token_ids.tolist(),
                        multi_modal_data={
                            "ocr_document":
                            OcrDocument(
                                token_bboxes=processed_doc.bboxes,
                                images=processed_doc.images,
                                roi_bboxes=processed_doc.roi_bboxes,
                                roi_token_indices=None,
                                pages=processed_doc.page_token_ids,
                                token_count=processed_doc.token_ids.shape[-1],
                                page_count=len(processed_doc.pages),
                            ),
                        },
                    ),
                    decoder_prompt=TokensPrompt(prompt_token_ids=[0], ),
                ))
        return result
