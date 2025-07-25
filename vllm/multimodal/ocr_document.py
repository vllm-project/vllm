import dataclasses

import torch
import torch.types

from vllm.inputs import InputContext
from vllm.multimodal.base import MultiModalPlugin
from vllm.multimodal.inputs import ModalityData, MultiModalInputs

OCR_DOCUMENT_PLUGIN_KEY = "ocr_document"


@dataclasses.dataclass
class OcrDocument:
    token_bboxes: torch.Tensor
    images: list[torch.Tensor] | torch.Tensor
    roi_bboxes: torch.Tensor
    roi_token_indices: torch.Tensor | None
    pages: torch.Tensor
    token_count: int = 0
    page_count: int = 0

    def to(self, *args, **kwargs):
        return OcrDocument(
            token_bboxes=self._to(self.token_bboxes, *args, **kwargs),
            images=self._to(self.images, *args, **kwargs),
            roi_bboxes=self._to(self.roi_bboxes, *args, **kwargs),
            roi_token_indices=self._to(self.roi_token_indices, *args,
                                       **kwargs),
            pages=self._to(self.pages, *args, **kwargs),
            token_count=self.token_count,
            page_count=self.page_count,
        )

    @staticmethod
    def _to(_obj, *args, **kwargs):
        if isinstance(_obj, torch.Tensor):
            return _obj.to(*args, **kwargs)
        else:
            return [x.to(*args, **kwargs) for x in _obj]


class OcrDocumentPlugin(MultiModalPlugin):
    """Multimodal plugin for OCR documents.

    This plugin works only with TiltModelRunner. Standard pipeline does not
    support batching of arbitrary objects.

    """

    def get_data_key(self) -> str:
        return OCR_DOCUMENT_PLUGIN_KEY

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: ModalityData[object],
    ) -> MultiModalInputs:
        if not isinstance(data, OcrDocument):
            raise TypeError()
        return MultiModalInputs({
            "ocr_document": data,
        })

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        # NOTE: TILT does not add any addtional tokens. Instead, multimodal
        # representations are merged into representations of every token.
        return 0
