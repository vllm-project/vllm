//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{FieldLayout, ImageFrame, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargsItem, MmSharedField, MmSlice,
    SliceSpec,
};

use super::{ModalitySupport, MultimodalModelInfo, PreparedItem, PreparedMedia, tensor};
use crate::error::{Error, Result, bail_multimodal, multimodal};

impl MultimodalModelInfo {
    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    pub(super) async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        let preprocessed = self.preprocess_images(support, &frames).await?;
        let replacements =
            self.spec
                .prompt_replacements_for(&self.context, &preprocessed, Modality::Image)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let items = self.build_image_items(preprocessed, &frames, uuids, model_dtype)?;

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched image frames with the model's resolved vision
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        support: &ModalitySupport,
        image_frames: &[Arc<ImageFrame>],
    ) -> Result<PreprocessedEncoderInputs> {
        let config = support.config.clone();
        let processor = support.processor;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || Ok(processor.preprocess(&images, &config)?))
            .await
            .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }

    /// Convert one batch of preprocessed image tensors into per-item engine
    /// kwargs.
    ///
    /// Tensor fields are sliced per item according to the model spec's field
    /// layout declarations.
    fn build_image_items(
        &self,
        preprocessed: PreprocessedEncoderInputs,
        frames: &[Arc<ImageFrame>],
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<Vec<PreparedItem>> {
        let len = frames.len();
        let tensors = tensor::collect_tensors(preprocessed, "pixel_values", model_dtype)?;

        let mut items = Vec::with_capacity(len);
        for (index, (frame, uuid)) in izip!(frames, uuids).enumerate() {
            let mut data = MmKwargsItem::new();
            for (key, tensor) in &tensors {
                let keep_on_cpu = self.spec.keep_on_cpu_keys.contains(key);
                let (value, field) = match self.spec.field_layouts.get(key) {
                    Some(FieldLayout::Batched) => (
                        tensor.batched_value_at(index)?,
                        MmField::Batched(MmBatchedField { keep_on_cpu }),
                    ),
                    Some(FieldLayout::Flat { sizes_key }) => {
                        let sizes = tensors.get(sizes_key).ok_or_else(|| {
                            multimodal!("flat tensor sizes key `{sizes_key}` is missing")
                        })?;
                        let (start, end) = tensor::flat_range_for_index(sizes, sizes_key, index)?;
                        (
                            tensor.flat_value_range(start, end)?,
                            MmField::Flat(MmFlatField {
                                slices: vec![MmSlice::Slice(SliceSpec {
                                    start: Some(0),
                                    stop: Some((end - start) as isize),
                                    step: None,
                                })],
                                dim: 0,
                                keep_on_cpu,
                            }),
                        )
                    }
                    None => (
                        tensor.clone(),
                        MmField::Shared(MmSharedField {
                            batch_size: len,
                            keep_on_cpu,
                        }),
                    ),
                };

                data.insert(
                    key.clone(),
                    MmFieldElem {
                        data: Some(value.try_into()?),
                        field,
                    },
                );
            }

            items.push(PreparedItem {
                data,
                hash: frame.hash.clone(),
                uuid,
            });
        }

        Ok(items)
    }
}
