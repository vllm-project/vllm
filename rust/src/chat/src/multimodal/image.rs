// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use llm_multimodal::{ImageDetail, ImageFrame, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::{ModalitySupport, MultimodalModelInfo, PreparedMedia, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary image encoder input.
pub(super) const IMAGE_PRIMARY_KEY: &str = "pixel_values";

impl MultimodalModelInfo {
    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    pub(super) async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
        cache_generation: u64,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        if uuids.len() != frames.len() {
            bail_multimodal!(
                "number of image UUIDs {} does not match number of images {}",
                uuids.len(),
                frames.len()
            );
        }

        let len = frames.len();
        let mut replacements = vec![None; len];
        let mut items = (0..len).map(|_| None).collect::<Vec<_>>();
        let mut missing_indices = Vec::new();
        let mut missing_frames = Vec::new();
        let mut missing_uuids = Vec::new();
        let mut missing_variants = Vec::new();

        for (index, (frame, uuid)) in frames.into_iter().zip(uuids).enumerate() {
            let variant = image_detail_variant(frame.detail);
            match self.processor_cache.get(Modality::Image, &frame.hash, model_dtype, variant) {
                Some(cached) => {
                    replacements[index] = Some(cached.replacement);
                    items[index] = Some(super::PreparedItem {
                        data: cached.data,
                        hash: frame.hash.clone(),
                        uuid,
                    });
                }
                None => {
                    missing_indices.push(index);
                    missing_frames.push(frame);
                    missing_uuids.push(uuid);
                    missing_variants.push(variant);
                }
            }
        }

        if !missing_frames.is_empty() {
            let preprocessed = self.preprocess_images(support, &missing_frames).await?;
            let missing_replacements =
                support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
            if missing_replacements.len() != missing_frames.len() {
                bail_multimodal!(
                    "number of image prompt replacements {} does not match number of images {}",
                    missing_replacements.len(),
                    missing_frames.len()
                );
            }
            let hashes = missing_frames.iter().map(|frame| frame.hash.clone()).collect();
            let missing_items = item::build_batched_items(
                &support.spec,
                preprocessed,
                hashes,
                missing_uuids,
                model_dtype,
            )?;

            for (((index, item), replacement), variant) in missing_indices
                .into_iter()
                .zip(missing_items)
                .zip(missing_replacements)
                .zip(missing_variants)
            {
                self.processor_cache.insert(
                    cache_generation,
                    Modality::Image,
                    &item.hash,
                    model_dtype,
                    variant,
                    &item.data,
                    &replacement,
                );
                replacements[index] = Some(replacement);
                items[index] = Some(item);
            }
        }

        let replacements = replacements
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| multimodal!("image processor cache merge left a missing replacement"))?;
        let items = items
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| multimodal!("image processor cache merge left a missing item"))?;

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
}

fn image_detail_variant(detail: ImageDetail) -> u8 {
    match detail {
        ImageDetail::Auto => 0,
        ImageDetail::Low => 1,
        ImageDetail::High => 2,
    }
}
