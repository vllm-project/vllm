// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared lowering from batched preprocessor output to per-item engine kwargs.

use llm_multimodal::{FieldLayout, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmFlatField, MmKwargsItem, MmSharedField, MmSlice,
    SliceSpec,
};

use super::{PreparedItem, ResolvedMultimodalSpec, tensor};
use crate::error::{Result, multimodal};

/// Split one batch of preprocessed tensors into engine kwargs per media item.
pub(super) fn build_batched_items(
    spec: &ResolvedMultimodalSpec,
    preprocessed: PreprocessedEncoderInputs,
    mm_hashes: Vec<String>,
    float_dtype: ModelDtype,
) -> Result<Vec<PreparedItem>> {
    let len = mm_hashes.len();
    let tensors = tensor::collect_tensors(preprocessed, spec.primary_key(), float_dtype)?;

    let mut items = Vec::with_capacity(len);
    for (index, mm_hash) in mm_hashes.into_iter().enumerate() {
        let mut data = MmKwargsItem::new();
        for (key, tensor) in &tensors {
            let keep_on_cpu = spec.keep_on_cpu_keys.contains(key);
            let (value, field) = match spec.field_layout_for(key) {
                Some(FieldLayout::Batched) => (
                    tensor.batched_wire_value_at(index)?,
                    MmField::Batched(MmBatchedField { keep_on_cpu }),
                ),
                Some(FieldLayout::Flat { sizes_key }) => {
                    let sizes = tensors.get(sizes_key).ok_or_else(|| {
                        multimodal!("flat tensor sizes key `{sizes_key}` is missing")
                    })?;
                    let (start, end) = tensor::flat_range_for_index(sizes, sizes_key, index)?;
                    (
                        tensor.flat_wire_value_range(start, end)?,
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
                    tensor.try_into()?,
                    MmField::Shared(MmSharedField {
                        batch_size: len,
                        keep_on_cpu,
                    }),
                ),
            };

            data.insert(
                key.clone(),
                MmFieldElem {
                    data: Some(value),
                    field,
                },
            );
        }

        items.push(PreparedItem { data, mm_hash });
    }

    Ok(items)
}
