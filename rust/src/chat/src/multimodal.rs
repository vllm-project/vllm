//! Chat-layer multimodal image preparation.
//!
//! This module owns the narrow image-only multimodal path for chat requests:
//! it extracts image parts from structured chat messages, fetches and
//! preprocesses them through `llm-multimodal`, expands rendered prompt
//! placeholders after tokenization, and builds the engine-facing
//! `MmFeatures` payload.
//!
//! Raw media stays above `vllm-text`; this module lowers it into token IDs and
//! opaque tensor payloads before the request is handed to text generation.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use itertools::izip;
use llm_multimodal::{
    AsyncMultiModalTracker, FieldLayout, MediaConnector, MediaConnectorConfig, MediaContentPart,
    Modality, ModelMetadata, ModelProcessorSpec, ModelRegistry, PreProcessorConfig,
    PreprocessedEncoderInputs as PreprocessedImages, PromptReplacement, Tokenizer as TokenResolver,
    TrackedMedia, VisionPreProcessor as ImagePreProcessor,
    VisionProcessorRegistry as ImageProcessorRegistry,
};
use tracing::warn;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmFeatureSpec, MmFeatures, MmField, MmFieldElem, MmFlatField, MmKwargsItem,
    MmSharedField, MmSlice, PlaceholderRange, SliceSpec,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use crate::error::{Error, Result, bail_multimodal, multimodal};
use crate::renderer::RenderedPrompt;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

mod tensor;

/// Resolved multimodal support for one loaded model.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    spec: ResolvedMultimodalSpec,
    image_processor: ResolvedImageProcessor,
    media_connector: Arc<MediaConnector>,
}

/// Model metadata and tokenizer access shared by all multimodal specs.
#[derive(Clone)]
struct MultimodalModelContext {
    model_id: String,
    model_type: Option<String>,
    config: serde_json::Value,
    tokenizer: TokenizerResolver,
}

impl MultimodalModelContext {
    fn metadata(&self) -> ModelMetadata<'_> {
        ModelMetadata {
            model_id: &self.model_id,
            tokenizer: &self.tokenizer,
            config: &self.config,
        }
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.0.as_ref()
    }

    /// Resolve a static model processor spec for one loaded model.
    fn resolve_model_spec(&self) -> Option<&'static dyn ModelProcessorSpec> {
        static REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);
        REGISTRY.lookup(&self.metadata())
    }

    /// Resolve a static image preprocessor for one loaded model.
    fn resolve_image_processor(&self) -> Option<&'static dyn ImagePreProcessor> {
        static REGISTRY: LazyLock<ImageProcessorRegistry> =
            LazyLock::new(ImageProcessorRegistry::with_defaults);
        REGISTRY.find(&self.model_id, self.model_type.as_deref())
    }
}

/// Static model-specific prompt and tensor-layout behavior.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    placeholder_token: String,
    placeholder_marker_token_id: u32,
    placeholder_embed_token_id: u32,
    field_layouts: HashMap<String, FieldLayout>,
    keep_on_cpu_keys: HashSet<String>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec, context: &MultimodalModelContext) -> Result<Self> {
        let metadata = context.metadata();
        let placeholder_token =
            raw.placeholder_token(&metadata).map_err(|error| multimodal!("{error}"))?;
        // This is the rendered prompt marker, so resolve it from the token
        // string itself. Do not use `ModelProcessorSpec::placeholder_token_id()`:
        // for some specs that ID is the replacement vision/patch token,
        // not necessarily the token ID of `placeholder_token`.
        let placeholder_marker_token_id =
            context.tokenizer().token_to_id(&placeholder_token).ok_or_else(|| {
                multimodal!(
                    "placeholder token `{placeholder_token}` is not in the tokenizer vocabulary"
                )
            })?;
        let placeholder_embed_token_id =
            raw.placeholder_token_id(&metadata).map_err(|error| multimodal!("{error}"))? as u32;

        Ok(Self {
            raw,
            placeholder_token,
            placeholder_marker_token_id,
            placeholder_embed_token_id,
            field_layouts: raw.field_layouts(),
            keep_on_cpu_keys: raw.keep_on_cpu_keys().into_iter().collect(),
        })
    }

    fn prompt_replacements(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedImages,
    ) -> Result<Vec<PromptReplacement>> {
        self.raw
            .prompt_replacements(&context.metadata(), preprocessed)
            .map_err(|error| multimodal!("{error}"))
    }
}

/// Static image preprocessor plus its loaded config.
#[derive(Clone)]
struct ResolvedImageProcessor {
    raw: &'static dyn ImagePreProcessor,
    config: PreProcessorConfig,
}

/// Request-scoped fetched media, kept together with tracker UUID metadata.
struct FetchedImageMedia {
    frames: Vec<Arc<llm_multimodal::ImageFrame>>,
    uuids: Vec<Option<String>>,
}

impl MultimodalModelInfo {
    /// Load and resolve multimodal support from model files.
    ///
    /// Returns `Ok(Some(_))` only when both the model spec and image processor
    /// are registered. File read/parse failures are real errors; unsupported
    /// model families are logged and returned as `Ok(None)`.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        config_path: Option<&Path>,
        preprocessor_config_path: Option<&Path>,
        tokenizer: DynTokenizer,
    ) -> Result<Option<Self>> {
        let config = match config_path {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .map_err(|error| multimodal!("failed to read config.json: {error}"))?;
                serde_json::from_str(&text)
                    .map_err(|error| multimodal!("failed to parse config.json: {error}"))?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let preprocessor_config = match preprocessor_config_path {
            Some(path) => {
                let text = fs::read_to_string(path).map_err(|error| {
                    multimodal!("failed to read preprocessor_config.json: {error}")
                })?;
                PreProcessorConfig::from_json(&text).map_err(|error| {
                    multimodal!("failed to parse preprocessor_config.json: {error}")
                })?
            }
            None => PreProcessorConfig::default(),
        };

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        let Some(spec) = context.resolve_model_spec() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "multimodal model spec is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };
        let spec = ResolvedMultimodalSpec::new(spec, &context)?;

        let Some(image_processor) = context.resolve_image_processor() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "image processor is not registered; disabling multimodal support for this model"
            );
            return Ok(None);
        };

        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .map_err(|error| multimodal!("{error}"))?,
        );

        Ok(Some(Self {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: image_processor,
                config: preprocessor_config,
            },
            media_connector,
        }))
    }

    /// Return the template-visible placeholder token for this model.
    ///
    /// The HF renderer uses this token while flattening image content in string
    /// content format.
    pub fn placeholder_token(&self) -> &str {
        &self.spec.placeholder_token
    }
}

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their image placeholders are expanded, and
/// preprocessed image features are attached for engine-core transport.
pub(crate) async fn finalize_rendered_prompt(
    request: &ChatRequest,
    rendered: RenderedPrompt,
    info: Option<&MultimodalModelInfo>,
    model_dtype: ModelDtype,
) -> Result<(Prompt, Option<MmFeatures>)> {
    if !request.has_multimodal() {
        return Ok((rendered.prompt, None));
    }
    let info = info.ok_or(Error::UnsupportedMultimodalRenderer)?;
    let mut prompt_token_ids = match rendered.prompt {
        Prompt::Text(prompt) => info
            .context
            .tokenizer()
            .encode(&prompt, request.add_special_tokens)
            .map_err(|error| multimodal!("{error}"))?,
        Prompt::TokenIds(token_ids) => token_ids,
    };
    let media_parts = extract_media_parts(request)?;
    let prepared = info.prepare_multimodal(media_parts, &mut prompt_token_ids, model_dtype).await?;

    Ok((Prompt::TokenIds(prompt_token_ids), Some(prepared)))
}

/// Extract image media parts from chat messages in message/content order.
///
/// Assistant history is skipped because generated assistant blocks are already
/// represented as text for prompt rendering in this crate.
fn extract_media_parts(request: &ChatRequest) -> Result<Vec<MediaContentPart>> {
    let mut all_parts = Vec::new();
    for message in &request.messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        for part in parts {
            match part {
                ChatContentPart::Text { .. } => {}
                ChatContentPart::ImageUrl {
                    image_url,
                    detail,
                    uuid,
                } => all_parts.push(MediaContentPart::ImageUrl {
                    url: image_url.clone(),
                    detail: *detail,
                    uuid: uuid.clone(),
                }),
            }
        }
    }
    Ok(all_parts)
}

impl MultimodalModelInfo {
    /// Run media fetch, image preprocessing, prompt expansion, and feature
    /// build.
    ///
    /// `prompt_token_ids` is mutated in place because placeholder expansion
    /// changes both the final prompt and the offsets recorded in
    /// `PlaceholderRange`.
    async fn prepare_multimodal(
        &self,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
        model_dtype: ModelDtype,
    ) -> Result<MmFeatures> {
        if media_parts.is_empty() {
            return Ok(Vec::new());
        }
        let media_parts_len = media_parts.len();

        let fetched = self.fetch_images(media_parts).await?;
        let preprocessed = self.preprocess_images(&fetched.frames).await?;
        let replacements = self.spec.prompt_replacements(&self.context, &preprocessed)?;
        let ranges = self.expand_prompt_tokens(prompt_token_ids, replacements)?;

        let features = self.build_features(preprocessed, fetched, ranges, model_dtype)?;
        if features.len() != media_parts_len {
            bail_multimodal!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts_len
            );
        }
        Ok(features)
    }

    /// Fetch all image parts and preserve their request-order UUID metadata.
    async fn fetch_images(&self, media_parts: Vec<MediaContentPart>) -> Result<FetchedImageMedia> {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        for part in media_parts {
            tracker.push_part(part).map_err(|error| multimodal!("{error}"))?;
        }

        let tracker_output = tracker.finalize().await.map_err(|error| multimodal!("{error}"))?;
        let images = tracker_output.data.get(&Modality::Image).cloned().unwrap_or_default();
        let uuids = tracker_output.uuids.get(&Modality::Image).cloned().unwrap_or_default();

        let frames = images
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Image(frame) => Ok(frame),
                _ => Err(Error::UnsupportedMultimodalContent("non-image")),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(FetchedImageMedia { frames, uuids })
    }

    /// Preprocess fetched image frames with the model's resolved image
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        image_frames: &[Arc<llm_multimodal::ImageFrame>],
    ) -> Result<PreprocessedImages> {
        let config = self.image_processor.config.clone();
        let processor = self.image_processor.raw;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || {
            processor.preprocess(&images, &config).map_err(|error| multimodal!("{error}"))
        })
        .await
        .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }

    /// Replace rendered placeholder markers with model-specific replacement
    /// tokens.
    ///
    /// Replacements are consumed in order, matching the original media-part
    /// order. The returned ranges point into the already-expanded prompt.
    fn expand_prompt_tokens(
        &self,
        prompt_token_ids: &mut Vec<u32>,
        replacements: Vec<PromptReplacement>,
    ) -> Result<Vec<PlaceholderRange>> {
        expand_prompt_token_ids(
            prompt_token_ids,
            replacements,
            self.spec.placeholder_marker_token_id,
            self.spec.placeholder_embed_token_id,
            &self.spec.placeholder_token,
        )
    }

    /// Convert preprocessed image tensors into engine-core multimodal features.
    ///
    /// One `MmFeatureSpec` is produced per image. Tensor fields are
    /// sliced according to the model spec's field layout declarations.
    fn build_features(
        &self,
        preprocessed: PreprocessedImages,
        images: FetchedImageMedia,
        ranges: Vec<PlaceholderRange>,
        model_dtype: ModelDtype,
    ) -> Result<MmFeatures> {
        let len = images.frames.len();
        let tensors = tensor::collect_tensors(preprocessed, model_dtype)?;

        let mut features = Vec::with_capacity(images.frames.len());
        for (index, (frame, uuid, range)) in izip!(images.frames, images.uuids, ranges).enumerate()
        {
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

            let hash = frame.hash.clone();
            features.push(MmFeatureSpec {
                data: Some(data),
                modality: "image".to_string(),
                identifier: uuid.unwrap_or_else(|| hash.clone()),
                mm_position: range,
                mm_hash: Some(hash),
            });
        }

        Ok(features)
    }
}

fn expand_prompt_token_ids(
    prompt_token_ids: &mut Vec<u32>,
    replacements: Vec<PromptReplacement>,
    placeholder_marker_token_id: u32,
    placeholder_embed_token_id: u32,
    placeholder_token: &str,
) -> Result<Vec<PlaceholderRange>> {
    if replacements.is_empty() {
        return Ok(Vec::new());
    }

    let replacement_growth = replacements.iter().fold(0usize, |total, replacement| {
        total.saturating_add(replacement.tokens.len().saturating_sub(1))
    });
    let mut expanded =
        Vec::with_capacity(prompt_token_ids.len().saturating_add(replacement_growth));
    let mut ranges = Vec::with_capacity(replacements.len());
    let mut cursor = 0usize;

    for replacement in replacements {
        if replacement.modality != Modality::Image {
            bail_multimodal!(
                "unsupported prompt replacement modality `{}`",
                replacement.modality
            );
        }

        let offset = find_next_token(prompt_token_ids, placeholder_marker_token_id, cursor)
            .ok_or_else(|| {
                multimodal!(
                    "placeholder token `{placeholder_token}` was not found in tokenized prompt"
                )
            })?;

        if replacement.tokens.is_empty() {
            bail_multimodal!("placeholder token `{placeholder_token}` expanded to no tokens");
        }

        let replacement_len = replacement.tokens.len();
        let is_embed = {
            let mask = replacement
                .tokens
                .iter()
                .map(|&token| token as u32 == placeholder_embed_token_id)
                .collect::<Vec<_>>();
            WireTensor::from_bool(vec![replacement_len], mask).map_err(Error::Multimodal)?
        };

        expanded.extend_from_slice(&prompt_token_ids[cursor..offset]);
        let expanded_offset = expanded.len();
        expanded.extend(replacement.tokens.into_iter().map(|token| token as u32));
        ranges.push(PlaceholderRange {
            offset: expanded_offset,
            length: replacement_len,
            is_embed: Some(is_embed),
        });
        cursor = offset + 1;
    }

    expanded.extend_from_slice(&prompt_token_ids[cursor..]);
    *prompt_token_ids = expanded;

    Ok(ranges)
}

/// Find `needle` in `haystack`, starting at `start`.
///
/// This is intentionally order-preserving rather than a global replace: each
/// image consumes the next placeholder occurrence.
fn find_next_token(haystack: &[u32], needle: u32, start: usize) -> Option<usize> {
    haystack
        .get(start..)?
        .iter()
        .position(|token| *token == needle)
        .map(|offset| start + offset)
}

/// Adapter from the frontend tokenizer trait to `llm-multimodal`.
#[derive(Clone)]
struct TokenizerResolver(DynTokenizer);

impl TokenResolver for TokenizerResolver {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
        self.0.encode(text, false).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::TokenId;
    use vllm_engine_core_client::protocol::tensor::WireArrayData;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    const LLAMA4_IMAGE_START_ID: u32 = 200088;
    const LLAMA4_IMAGE_END_ID: u32 = 200089;
    const LLAMA4_IMAGE_ID: u32 = 200090;
    const LLAMA4_PATCH_ID: u32 = 200092;
    const LLAMA4_TILE_X_SEPARATOR_ID: u32 = 200093;
    const LLAMA4_TILE_Y_SEPARATOR_ID: u32 = 200094;

    fn llama4_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_start|>", LLAMA4_IMAGE_START_ID)
            .with_regular_token("<|image_end|>", LLAMA4_IMAGE_END_ID)
            .with_regular_token("<|image|>", LLAMA4_IMAGE_ID)
            .with_regular_token("<|patch|>", LLAMA4_PATCH_ID)
            .with_regular_token("<|tile_x_separator|>", LLAMA4_TILE_X_SEPARATOR_ID)
            .with_regular_token("<|tile_y_separator|>", LLAMA4_TILE_Y_SEPARATOR_ID)
    }

    fn test_info(model_type: &str, config: serde_json::Value) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(llama4_tokenizer())),
        };
        let spec = context
            .resolve_model_spec()
            .unwrap_or_else(|| panic!("{model_type} spec should match"));
        let spec = ResolvedMultimodalSpec::new(spec, &context).unwrap();
        let raw_image_processor = context
            .resolve_image_processor()
            .unwrap_or_else(|| panic!("{model_type} image processor should match"));
        let media_connector = Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default()).unwrap(),
        );

        MultimodalModelInfo {
            context,
            spec,
            image_processor: ResolvedImageProcessor {
                raw: raw_image_processor,
                config: PreProcessorConfig::default(),
            },
            media_connector,
        }
    }

    fn llama4_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "llama4",
            "image_token_index": LLAMA4_PATCH_ID,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        test_info("llama4", config)
    }

    fn llama4_single_tile_replacement() -> PromptReplacement {
        PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            vec![
                LLAMA4_IMAGE_START_ID as TokenId,
                LLAMA4_IMAGE_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_IMAGE_END_ID as TokenId,
            ],
        )
    }

    fn llama4_multi_tile_replacement() -> PromptReplacement {
        PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            vec![
                LLAMA4_IMAGE_START_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_TILE_X_SEPARATOR_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_TILE_Y_SEPARATOR_ID as TokenId,
                LLAMA4_IMAGE_ID as TokenId,
                LLAMA4_PATCH_ID as TokenId,
                LLAMA4_IMAGE_END_ID as TokenId,
            ],
        )
    }

    fn assert_bool_mask(range: &PlaceholderRange, expected: &[bool]) {
        let tensor = range.is_embed.as_ref().expect("is_embed mask");
        assert_eq!(tensor.dtype, "bool");
        assert_eq!(tensor.shape, vec![expected.len()]);
        assert_eq!(
            tensor.data,
            WireArrayData::RawView(expected.iter().map(|value| u8::from(*value)).collect())
        );
    }

    #[test]
    fn expand_prompt_tokens_marks_only_llama4_patch_tokens_as_embed() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let replacements = vec![llama4_multi_tile_replacement()];

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_TILE_X_SEPARATOR_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_TILE_Y_SEPARATOR_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                2,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 8);
        assert_bool_mask(
            &ranges[0],
            &[false, true, false, true, false, false, true, false],
        );
    }

    #[test]
    fn expand_prompt_tokens_errors_when_placeholder_missing() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, 2, 3];
        let replacements = vec![llama4_single_tile_replacement()];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
    }

    #[test]
    fn expand_prompt_tokens_ignores_empty_replacements() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, Vec::new()).unwrap();

        assert!(ranges.is_empty());
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_leaves_prompt_unchanged_when_later_placeholder_missing() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let replacements = vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(matches!(error, Error::Multimodal(message) if message.contains("not found")));
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_errors_when_replacement_is_empty() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2];
        let original_prompt_token_ids = prompt_token_ids.clone();
        let replacements = vec![PromptReplacement::sequence(
            Modality::Image,
            "<|image|>",
            Vec::new(),
        )];

        let error = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expanded to no tokens"))
        );
        assert_eq!(prompt_token_ids, original_prompt_token_ids);
    }

    #[test]
    fn expand_prompt_tokens_skips_llama4_image_marker_inside_replacement() {
        let info = llama4_info();
        let mut prompt_token_ids = vec![1, LLAMA4_IMAGE_ID, 2, LLAMA4_IMAGE_ID, 3];
        let replacements = vec![
            llama4_single_tile_replacement(),
            llama4_single_tile_replacement(),
        ];

        let ranges = info.expand_prompt_tokens(&mut prompt_token_ids, replacements).unwrap();

        assert_eq!(
            prompt_token_ids,
            vec![
                1,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                2,
                LLAMA4_IMAGE_START_ID,
                LLAMA4_IMAGE_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_PATCH_ID,
                LLAMA4_IMAGE_END_ID,
                3,
            ]
        );
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 5);
        assert_bool_mask(&ranges[0], &[false, false, true, true, false]);
        assert_eq!(ranges[1].offset, 7);
        assert_eq!(ranges[1].length, 5);
        assert_bool_mask(&ranges[1], &[false, false, true, true, false]);
    }
}
