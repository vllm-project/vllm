// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat-layer multimodal media preparation.
//!
//! This module owns the multimodal path for chat requests: it extracts media
//! parts from structured chat messages, fetches and preprocesses them through
//! `llm-multimodal`, expands rendered prompt placeholders after tokenization,
//! and builds the engine-facing `MmFeatures` payload.
//!
//! Raw media stays above `vllm-text`; this module lowers it into token IDs and
//! opaque tensor payloads before the request is handed to text generation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use itertools::{Either, izip};
use llm_multimodal::registry::ModelRegistryError;
use llm_multimodal::{
    AsyncMultiModalTracker, AudioClip, AudioPreProcessor, EncoderFieldLayouts, FieldLayout,
    FrameSampling, ImageFrame, MediaConnector, MediaConnectorConfig, MediaContentPart,
    MediaItemInfo, Modality, ModelMetadata, ModelProcessorSpec, PreProcessorConfig,
    PreprocessedEncoderInputs, PromptReplacement, Tokenizer as TokenResolver, TrackedMedia,
    VideoClip, VisionPreProcessor, VisionPreprocessingContext,
};
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use tracing::{Instrument as _, warn};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{MmFeatureSpec, MmFeatures, MmKwargsItem};
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use crate::error::{Error, Result, bail_multimodal, multimodal};
use crate::renderer::{MediaPartSource, RenderedPrompt};
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

mod audio;
mod expand;
mod image;
mod input;
mod item;
mod models;
mod preprocessed;
mod tensor;
mod timing;
mod video;

use self::expand::expand_prompt_token_ids;
pub use self::input::MultimodalInput;
use self::timing::MM_STAGE_TARGET;
pub use self::timing::{mm_request_span, mm_timing_layer};

/// Resolved multimodal support for one loaded model.
#[derive(Clone)]
pub struct MultimodalModelInfo {
    context: MultimodalModelContext,
    /// Rendered placeholder marker IDs for all resolved modalities.
    placeholder_token_ids: Vec<u32>,
    image: Option<VisionModalitySupport>,
    video: Option<VisionModalitySupport>,
    audio: Option<AudioModalitySupport>,
    media_connector: Arc<MediaConnector>,
    /// Maximum number of input items allowed per prompt for each modality.
    limit_mm_per_prompt: MmLimitPerPrompt,
}

/// Per-modality item-count limits configured by `--limit-mm-per-prompt`.
///
/// Modalities absent from the map are unlimited.
pub type MmLimitPerPrompt = HashMap<MmModality, MmLimitSpec>;

pub use vllm_engine_core_client::protocol::multimodal::MmModality;

/// One modality's limit, in either of the two shapes Python accepts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    untagged,
    expecting = "an item count, or an object with an optional `count` field"
)]
pub enum MmLimitSpec {
    /// Legacy form: `"image": 16`
    Count(usize),

    /// Configurable form:
    /// `"video": {"count": 1, "num_frames": 32}`
    Options {
        /// Absent means unlimited, matching an absent modality.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        count: Option<usize>,

        /// Preserve Python-owned options for forwarding. Never interpreted
        /// here: they size the engine's dummy-profiling encoder cache, which
        /// has no Rust counterpart.
        #[serde(flatten)]
        extra: BTreeMap<String, serde_json::Value>,
    },
}

impl MmLimitSpec {
    /// The configured item count, or `None` when this modality is unlimited.
    pub fn count(&self) -> Option<usize> {
        match self {
            Self::Count(count) => Some(*count),
            Self::Options { count, .. } => *count,
        }
    }
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
    ///
    /// Local model support ([`models`]) is consulted before the
    /// `llm-multimodal` registry.
    fn resolve_model_spec(&self) -> Option<&'static dyn ModelProcessorSpec> {
        models::lookup_spec(&self.metadata())
    }

    /// Build a vision preprocessor for one loaded model and modality.
    fn resolve_vision_processor(
        &self,
        model_spec: &'static dyn ModelProcessorSpec,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> Result<Option<Arc<dyn VisionPreProcessor>>> {
        match model_spec.vision_processor(&self.config, preprocessor_config, modality) {
            Ok(processor) => Ok(Some(Arc::from(processor))),
            Err(ModelRegistryError::UnsupportedModality { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Resolve an audio preprocessor for one loaded model.
    fn resolve_audio_processor(
        &self,
        model_spec: &'static dyn ModelProcessorSpec,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Option<Arc<dyn AudioPreProcessor>>> {
        Ok(model_spec.audio_processor(&self.config, preprocessor_config)?.map(Arc::from))
    }
}

/// Static model-specific tensor-layout behavior for one modality.
#[derive(Clone)]
struct ResolvedMultimodalSpec {
    raw: &'static dyn ModelProcessorSpec,
    modality: Modality,
    field_layouts: EncoderFieldLayouts,
    keep_on_cpu_keys: HashSet<String>,
    /// Spec-declared wire key for the primary encoder input tensor, when it
    /// differs from the per-modality default (e.g. `"patches"` for
    /// DeepSeek-V4.1 images).
    encoder_input_key: Option<String>,
}

impl ResolvedMultimodalSpec {
    fn new(raw: &'static dyn ModelProcessorSpec, modality: Modality) -> Self {
        Self {
            raw,
            modality,
            field_layouts: raw.encoder_field_layouts_for(modality),
            keep_on_cpu_keys: raw.keep_on_cpu_keys_for(modality).into_iter().collect(),
            encoder_input_key: raw.encoder_input_key_for(modality),
        }
    }

    fn primary_key(&self) -> &str {
        // Models whose forward path reads the primary tensor under a
        // model-specific kwarg name (e.g. MuseSpark's `images` / `videos` /
        // `input_features`) declare it in the spec; everything else keeps the
        // per-modality convention.
        if let Some(key) = &self.encoder_input_key {
            return key;
        }
        match self.modality {
            Modality::Image => image::IMAGE_PRIMARY_KEY,
            Modality::Video => video::VIDEO_PRIMARY_KEY,
            Modality::Audio => audio::AUDIO_PRIMARY_KEY,
            Modality::ImageEmbeds => unreachable!("image embeds use no encoder preprocessor"),
        }
    }

    fn field_layout_for(&self, key: &str) -> Option<&FieldLayout> {
        if key == self.primary_key() {
            Some(&self.field_layouts.encoder_input)
        } else {
            self.field_layouts.model_specific.get(key)
        }
    }

    fn prompt_replacements_for(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> Result<Vec<PromptReplacement>> {
        Ok(self
            .raw
            .prompt_replacements_for(&context.metadata(), preprocessed, self.modality)?)
    }

    /// Prompt replacements given the media items' decoder-side facts (e.g.
    /// how a video's frames were sampled) and the config the modality was
    /// preprocessed with. Specs without a media-aware hook ignore both and
    /// behave exactly like [`Self::prompt_replacements_for`].
    fn prompt_replacements_with_media(
        &self,
        context: &MultimodalModelContext,
        preprocessed: &PreprocessedEncoderInputs,
        media: &[MediaItemInfo],
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Vec<PromptReplacement>> {
        Ok(self.raw.prompt_replacements_with_media(
            &context.metadata(),
            preprocessed,
            self.modality,
            media,
            preprocessor_config,
        )?)
    }
}

/// Resolved placeholder tokens for one modality.
#[derive(Debug, Clone)]
struct ResolvedPlaceholder {
    token: String,
    /// The token IDs emitted for `token` in the rendered prompt.
    marker_token_ids: Vec<u32>,
    /// The model-declared embed token ID marked in `is_embed` masks.
    embed_token_id: u32,
}

impl ResolvedPlaceholder {
    fn resolve(
        raw: &'static dyn ModelProcessorSpec,
        context: &MultimodalModelContext,
        modality: Modality,
    ) -> Result<Self> {
        let metadata = context.metadata();
        let token = raw.placeholder_token_for(&metadata, modality)?;
        // This is the rendered prompt marker. It may be ordinary text that
        // encodes to multiple tokens, while `placeholder_token_id_for()` is
        // the model's replacement vision/patch token ID.
        let marker_token_ids = context
            .tokenizer()
            .token_to_id(&token)
            .map(|token_id| vec![token_id])
            .or_else(|| context.tokenizer().encode(&token, false).ok())
            .filter(|token_ids| !token_ids.is_empty())
            .ok_or_else(|| multimodal!("placeholder `{token}` could not be tokenized"))?;
        let embed_token_id = raw.placeholder_token_id_for(&metadata, modality)? as u32;

        Ok(Self {
            token,
            marker_token_ids,
            embed_token_id,
        })
    }
}

/// Model-owned vision preprocessor plus resolved placeholder tokens and
/// the model's shared tensor-layout spec.
#[derive(Clone)]
struct VisionModalitySupport {
    spec: ResolvedMultimodalSpec,
    placeholder: ResolvedPlaceholder,
    /// Built by the spec with this modality's preprocessor config resolved
    /// into it; reused across requests.
    processor: Arc<dyn VisionPreProcessor>,
    /// The modality's preprocessor config, consulted per request by the
    /// video sampling policy and media-aware prompt replacements.
    config: PreProcessorConfig,
}

/// Model-owned audio preprocessor plus the resolved model contract for its output.
#[derive(Clone)]
struct AudioModalitySupport {
    spec: ResolvedMultimodalSpec,
    placeholder: ResolvedPlaceholder,
    processor: Arc<dyn AudioPreProcessor>,
}

/// Model-repo config file locations consumed by multimodal support.
#[derive(Debug, Default, Clone, Copy)]
pub struct MultimodalConfigFiles<'a> {
    pub config: Option<&'a Path>,
    /// Shared preprocessor config (`preprocessor_config.json`), consumed by
    /// image/audio processors and used as the video fallback.
    pub preprocessor_config: Option<&'a Path>,
    /// Video-specific preprocessor config (`video_preprocessor_config.json`).
    pub video_preprocessor_config: Option<&'a Path>,
    /// Combined processor config (`processor_config.json`), whose modality
    /// sections take precedence over legacy standalone files.
    pub processor_config: Option<&'a Path>,
}

/// Load a modality's section from the combined processor config, falling back
/// to its legacy standalone file.
fn load_preprocessor_config(
    dedicated_path: Option<&Path>,
    dedicated_name: &str,
    processor_config_path: Option<&Path>,
    processor_section: &str,
) -> Result<Option<PreProcessorConfig>> {
    if let Some(path) = processor_config_path {
        let text = fs::read_to_string(path)
            .map_err(|error| multimodal!("failed to read processor_config.json: {error}"))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| multimodal!("failed to parse processor_config.json: {error}"))?;
        if let Some(processor) = value.get(processor_section).filter(|value| !value.is_null()) {
            let config = PreProcessorConfig::from_value(processor.clone()).map_err(|error| {
                multimodal!(
                    "failed to parse {processor_section} from processor_config.json: {error}"
                )
            })?;
            return Ok(Some(config));
        }
    }
    let Some(path) = dedicated_path else {
        return Ok(None);
    };
    let text = fs::read_to_string(path)
        .map_err(|error| multimodal!("failed to read {dedicated_name}: {error}"))?;
    let config = PreProcessorConfig::from_json(&text)
        .map_err(|error| multimodal!("failed to parse {dedicated_name}: {error}"))?;
    Ok(Some(config))
}

/// Request-scoped fetched media, split per modality with tracker UUID
/// metadata preserved in request order.
struct FetchedMedia {
    images: Vec<Arc<ImageFrame>>,
    image_uuids: Vec<Option<String>>,
    videos: Vec<Arc<VideoClip>>,
    video_uuids: Vec<Option<String>>,
    audios: Vec<Arc<AudioClip>>,
    audio_uuids: Vec<Option<String>>,
}

/// One modality's preprocessed output, ready for the shared expansion and
/// feature-assembly tail.
struct PreparedMedia {
    modality: Modality,
    placeholder: ResolvedPlaceholder,
    /// One replacement per media item, in request order.
    replacements: Vec<PromptReplacement>,
    /// One entry per media item, aligned with `replacements`.
    items: Vec<PreparedItem>,
}

/// One media item's complete engine kwargs plus identity metadata.
struct PreparedItem {
    data: MmKwargsItem,
    hash: String,
    uuid: Option<String>,
}

impl MultimodalModelInfo {
    /// Load and resolve multimodal support from model files.
    ///
    /// Returns `Ok(Some(_))` only when the model spec is registered and at
    /// least one modality resolves. File read/parse failures are real errors;
    /// unsupported model families are logged and returned as `Ok(None)`.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        files: MultimodalConfigFiles<'_>,
        tokenizer: DynTokenizer,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<Self>> {
        let config = match files.config {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .map_err(|error| multimodal!("failed to read config.json: {error}"))?;
                serde_json::from_str(&text)
                    .map_err(|error| multimodal!("failed to parse config.json: {error}"))?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let preprocessor_config = load_preprocessor_config(
            files.preprocessor_config,
            "preprocessor_config.json",
            files.processor_config,
            "image_processor",
        )?
        .unwrap_or_default();
        let video_preprocessor_config = load_preprocessor_config(
            files.video_preprocessor_config,
            "video_preprocessor_config.json",
            files.processor_config,
            "video_processor",
        )?
        .unwrap_or_else(|| preprocessor_config.clone());

        let audio_preprocessor_config = load_preprocessor_config(
            files.preprocessor_config,
            "preprocessor_config.json",
            files.processor_config,
            "feature_extractor",
        )?
        .unwrap_or_else(|| preprocessor_config.clone());

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        Self::from_loaded(
            context,
            preprocessor_config,
            video_preprocessor_config,
            audio_preprocessor_config,
            limit_mm_per_prompt,
        )
    }

    /// Resolve multimodal support from an assembled context and parsed
    /// preprocessor configs.
    fn from_loaded(
        context: MultimodalModelContext,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: PreProcessorConfig,
        audio_preprocessor_config: PreProcessorConfig,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<Self>> {
        let (image, video) =
            Self::resolve_vision_lanes(&context, preprocessor_config, video_preprocessor_config)?;
        let audio = Self::resolve_audio_lane(&context, &audio_preprocessor_config)?;

        if image.is_none() && video.is_none() && audio.is_none() {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "no multimodal modality resolved; disabling multimodal support for this model"
            );
            return Ok(None);
        }

        let media_connector = Arc::new(MediaConnector::new(
            // The connector finishes the build (forcing its redirect policy
            // and DNS resolver). llm-multimodal enables no reqwest TLS
            // backend; this workspace's `native-tls` feature provides it.
            reqwest_0_13::Client::builder(),
            MediaConnectorConfig::default(),
        )?);

        let placeholder_token_ids = [
            image.as_ref().map(|support| support.placeholder.marker_token_ids.as_slice()),
            video.as_ref().map(|support| support.placeholder.marker_token_ids.as_slice()),
            audio.as_ref().map(|support| support.placeholder.marker_token_ids.as_slice()),
        ]
        .into_iter()
        .flatten()
        .flatten()
        .copied()
        .collect();

        Ok(Some(Self {
            context,
            placeholder_token_ids,
            image,
            video,
            audio,
            media_connector,
            limit_mm_per_prompt,
        }))
    }

    /// Resolve the spec-backed vision lanes (image and video).
    ///
    /// A missing model spec or vision processor disables both lanes but is
    /// not an error: unsupported model families are logged and skipped.
    /// Processor construction errors propagate to model loading.
    fn resolve_vision_lanes(
        context: &MultimodalModelContext,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: PreProcessorConfig,
    ) -> Result<(Option<VisionModalitySupport>, Option<VisionModalitySupport>)> {
        let Some(raw_spec) = context.resolve_model_spec() else {
            warn!(
                model_id = context.model_id,
                model_type = context.model_type,
                "multimodal model spec is not registered; disabling image/video support for this model"
            );
            return Ok((None, None));
        };

        // Warn and disable the modality if the placeholder resolution fails.
        let resolve_placeholder =
            |modality: Modality| match ResolvedPlaceholder::resolve(raw_spec, context, modality) {
                Ok(placeholder) => Some(placeholder),
                Err(error) => {
                    warn!(
                        model_id = context.model_id,
                        %modality,
                        error = %error.as_report(),
                        "placeholder tokens did not resolve; disabling this modality for this model"
                    );
                    None
                }
            };

        let image = if let Some(placeholder) = resolve_placeholder(Modality::Image) {
            context
                .resolve_vision_processor(raw_spec, &preprocessor_config, Modality::Image)?
                .map(|processor| VisionModalitySupport {
                    spec: ResolvedMultimodalSpec::new(raw_spec, Modality::Image),
                    placeholder,
                    processor,
                    config: preprocessor_config,
                })
        } else {
            None
        };

        let video = if let Some(placeholder) = resolve_placeholder(Modality::Video) {
            // Placeholder expansion attributes marker sequences to modalities,
            // so a marker shared with the image modality is ambiguous.
            let image_marker = image.as_ref().map(|image| &image.placeholder.marker_token_ids);
            if image_marker == Some(&placeholder.marker_token_ids) {
                warn!(
                    model_id = context.model_id,
                    token = placeholder.token,
                    "video placeholder token collides with the image placeholder; disabling video support for this model"
                );
                None
            } else {
                context
                    .resolve_vision_processor(
                        raw_spec,
                        &video_preprocessor_config,
                        Modality::Video,
                    )?
                    .map(|processor| VisionModalitySupport {
                        spec: ResolvedMultimodalSpec::new(raw_spec, Modality::Video),
                        placeholder,
                        processor,
                        config: video_preprocessor_config,
                    })
            }
        } else {
            None
        };

        Ok((image, video))
    }

    /// Resolve a spec-backed audio lane when the model advertises audio support.
    fn resolve_audio_lane(
        context: &MultimodalModelContext,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Option<AudioModalitySupport>> {
        let Some(raw_spec) = context.resolve_model_spec() else {
            return Ok(None);
        };
        if !raw_spec.modality_limits(&context.metadata())?.contains_key(&Modality::Audio) {
            return Ok(None);
        }

        let Some(processor) = context.resolve_audio_processor(raw_spec, preprocessor_config)?
        else {
            warn!(
                model_id = context.model_id,
                model_spec = raw_spec.name(),
                "model spec does not provide an audio processor; disabling audio support for this model"
            );
            return Ok(None);
        };
        Ok(Some(AudioModalitySupport {
            spec: ResolvedMultimodalSpec::new(raw_spec, Modality::Audio),
            placeholder: ResolvedPlaceholder::resolve(raw_spec, context, Modality::Audio)?,
            processor,
        }))
    }

    /// Return the template-visible placeholder token for one modality, when
    /// this model supports it.
    ///
    /// The HF renderer uses these tokens while flattening media content in
    /// string content format.
    pub fn placeholder_token(&self, modality: Modality) -> Option<&str> {
        match modality {
            Modality::Image => self.image.as_ref()?.placeholder.token.as_str().into(),
            Modality::Video => self.video.as_ref()?.placeholder.token.as_str().into(),
            Modality::Audio => self.audio.as_ref()?.placeholder.token.as_str().into(),
            Modality::ImageEmbeds => None,
        }
    }

    fn vision_preprocessing_context(
        &self,
        prompt_token_ids: &[u32],
        max_model_len: Option<usize>,
    ) -> VisionPreprocessingContext {
        let text_prompt_length = prompt_token_ids
            .iter()
            .filter(|token_id| !self.placeholder_token_ids.contains(token_id))
            .count();
        VisionPreprocessingContext {
            token_budget: max_model_len.map(|limit| limit.saturating_sub(text_prompt_length)),
        }
    }
}

/// Finalize a rendered chat prompt into text-generation input.
///
/// Text-only requests pass through unchanged as `Prompt::Text`. Multimodal
/// requests are tokenized in chat, their media placeholders are expanded, and
/// preprocessed media features are attached for engine-core transport.
pub(crate) async fn finalize_rendered_prompt(
    request: &ChatRequest,
    rendered: RenderedPrompt,
    info: Option<&MultimodalModelInfo>,
    model_dtype: ModelDtype,
    max_model_len: Option<usize>,
) -> Result<(Prompt, Option<MmFeatures>)> {
    let media_parts = extract_media_parts(request, rendered.media_order.as_deref())?;
    if media_parts.is_empty() {
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
    let prepared = info
        .prepare_multimodal(
            media_parts,
            &mut prompt_token_ids,
            model_dtype,
            max_model_len,
        )
        .instrument(mm_request_span(&request.request_id))
        .await?;

    Ok((Prompt::TokenIds(prompt_token_ids), Some(prepared)))
}

/// Resolve media parts in the placeholder order reported by the renderer.
fn extract_media_parts(
    request: &ChatRequest,
    media_order: Option<&[MediaPartSource]>,
) -> Result<Vec<MediaContentPart>> {
    let parts = match media_order {
        // Resolve exactly the sources selected by the renderer, in placeholder order.
        Some(media_order) => Either::Left(media_order.iter().map(|source| -> Result<_> {
            let message = request.messages.get(source.message_index).ok_or_else(|| {
                multimodal!(
                    "renderer reported missing chat message {} for multimodal content",
                    source.message_index
                )
            })?;
            let content = match message {
                ChatMessage::System { content }
                | ChatMessage::Developer { content, .. }
                | ChatMessage::User { content }
                | ChatMessage::ToolResponse { content, .. }
                | ChatMessage::Custom { content, .. } => content,
                ChatMessage::Assistant { .. } => {
                    bail_multimodal!("renderer reported multimodal assistant content")
                }
            };
            let ChatContent::Parts(parts) = content else {
                bail_multimodal!("renderer reported multimodal content in a text-only chat message")
            };
            let part = parts.get(source.content_part_index).ok_or_else(|| {
                multimodal!(
                    "renderer reported missing content part {} in chat message {}",
                    source.content_part_index,
                    source.message_index
                )
            })?;
            Ok(part)
        })),
        // Fall back to all media in request message and content-part order (e.g. Jinja).
        None => Either::Right(request.messages.iter().flat_map(|message| {
            let content = match message {
                ChatMessage::System { content }
                | ChatMessage::Developer { content, .. }
                | ChatMessage::User { content }
                | ChatMessage::ToolResponse { content, .. }
                | ChatMessage::Custom { content, .. } => Some(content),
                ChatMessage::Assistant { .. } => None,
            };
            match content {
                Some(ChatContent::Parts(parts)) => parts.as_slice(),
                _ => &[],
            }
            .iter()
            .filter(|part| part.is_multimodal())
            .map(Ok)
        })),
    };
    parts
        .map(|part| match part? {
            ChatContentPart::Text { .. } => {
                bail_multimodal!("renderer reported a text part as multimodal content")
            }
            ChatContentPart::ImageUrl {
                image_url,
                detail,
                uuid,
            } => Ok(MediaContentPart::ImageUrl {
                url: image_url.clone(),
                detail: *detail,
                uuid: uuid.clone(),
                max_long_side_pixel: None,
            }),
            ChatContentPart::VideoUrl { video_url, uuid } => Ok(MediaContentPart::VideoUrl {
                url: video_url.clone(),
                uuid: uuid.clone(),
                fps: None,
                max_long_side_pixel: None,
            }),
            ChatContentPart::InputAudio { data, format, uuid } => Ok(MediaContentPart::AudioUrl {
                url: input_audio_data_url(data, format.as_deref())?,
                uuid: uuid.clone(),
            }),
            ChatContentPart::AudioUrl { audio_url, uuid } => Ok(MediaContentPart::AudioUrl {
                url: audio_url.clone(),
                uuid: uuid.clone(),
            }),
        })
        .collect()
}

/// Wrap OpenAI base64 audio in a data URL consumed by `MediaConnector`.
fn input_audio_data_url(data: &str, format: Option<&str>) -> Result<String> {
    let mime_type = match format {
        Some(format) if format.eq_ignore_ascii_case("wav") => "audio/wav",
        Some(format) if format.eq_ignore_ascii_case("mp3") => "audio/mpeg",
        Some(format) => bail_multimodal!("unsupported input_audio format `{format}`"),
        None => "application/octet-stream",
    };
    Ok(format!("data:{mime_type};base64,{data}"))
}

/// The modality a content part counts against, or `None` for plain text.
///
/// Embedding inputs share their base modality's budget rather than getting one
/// of their own, matching Python's `modality.replace("_embeds", "")` in
/// `vllm/renderers/chat_utils.py`.
fn media_part_limit_modality(part: &MediaContentPart) -> Option<MmModality> {
    match part {
        MediaContentPart::Text { .. } => None,
        MediaContentPart::ImageUrl { .. }
        | MediaContentPart::ImageData { .. }
        | MediaContentPart::ImageEmbeds { .. } => Some(MmModality::Image),
        MediaContentPart::AudioUrl { .. } | MediaContentPart::AudioData { .. } => {
            Some(MmModality::Audio)
        }
        MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
            Some(MmModality::Video)
        }
    }
}

impl MultimodalModelInfo {
    /// Reject requests exceeding `--limit-mm-per-prompt`'s configured
    /// per-modality item count, before any fetch/decode work is spent on them.
    ///
    /// Modalities without a configured count are unlimited.
    fn validate_mm_limits(&self, media_parts: &[MediaContentPart]) -> Result<()> {
        self.validate_modality_limits(media_parts.iter().filter_map(media_part_limit_modality))
    }

    fn validate_modality_limits(
        &self,
        modalities: impl IntoIterator<Item = MmModality>,
    ) -> Result<()> {
        let mut counts: HashMap<MmModality, usize> = HashMap::new();
        for modality in modalities {
            *counts.entry(modality).or_default() += 1;
        }

        for (modality, count) in counts {
            let Some(limit) = self.limit_mm_per_prompt.get(&modality).and_then(MmLimitSpec::count)
            else {
                continue;
            };
            if count > limit {
                return Err(Error::MmLimitExceeded {
                    modality: modality.as_str().to_string(),
                    limit,
                });
            }
        }

        Ok(())
    }

    /// Validate inline storage, batching, and placeholder ranges, then check
    /// this model's supported modalities and item-count limits.
    ///
    /// `prompt_len` must include all expanded multimodal placeholders.
    pub(crate) fn prepare_preprocessed(
        &self,
        mut features: MmFeatures,
        prompt_len: usize,
    ) -> Result<MmFeatures> {
        preprocessed::validate_features(&mut features, prompt_len)?;
        for feature in &features {
            let supported = match feature.modality {
                MmModality::Image => self.image.is_some(),
                MmModality::Video => self.video.is_some(),
                MmModality::Audio => self.audio.is_some(),
            };
            if !supported {
                return Err(Error::UnsupportedModality {
                    modality: feature.modality.as_str().to_owned(),
                });
            }
        }
        self.validate_modality_limits(features.iter().map(|feature| feature.modality))?;
        Ok(features)
    }

    /// Run media fetch, per-modality preprocessing, prompt expansion, and
    /// feature build.
    ///
    /// `prompt_token_ids` is mutated in place because placeholder expansion
    /// changes both the final prompt and the offsets recorded in
    /// `PlaceholderRange`.
    ///
    /// `max_model_len`, when known, is the token budget image processors may
    /// size the prompt's images by, less its text tokens. The expanded
    /// prompt's length is validated against `max_model_len` downstream, when
    /// text generation resolves the request's max tokens.
    pub(crate) async fn prepare_multimodal(
        &self,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
        model_dtype: ModelDtype,
        max_model_len: Option<usize>,
    ) -> Result<MmFeatures> {
        if media_parts.is_empty() {
            return Ok(Vec::new());
        }
        self.prepare_multimodal_timed(media_parts, prompt_token_ids, model_dtype, max_model_len)
            .await
    }

    /// Timed body of [`Self::prepare_multimodal`]; each stage is a `tracing`
    /// span aggregated per request by the `vllm-tracing` timing layer.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "preprocessor_total")
    )]
    async fn prepare_multimodal_timed(
        &self,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
        model_dtype: ModelDtype,
        max_model_len: Option<usize>,
    ) -> Result<MmFeatures> {
        let media_parts_len = media_parts.len();
        self.validate_mm_limits(&media_parts)?;
        let vision_context = self.vision_preprocessing_context(prompt_token_ids, max_model_len);
        let fetched = self.fetch_media(media_parts).await?;

        let mut prepared = Vec::new();
        if !fetched.images.is_empty() {
            prepared.push(
                self.prepare_images(
                    fetched.images,
                    fetched.image_uuids,
                    model_dtype,
                    vision_context,
                )
                .await?,
            );
        }
        if !fetched.videos.is_empty() {
            prepared
                .push(self.prepare_videos(fetched.videos, fetched.video_uuids, model_dtype).await?);
        }
        if !fetched.audios.is_empty() {
            prepared.push(self.prepare_audios(fetched.audios, fetched.audio_uuids).await?);
        }

        let mut ranges = expand_prompt_token_ids(prompt_token_ids, &prepared)?;

        let mut features = Vec::with_capacity(media_parts_len);
        for media in prepared {
            let media_ranges = ranges.remove(&media.modality).unwrap_or_default();
            if media_ranges.len() != media.items.len() {
                bail_multimodal!(
                    "number of expanded `{}` placeholders {} does not match number of media items {}",
                    media.modality,
                    media_ranges.len(),
                    media.items.len()
                );
            }
            for (item, range) in izip!(media.items, media_ranges) {
                features.push(MmFeatureSpec {
                    data: Some(item.data),
                    modality: match media.modality {
                        Modality::Image | Modality::ImageEmbeds => MmModality::Image,
                        Modality::Video => MmModality::Video,
                        Modality::Audio => MmModality::Audio,
                    },
                    identifier: item.uuid.unwrap_or_else(|| item.hash.clone()),
                    mm_position: range,
                    mm_hash: Some(item.hash),
                });
            }
        }
        // Mirror the Python frontend (`argsort_mm_positions`): features are
        // ordered by their placeholder position in the prompt.
        features.sort_by_key(|feature| feature.mm_position.offset);

        if features.len() != media_parts_len {
            bail_multimodal!(
                "number of built multimodal features {} does not match number of media parts {}",
                features.len(),
                media_parts_len
            );
        }
        Ok(features)
    }

    /// The resolved video spec's sampling policy for the media tracker: how
    /// the connector picks a clip's frames, and the frame rate a request
    /// without `fps` is sampled at — both read from the video preprocessor
    /// config where the model's reference processor reads them (e.g.
    /// GLM-5.3-Flash keeps 32 evenly-spread frames and resamples at the
    /// configured `fps`). `None` when the model has no video lane; the
    /// tracker then keeps its connector defaults.
    fn video_sampling_policy(&self) -> Option<(FrameSampling, Option<f64>)> {
        let video = self.video.as_ref()?;
        let config = &video.config;
        Some((
            video.spec.raw.video_frame_sampling_with(Some(config)),
            video.spec.raw.default_video_sample_fps_with(Some(config)),
        ))
    }

    /// Fetch all connector-backed media parts and split them per modality,
    /// preserving their request-order UUID metadata.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "media_fetch")
    )]
    async fn fetch_media(&self, media_parts: Vec<MediaContentPart>) -> Result<FetchedMedia> {
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector));
        if let Some((sampling, default_fps)) = self.video_sampling_policy() {
            tracker = tracker
                .with_video_frame_sampling(sampling)
                .with_default_video_sample_fps(default_fps);
        }
        for part in media_parts {
            tracker.push_part(part)?;
        }

        let mut tracker_output = tracker.finalize().await?;

        let images = tracker_output
            .data
            .remove(&Modality::Image)
            .unwrap_or_default()
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Image(frame) => Ok(frame),
                _ => Err(multimodal!(
                    "tracker returned non-image media for the image modality"
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let image_uuids = tracker_output.uuids.remove(&Modality::Image).unwrap_or_default();

        let videos = tracker_output
            .data
            .remove(&Modality::Video)
            .unwrap_or_default()
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Video(clip) => Ok(clip),
                _ => Err(multimodal!(
                    "tracker returned non-video media for the video modality"
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let video_uuids = tracker_output.uuids.remove(&Modality::Video).unwrap_or_default();

        let audios = tracker_output
            .data
            .remove(&Modality::Audio)
            .unwrap_or_default()
            .into_iter()
            .map(|media| match media {
                TrackedMedia::Audio(clip) => Ok(clip),
                _ => Err(multimodal!(
                    "tracker returned non-audio media for the audio modality"
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_uuids = tracker_output.uuids.remove(&Modality::Audio).unwrap_or_default();

        Ok(FetchedMedia {
            images,
            image_uuids,
            videos,
            video_uuids,
            audios,
            audio_uuids,
        })
    }
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

    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    pub(super) const LLAMA4_IMAGE_START_ID: u32 = 200088;
    pub(super) const LLAMA4_IMAGE_END_ID: u32 = 200089;
    pub(super) const LLAMA4_IMAGE_ID: u32 = 200090;
    pub(super) const LLAMA4_PATCH_ID: u32 = 200092;
    pub(super) const LLAMA4_TILE_X_SEPARATOR_ID: u32 = 200093;
    pub(super) const LLAMA4_TILE_Y_SEPARATOR_ID: u32 = 200094;

    pub(super) const QWEN3_IMAGE_PAD_ID: u32 = 151655;
    pub(super) const QWEN3_VIDEO_PAD_ID: u32 = 151656;

    pub(super) const STEP3_IM_PATCH_ID: u32 = 151680;
    pub(super) const STEP3_IM_START_ID: u32 = 151681;
    pub(super) const STEP3_IM_END_ID: u32 = 151682;
    pub(super) const STEP3_PATCH_START_ID: u32 = 151683;
    pub(super) const STEP3_PATCH_END_ID: u32 = 151684;
    pub(super) const STEP3_PATCH_NEWLINE_ID: u32 = 151685;

    pub(super) const DEEPSEEK_V41_IMAGE_ID: u32 = 129264;
    pub(super) const DEEPSEEK_V41_IMAGE_PAD_ID: u32 = 129265;

    fn llama4_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_start|>", LLAMA4_IMAGE_START_ID)
            .with_regular_token("<|image_end|>", LLAMA4_IMAGE_END_ID)
            .with_regular_token("<|image|>", LLAMA4_IMAGE_ID)
            .with_regular_token("<|patch|>", LLAMA4_PATCH_ID)
            .with_regular_token("<|tile_x_separator|>", LLAMA4_TILE_X_SEPARATOR_ID)
            .with_regular_token("<|tile_y_separator|>", LLAMA4_TILE_Y_SEPARATOR_ID)
    }

    pub(super) fn qwen3_vl_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image_pad|>", QWEN3_IMAGE_PAD_ID)
            .with_regular_token("<|video_pad|>", QWEN3_VIDEO_PAD_ID)
    }

    /// Step5-VL reuses Step3-VL's six multimodal special tokens.
    pub(super) fn step3_vl_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<im_patch>", STEP3_IM_PATCH_ID)
            .with_regular_token("<im_start>", STEP3_IM_START_ID)
            .with_regular_token("<im_end>", STEP3_IM_END_ID)
            .with_regular_token("<patch_start>", STEP3_PATCH_START_ID)
            .with_regular_token("<patch_end>", STEP3_PATCH_END_ID)
            .with_regular_token("<patch_newline>", STEP3_PATCH_NEWLINE_ID)
    }

    fn test_info_with_limits(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            limit_mm_per_prompt,
        )
        .unwrap()
        .unwrap_or_else(|| panic!("{model_type} multimodal support should resolve"))
    }

    fn test_info(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
    ) -> MultimodalModelInfo {
        test_info_with_limits(model_type, config, tokenizer, HashMap::new())
    }

    fn llama4_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "llama4",
            "image_token_index": LLAMA4_PATCH_ID,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        test_info("llama4", config, llama4_tokenizer())
    }

    fn qwen3_vl_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_vl",
            "image_token_id": QWEN3_IMAGE_PAD_ID,
            "video_token_id": QWEN3_VIDEO_PAD_ID,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        })
    }

    pub(super) fn qwen3_vl_info() -> MultimodalModelInfo {
        test_info("qwen3_vl", qwen3_vl_config(), qwen3_vl_tokenizer())
    }

    pub(super) const GLM53_IMAGE_ID: u32 = 154854;
    pub(super) const GLM53_VIDEO_ID: u32 = 154855;
    pub(super) const GLM53_BEGIN_ID: u32 = 154830;
    pub(super) const GLM53_END_ID: u32 = 154831;

    pub(super) fn glm53_flash_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|image|>", GLM53_IMAGE_ID)
            .with_regular_token("<|video|>", GLM53_VIDEO_ID)
            .with_regular_token("<|begin_of_image|>", GLM53_BEGIN_ID)
            .with_regular_token("<|end_of_image|>", GLM53_END_ID)
    }

    pub(super) fn glm53_flash_video_config() -> PreProcessorConfig {
        PreProcessorConfig::from_value(serde_json::json!({
            "do_resize": true,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "patch_expand_factor": 1,
            "min_image_tokens": 16,
            "max_image_tokens": 240_000,
            "fps": 1
        }))
        .unwrap()
    }

    pub(super) fn glm53_flash_info(video_config: PreProcessorConfig) -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: "zai-org/GLM-5.3-Flash".to_string(),
            model_type: Some("glm53_flash".to_string()),
            config: serde_json::json!({
                "model_type": "glm53_flash",
                "image_token_id": GLM53_IMAGE_ID,
            }),
            tokenizer: TokenizerResolver(Arc::new(glm53_flash_tokenizer())),
        };
        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            video_config,
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("GLM-5.3-Flash multimodal support")
    }

    #[test]
    fn video_sampling_policy_comes_from_the_video_spec() {
        // GLM-5.3-Flash: the GLM-5 family's own loader rule walks the clip
        // at the configured rate (`RateWalk`), and the processor resamples
        // at the video config's `fps` (default 2.0).
        let info = glm53_flash_info(glm53_flash_video_config());
        assert_eq!(
            info.video_sampling_policy(),
            Some((
                FrameSampling::RateWalk {
                    max_frames: 2048,
                    temporal_patch_size: 2
                },
                Some(1.0)
            ))
        );
        let info = glm53_flash_info(PreProcessorConfig::default());
        assert_eq!(
            info.video_sampling_policy(),
            Some((
                FrameSampling::RateWalk {
                    max_frames: 2048,
                    temporal_patch_size: 2
                },
                Some(2.0)
            ))
        );

        // Qwen3-VL declares its family's even-rounded rule and no rate, so a
        // request without `fps` keeps the connector default.
        assert_eq!(
            qwen3_vl_info().video_sampling_policy(),
            Some((FrameSampling::EvenRounded, None))
        );

        // A model without a video lane leaves the tracker at connector
        // defaults.
        assert_eq!(llama4_info().video_sampling_policy(), None);
    }

    #[test]
    fn qwen3_vl_resolves_image_and_video_support() {
        let info = qwen3_vl_info();

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<|image_pad|>")
        );
        assert_eq!(
            info.placeholder_token(Modality::Video),
            Some("<|video_pad|>")
        );
        assert_ne!(
            info.image.as_ref().unwrap().placeholder.marker_token_ids,
            info.video.as_ref().unwrap().placeholder.marker_token_ids,
        );
    }

    #[test]
    fn vision_preprocessing_context_uses_runtime_budget_and_text_tokens() {
        let info = qwen3_vl_info();
        let context = info.vision_preprocessing_context(
            &[11, QWEN3_IMAGE_PAD_ID, 12, QWEN3_VIDEO_PAD_ID, 13],
            Some(4096),
        );

        assert_eq!(context.token_budget, Some(4096 - 3));
        assert_eq!(
            info.vision_preprocessing_context(&[11, 12, 13], Some(2)).token_budget,
            Some(0)
        );
        assert_eq!(
            info.vision_preprocessing_context(&[11], None).token_budget,
            None
        );
    }

    fn nemotron_omni_info() -> MultimodalModelInfo {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<image>", 1018)
            .with_regular_token("<img>", 1019)
            .with_regular_token("</img>", 1020);
        // The upstream spec refuses a config whose geometry differs from the
        // checkpoint family it implements, so the test config spells the real
        // checkpoint values.
        let config = serde_json::json!({
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_config": {
                "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                "video_target_num_patches": 1024,
                "video_temporal_patch_size": 2,
                "video_maintain_aspect_ratio": true
            }
        });
        let context = MultimodalModelContext {
            model_id: "nemotron-omni-test".to_string(),
            model_type: Some("NemotronH_Nano_Omni_Reasoning_V3".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };
        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("Nemotron-H Omni multimodal support")
    }

    #[tokio::test]
    async fn nemotron_preprocessing_builds_engine_features() {
        use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

        let info = nemotron_omni_info();
        let media = MediaContentPart::ImageUrl {
            url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        };
        let mut tokens = vec![11; 100];
        tokens.push(1018);

        let features = info
            .prepare_multimodal(vec![media], &mut tokens, ModelDtype::Float32, None)
            .await
            .unwrap();
        assert_eq!(features.len(), 1);
        let feature = &features[0];
        assert_eq!(
            (feature.mm_position.offset, feature.mm_position.length),
            (100, 258)
        );
        assert_eq!(tokens.len(), 358);
        assert_eq!(tokens[100], 1019);
        assert!(tokens[101..357].iter().all(|&token| token == 1018));
        assert_eq!(tokens[357], 1020);
        let data = feature.data.as_ref().unwrap();
        // The 1x1 image is upscaled to the 1024-patch minimum grid of 16px
        // patches; the 2x2 pixel shuffle turns them into 256 feature tokens.
        assert!(matches!(
            data["pixel_values"].data.as_ref(),
            Some(MmKwargValue::Tensor(tensor)) if tensor.shape == [1024, 3 * 16 * 16]
        ));
        assert!(matches!(
            data["patches_per_image"].data.as_ref(),
            Some(MmKwargValue::Tensor(tensor)) if tensor.shape.is_empty()
        ));
        assert!(matches!(
            data["image_grid_hw"].data.as_ref(),
            Some(MmKwargValue::Tensor(tensor)) if tensor.shape == [2]
        ));
    }

    #[tokio::test]
    async fn nemotron_image_grid_fits_the_vision_token_budget() {
        use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

        /// Prepare a 640x480 image after 100 text tokens, returning its
        /// placeholder length and patch count.
        async fn prepare(
            info: &MultimodalModelInfo,
            max_model_len: Option<usize>,
        ) -> (usize, usize) {
            let media = MediaContentPart::ImageUrl {
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQAAAAAdquFJAAAAPUlEQVR42u3BAQ0AAADCoPdPbQ43oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgzwCX4AABS8wbSgAAAABJRU5ErkJggg==".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            };
            let mut tokens = vec![11; 100];
            tokens.push(1018);
            let features = info
                .prepare_multimodal(vec![media], &mut tokens, ModelDtype::Float32, max_model_len)
                .await
                .unwrap();
            let feature = &features[0];
            let Some(MmKwargValue::Tensor(pixel_values)) =
                feature.data.as_ref().unwrap()["pixel_values"].data.as_ref()
            else {
                panic!("pixel_values should be a tensor");
            };
            (feature.mm_position.length, pixel_values.shape[0])
        }

        let info = nemotron_omni_info();
        // Without a budget, or with one the image fits beside the 100 text
        // tokens, it keeps its 30x40 patch grid: 300 tokens inside `<img>`.
        assert_eq!(prepare(&info, None).await, (302, 1200));
        assert_eq!(prepare(&info, Some(404)).await, (302, 1200));
        // 303 tokens left, one short of those 300 plus the tiler's 4-token
        // reserve: the grid shrinks to 30x38.
        assert_eq!(prepare(&info, Some(403)).await, (287, 1140));
        // No room left: the image still gets the minimum grid of 28x36, as
        // Python's tiler does, and the length check refuses the prompt
        // downstream.
        assert_eq!(prepare(&info, Some(100)).await, (254, 1008));
    }

    #[test]
    fn qwen3_vl_without_video_token_id_disables_video_support_only() {
        let config = serde_json::json!({
            "model_type": "qwen3_vl",
            "image_token_id": QWEN3_IMAGE_PAD_ID,
            "vision_config": {"patch_size": 16}
        });
        let info = test_info("qwen3_vl", config, qwen3_vl_tokenizer());

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<|image_pad|>")
        );
        assert_eq!(info.placeholder_token(Modality::Video), None);
    }

    #[test]
    fn llama4_resolves_image_support_only() {
        let info = llama4_info();

        assert_eq!(info.placeholder_token(Modality::Image), Some("<|image|>"));
        assert_eq!(info.placeholder_token(Modality::Video), None);
    }

    // Token IDs from the Muse-Spark-1.2 checkpoint's `tokenizer_config.json`.
    fn muse_spark_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|audio|>", 200038)
            .with_regular_token("<|image_start|>", 200080)
            .with_regular_token("<|image_end|>", 200081)
            .with_regular_token("<|vid_start|>", 200082)
            .with_regular_token("<|vid_end|>", 200083)
            .with_regular_token("<|vid_frame_separator|>", 200087)
            .with_regular_token("<|image|>", 200090)
            .with_regular_token("<|video|>", 200091)
            .with_regular_token("<|patch|>", 200092)
            .with_regular_token("<|soft_audio_start|>", 201463)
            .with_regular_token("<|soft_audio_end|>", 201464)
    }

    fn muse_spark_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "muse_spark",
            "audio_config": {"patch_sizes": [16, 6, 5, 4]},
        });
        test_info("muse_spark", config, muse_spark_tokenizer())
    }

    #[test]
    fn processor_sections_take_precedence_over_legacy_files() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("preprocessor_config.json");
        let combined = dir.path().join("processor_config.json");
        fs::write(&legacy, r#"{"sampling_rate":8000}"#).unwrap();
        for section in ["image_processor", "video_processor", "feature_extractor"] {
            fs::write(
                &combined,
                serde_json::json!({section: {"sampling_rate":24000}}).to_string(),
            )
            .unwrap();
            let config = load_preprocessor_config(
                Some(&legacy),
                "preprocessor_config.json",
                Some(&combined),
                section,
            )
            .unwrap()
            .unwrap();
            assert_eq!(config.get_extra::<usize>("sampling_rate"), Some(24000));
        }
        fs::write(&combined, "{}").unwrap();
        let config = load_preprocessor_config(
            Some(&legacy),
            "preprocessor_config.json",
            Some(&combined),
            "feature_extractor",
        )
        .unwrap()
        .unwrap();
        assert_eq!(config.get_extra::<usize>("sampling_rate"), Some(8000));
    }

    #[tokio::test]
    async fn muse_spark_combined_processor_config_controls_media_features() {
        use llm_multimodal::{
            AudioSource, VideoSource,
            audio::DecodedAudio,
            types::{DecodedRgbFrame, DecodedRgbVideo},
        };

        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        let processor = dir.path().join("processor_config.json");
        fs::write(
            &config,
            r#"{"model_type":"muse_spark","audio_config":{"patch_sizes":[16,6,5,4]}}"#,
        )
        .unwrap();
        fs::write(
            &processor,
            r#"{
            "image_processor": {"patch_size":14,"merge_size":2},
            "video_processor": {"patch_size":14,"merge_size":2,"temporal_patch_size":2},
            "feature_extractor": {"sampling_rate":8000,"samples_per_token":80}
        }"#,
        )
        .unwrap();
        let info = MultimodalModelInfo::from_paths(
            "generic-checkpoint".to_owned(),
            Some("muse_spark".to_owned()),
            MultimodalConfigFiles {
                config: Some(&config),
                processor_config: Some(&processor),
                preprocessor_config: None,
                video_preprocessor_config: None,
            },
            Arc::new(muse_spark_tokenizer()),
            HashMap::new(),
        )
        .unwrap()
        .unwrap();
        let audio = Arc::new(AudioClip::new(
            bytes::Bytes::new(),
            DecodedAudio {
                samples: vec![0.25; 160],
                sample_rate: 16000,
            },
            AudioSource::InlineBytes,
            "audio".to_owned(),
        ));
        let prepared = info.prepare_audios(vec![audio], vec![None]).await.unwrap();
        assert_eq!(prepared.replacements[0].tokens, [201463, 200038, 201464]);

        let frame_len = 3 * 28 * 28;
        let video = DecodedRgbVideo::new(
            bytes::Bytes::from(vec![128; frame_len * 2]),
            (0..2)
                .map(|index| DecodedRgbFrame {
                    width: 28,
                    height: 28,
                    offset: index * frame_len,
                    len: frame_len,
                })
                .collect(),
        );
        let clip = Arc::new(VideoClip::new_rgb_with_sample_fps(
            video,
            bytes::Bytes::new(),
            VideoSource::InlineBytes,
            "video".to_owned(),
            2.0,
        ));
        let prepared =
            info.prepare_videos(vec![clip], vec![None], ModelDtype::Float32).await.unwrap();
        let token_ids =
            prepared.replacements[0].tokens.iter().map(|&id| id as u32).collect::<Vec<_>>();
        // Two frames uniformly sampled at 2 fps with temporal_patch_size=2
        // form one clip whose first frame is stamped at 0/2.0 = 0.00s.
        assert_eq!(
            info.context.tokenizer().decode(&token_ids, false).unwrap(),
            "<|vid_start|>Time: 0.00s<|video|><|vid_end|>"
        );
    }

    #[test]
    fn muse_spark_resolves_all_modalities_with_model_kwarg_names() {
        let info = muse_spark_info();

        assert_eq!(info.placeholder_token(Modality::Image), Some("<|patch|>"));
        assert_eq!(info.placeholder_token(Modality::Video), Some("<|video|>"));
        assert_eq!(info.placeholder_token(Modality::Audio), Some("<|audio|>"));
        let image = info.image.as_ref().unwrap();
        assert_eq!(image.placeholder.marker_token_ids, vec![200092]);
        assert_eq!(image.placeholder.embed_token_id, 200092); // <|patch|>
        let video = info.video.as_ref().unwrap();
        assert_eq!(video.placeholder.embed_token_id, 200091); // <|video|>
        let audio = info.audio.as_ref().unwrap();
        assert_eq!(audio.placeholder.embed_token_id, 200038); // <|audio|>

        // The MuseSpark engine reads its forward kwargs under model-specific
        // names rather than the per-modality defaults.
        assert_eq!(image.spec.primary_key(), "images");
        assert_eq!(video.spec.primary_key(), "videos");
        assert_eq!(audio.spec.primary_key(), "input_features");
    }

    #[test]
    fn muse_spark_without_audio_encoder_disables_audio_lane_only() {
        let config = serde_json::json!({
            "model_type": "muse_spark",
            "audio_config": null,
        });
        let info = test_info("muse_spark", config, muse_spark_tokenizer());

        assert_eq!(info.placeholder_token(Modality::Image), Some("<|patch|>"));
        assert_eq!(info.placeholder_token(Modality::Video), Some("<|video|>"));
        assert!(info.audio.is_none());
    }

    fn step5_vl_info() -> MultimodalModelInfo {
        let config = serde_json::json!({"model_type": "step5_vl"});
        test_info("step5_vl", config, step3_vl_tokenizer())
    }

    #[test]
    fn step5_vl_resolves_image_support_only() {
        let info = step5_vl_info();

        assert_eq!(info.placeholder_token(Modality::Image), Some("<im_patch>"));
        assert_eq!(info.placeholder_token(Modality::Video), None);
        assert_eq!(info.placeholder_token(Modality::Audio), None);

        let image = info.image.as_ref().expect("image support should resolve");
        // The marker and the embed token are the same `<im_patch>` token; the
        // structural tokens only appear inside the replacement sequence.
        assert_eq!(image.placeholder.marker_token_ids, vec![STEP3_IM_PATCH_ID]);
        assert_eq!(image.placeholder.embed_token_id, STEP3_IM_PATCH_ID);
        assert_eq!(image.processor.model_name(), "step3_vl");
    }

    #[test]
    fn step5_vl_declares_unlimited_image_modality_only() {
        let info = step5_vl_info();
        let raw = info.context.resolve_model_spec().expect("step3_vl spec should resolve");
        let limits = raw.modality_limits(&info.context.metadata()).unwrap();

        assert_eq!(limits.get(&Modality::Image), Some(&usize::MAX));
        assert_eq!(limits.get(&Modality::Video), None);
        assert_eq!(limits.get(&Modality::Audio), None);
    }

    #[test]
    fn step5_text_only_model_does_not_resolve_multimodal_support() {
        let context = MultimodalModelContext {
            model_id: "step5-test".to_string(),
            model_type: Some("step5".to_string()),
            config: serde_json::json!({"model_type": "step5"}),
            tokenizer: TokenizerResolver(Arc::new(TestTokenizer::new())),
        };

        let info = MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .expect("resolution should not error");

        assert!(info.is_none());
    }

    #[test]
    fn deepseek_v41_uses_spec_declared_encoder_input_key() {
        let context = MultimodalModelContext {
            model_id: "deepseek-v4.1-test".to_string(),
            model_type: Some("deepseek_v41".to_string()),
            config: serde_json::json!({"model_type": "deepseek_v41"}),
            tokenizer: TokenizerResolver(Arc::new(TestTokenizer::new())),
        };
        let raw = context.resolve_model_spec().expect("deepseek_v41 spec should resolve");
        let spec = ResolvedMultimodalSpec::new(raw, Modality::Image);

        assert_eq!(spec.primary_key(), "patches");
        assert!(matches!(
            spec.field_layout_for("patches"),
            Some(FieldLayout::Flat { sizes_key }) if sizes_key == "patches_per_image"
        ));
    }

    fn deepseek_v41_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "deepseek_v41",
            "image_token_id": DEEPSEEK_V41_IMAGE_ID,
        });
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<｜deepseek_image｜>", DEEPSEEK_V41_IMAGE_ID)
            .with_regular_token("<|place_holder_mm_span_0436|>", DEEPSEEK_V41_IMAGE_PAD_ID);
        test_info("deepseek_v41", config, tokenizer)
    }

    #[test]
    fn deepseek_v41_resolves_image_support_only() {
        let info = deepseek_v41_info();

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<｜deepseek_image｜>")
        );
        assert_eq!(info.placeholder_token(Modality::Video), None);
        let image = info.image.as_ref().expect("image support");
        assert_eq!(
            image.placeholder.marker_token_ids,
            vec![DEEPSEEK_V41_IMAGE_ID]
        );
        assert_eq!(image.placeholder.embed_token_id, DEEPSEEK_V41_IMAGE_ID);
        // The engine's forward kwargs pop `patches` (not `pixel_values`).
        assert_eq!(image.spec.primary_key(), "patches");
    }

    #[test]
    fn input_audio_uses_connector_data_urls() {
        assert_eq!(
            input_audio_data_url("AAAA", Some("wav")).unwrap(),
            "data:audio/wav;base64,AAAA"
        );
        assert_eq!(
            input_audio_data_url("BBBB", Some("MP3")).unwrap(),
            "data:audio/mpeg;base64,BBBB"
        );
        assert_eq!(
            input_audio_data_url("CCCC", None).unwrap(),
            "data:application/octet-stream;base64,CCCC"
        );
        assert!(input_audio_data_url("AAAA", Some("flac")).is_err());
    }

    #[test]
    fn extract_media_parts_uses_request_order_or_explicit_selection() {
        let request = ChatRequest {
            messages: vec![
                ChatMessage::user(vec![
                    ChatContentPart::text("before image"),
                    ChatContentPart::ImageUrl {
                        image_url: "data:image/png;base64,image-b".to_string(),
                        detail: None,
                        uuid: Some("image-b".to_string()),
                    },
                ]),
                ChatMessage::user(vec![ChatContentPart::ImageUrl {
                    image_url: "data:image/png;base64,image-a".to_string(),
                    detail: None,
                    uuid: Some("image-a".to_string()),
                }]),
                ChatMessage::assistant_text("answer"),
                ChatMessage::user("follow-up"),
            ],
            ..ChatRequest::for_test()
        };
        let media_order = [
            MediaPartSource {
                message_index: 1,
                content_part_index: 0,
            },
            MediaPartSource {
                message_index: 0,
                content_part_index: 1,
            },
        ];

        let selections = [None, Some(&media_order[..]), Some(&[][..])];
        let uuids = selections.map(|order| {
            extract_media_parts(&request, order)
                .unwrap()
                .into_iter()
                .map(|part| match part {
                    MediaContentPart::ImageUrl { uuid, .. } => uuid,
                    _ => panic!("expected image URL"),
                })
                .collect::<Vec<_>>()
        });

        expect_test::expect![[r#"
            [
                [
                    Some(
                        "image-b",
                    ),
                    Some(
                        "image-a",
                    ),
                ],
                [
                    Some(
                        "image-a",
                    ),
                    Some(
                        "image-b",
                    ),
                ],
                [],
            ]
        "#]]
        .assert_debug_eq(&uuids);
    }

    fn image_url_part() -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: "https://example.com/image.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }

    fn qwen3_vl_info_with_limits(limit_mm_per_prompt: MmLimitPerPrompt) -> MultimodalModelInfo {
        test_info_with_limits(
            "qwen3_vl",
            qwen3_vl_config(),
            qwen3_vl_tokenizer(),
            limit_mm_per_prompt,
        )
    }

    #[test]
    fn validate_mm_limits_ignores_text_parts() {
        let info = qwen3_vl_info();
        let parts = vec![
            MediaContentPart::Text {
                text: "hello".to_string(),
            },
            MediaContentPart::Text {
                text: "world".to_string(),
            },
        ];
        assert!(info.validate_mm_limits(&parts).is_ok());
    }

    #[test]
    fn validate_mm_limits_leaves_unconfigured_modalities_unlimited() {
        let info = qwen3_vl_info();
        let parts: Vec<_> = std::iter::repeat_with(image_url_part).take(1_000).collect();

        assert!(info.validate_mm_limits(&parts).is_ok());
    }

    #[test]
    fn validate_mm_limits_enforces_configured_limit_at_the_boundary() {
        let info =
            qwen3_vl_info_with_limits(HashMap::from([(MmModality::Image, MmLimitSpec::Count(1))]));
        assert!(info.validate_mm_limits(&[image_url_part()]).is_ok());

        let error = info.validate_mm_limits(&[image_url_part(), image_url_part()]).unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "At most 1 image(s) may be provided in one prompt."
        );
        // Confirms the HTTP-mapping bug found during implementation stays fixed:
        // this must map to 400, not 500.
        assert!(error.is_request_validation_error());
    }

    #[test]
    fn validate_mm_limits_counts_image_embeds_against_the_image_limit() {
        let info =
            qwen3_vl_info_with_limits(HashMap::from([(MmModality::Image, MmLimitSpec::Count(1))]));
        let image_embeds_part = MediaContentPart::ImageEmbeds {
            payload: serde_json::Value::String("AAAA".to_string()),
            uuid: None,
        };

        let error = info.validate_mm_limits(&[image_url_part(), image_embeds_part]).unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "At most 1 image(s) may be provided in one prompt."
        );
    }

    /// An options object without a `count` carries only profiling keys, which
    /// say nothing about how many items are allowed.
    #[test]
    fn validate_mm_limits_treats_a_count_less_options_object_as_unlimited() {
        let info = qwen3_vl_info_with_limits(HashMap::from([(
            MmModality::Image,
            MmLimitSpec::Options {
                count: None,
                extra: BTreeMap::from([("width".to_string(), serde_json::json!(512))]),
            },
        )]));

        assert!(info.validate_mm_limits(&[image_url_part(), image_url_part()]).is_ok());
    }

    fn parse_limits(json: &str) -> MmLimitPerPrompt {
        serde_json::from_str(json).expect("limit map should parse")
    }

    #[test]
    fn limit_map_parses_both_shapes_python_accepts() {
        let limits = parse_limits(r#"{"image": 16, "video": {"count": 1, "num_frames": 32}}"#);

        assert_eq!(limits[&MmModality::Image].count(), Some(16));
        assert_eq!(limits[&MmModality::Video].count(), Some(1));
        assert_eq!(limits.get(&MmModality::Audio), None);
    }

    #[test]
    fn limit_map_rejects_keys_python_does_not_accept() {
        assert!(serde_json::from_str::<MmLimitPerPrompt>(r#"{"image_embeds": 1}"#).is_err());
    }

    /// Managed mode forwards this map back to Python as JSON, where
    /// `BaseDummyOptions.count` is a non-optional `int` under `extra="forbid"`.
    /// Emitting `"count": null` would make the engine subprocess fail to start.
    #[test]
    fn limit_map_round_trips_without_emitting_a_null_count() {
        let source = r#"{"video":{"num_frames":32}}"#;
        let encoded = serde_json::to_string(&parse_limits(source)).expect("map should serialize");

        assert_eq!(encoded, source);
    }
}
