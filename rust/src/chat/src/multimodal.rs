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
    TransparencyPolicy, VideoClip, VisionPreProcessor, VisionPreprocessingContext,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{Instrument as _, warn};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{MmFeatureSpec, MmFeatures, MmKwargsItem};
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};
use vllm_tokenizer::parse_checkpoint_json;

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
    /// Modalities the engine accepts for this checkpoint but no lane above
    /// serves, so requests carrying them name the Rust frontend.
    unserved_modalities: Vec<MmModality>,
    media_connector: Arc<MediaConnector>,
    /// Maximum number of input items allowed per prompt for each modality:
    /// the `--limit-mm-per-prompt` count, else the model's default, else 999.
    item_limits: HashMap<MmModality, usize>,
}

/// Per-modality item-count limits configured by `--limit-mm-per-prompt`.
///
/// Modalities absent from the map default to the model's default count, else
/// 999 items.
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
        /// Absent means the default limit, matching an absent modality.
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
    /// The configured item count, or `None` when the default limit applies.
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

/// A modality's processor config as the checkpoint writes it, kept as JSON
/// until startup options are merged in, so they use the checkpoint's own key
/// names (`resample`, not the parsed config's `resampling`).
#[derive(Debug, Clone, Default)]
struct CheckpointProcessorConfig {
    /// Where the config came from, for parse errors.
    source: String,
    config: serde_json::Map<String, Value>,
}

/// Load a modality's section from the combined processor config, falling back
/// to its legacy standalone file.
fn load_preprocessor_config(
    dedicated_path: Option<&Path>,
    dedicated_name: &str,
    processor_config_path: Option<&Path>,
    processor_section: &str,
) -> Result<Option<CheckpointProcessorConfig>> {
    if let Some(path) = processor_config_path {
        let text = fs::read_to_string(path)
            .map_err(|error| multimodal!("failed to read processor_config.json: {error}"))?;
        let value: serde_json::Value = parse_checkpoint_json(&text)
            .map_err(|error| multimodal!("failed to parse processor_config.json: {error}"))?;
        if let Some(processor) = value.get(processor_section).filter(|value| !value.is_null()) {
            let source = format!("{processor_section} from processor_config.json");
            let config = serde_json::from_value(processor.clone())
                .map_err(|error| multimodal!("failed to parse {source}: {error}"))?;
            return Ok(Some(CheckpointProcessorConfig { source, config }));
        }
    }
    let Some(path) = dedicated_path else {
        return Ok(None);
    };
    let text = fs::read_to_string(path)
        .map_err(|error| multimodal!("failed to read {dedicated_name}: {error}"))?;
    let config = parse_checkpoint_json(&text)
        .map_err(|error| multimodal!("failed to parse {dedicated_name}: {error}"))?;
    Ok(Some(CheckpointProcessorConfig {
        source: dedicated_name.to_owned(),
        config,
    }))
}

/// Resolve startup options into an effective config.
fn resolve_processor_config_overrides(
    checkpoint: CheckpointProcessorConfig,
    kwargs: &HashMap<String, Value>,
    scope: &str,
) -> Result<PreProcessorConfig> {
    if kwargs.get("text_kwargs").is_some_and(|value| {
        !value.is_null() && !value.as_object().is_some_and(|values| values.is_empty())
    }) {
        bail_multimodal!("mm_processor_kwargs.text_kwargs is not supported");
    }
    let flat = kwargs
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "images_kwargs" | "videos_kwargs" | "audio_kwargs" | "text_kwargs"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    let mut effective = Value::Object(checkpoint.config);
    merge_processor_config(&mut effective, &Value::Object(flat));
    if let Some(scoped) = kwargs.get(scope).filter(|value| !value.is_null()) {
        if !scoped.is_object() {
            bail_multimodal!("mm_processor_kwargs.{scope} must be an object");
        }
        merge_processor_config(&mut effective, scoped);
    }
    PreProcessorConfig::from_value(effective)
        .map_err(|error| multimodal!("failed to parse {}: {error}", checkpoint.source))
}

/// Recursively merge mappings; other values replace the checkpoint value.
fn merge_processor_config(config: &mut Value, overrides: &Value) {
    if let (Value::Object(config), Value::Object(overrides)) = (&mut *config, overrides) {
        for (key, value) in overrides {
            merge_processor_config(config.entry(key.clone()).or_insert(Value::Null), value);
        }
    } else {
        *config = overrides.clone();
    }
}

/// Whether `--limit-mm-per-prompt` disables `modality` altogether.
fn limited_to_zero(limit_mm_per_prompt: &MmLimitPerPrompt, modality: MmModality) -> bool {
    limit_mm_per_prompt.get(&modality).and_then(MmLimitSpec::count) == Some(0)
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
    /// Returns `Ok(None)` for a text-only model: no registered model spec and
    /// no media preprocessor config (or every modality limited to 0), or a
    /// spec that declares no modality for this checkpoint. A model whose media
    /// the engine accepts but the frontend can serve none of (media
    /// preprocessor config but no registered spec, or no declared modality
    /// resolving) is an error, so it fails at launch instead of rejecting
    /// every media request. A declared modality that does not resolve while
    /// another does is warned about and rejected per request. File read/parse
    /// failures are errors too.
    pub fn from_paths(
        model_id: String,
        model_type: Option<String>,
        files: MultimodalConfigFiles<'_>,
        tokenizer: DynTokenizer,
        limit_mm_per_prompt: MmLimitPerPrompt,
        mm_processor_kwargs: &HashMap<String, serde_json::Value>,
    ) -> Result<Option<Self>> {
        let config = match files.config {
            Some(path) => {
                let text = fs::read_to_string(path)
                    .map_err(|error| multimodal!("failed to read config.json: {error}"))?;
                parse_checkpoint_json(&text)
                    .map_err(|error| multimodal!("failed to parse config.json: {error}"))?
            }
            None => serde_json::Value::Object(Default::default()),
        };
        let image_config = load_preprocessor_config(
            files.preprocessor_config,
            "preprocessor_config.json",
            files.processor_config,
            "image_processor",
        )?;
        let video_config = load_preprocessor_config(
            files.video_preprocessor_config,
            "video_preprocessor_config.json",
            files.processor_config,
            "video_processor",
        )?;
        let audio_config = load_preprocessor_config(
            files.preprocessor_config,
            "preprocessor_config.json",
            files.processor_config,
            "feature_extractor",
        )?;
        let has_media_config =
            image_config.is_some() || video_config.is_some() || audio_config.is_some();
        let preprocessor_config = image_config.unwrap_or_default();
        let video_preprocessor_config = video_config.unwrap_or_else(|| preprocessor_config.clone());
        let audio_preprocessor_config = audio_config.unwrap_or_else(|| preprocessor_config.clone());

        let context = MultimodalModelContext {
            model_id,
            model_type,
            config,
            tokenizer: TokenizerResolver(tokenizer),
        };

        let Some(model_spec) = context.resolve_model_spec() else {
            // Every limit at 0 is Python's `--language-model-only`.
            let media_disabled = [MmModality::Image, MmModality::Video, MmModality::Audio]
                .into_iter()
                .all(|modality| limited_to_zero(&limit_mm_per_prompt, modality));
            if has_media_config && !media_disabled {
                return Err(Error::MultimodalModelUnsupported {
                    model_id: context.model_id,
                    model_type: context.model_type.unwrap_or_default(),
                });
            }
            return Ok(None);
        };

        Self::from_loaded(
            context,
            model_spec,
            resolve_processor_config_overrides(
                preprocessor_config,
                mm_processor_kwargs,
                "images_kwargs",
            )?,
            resolve_processor_config_overrides(
                video_preprocessor_config,
                mm_processor_kwargs,
                "videos_kwargs",
            )?,
            resolve_processor_config_overrides(
                audio_preprocessor_config,
                mm_processor_kwargs,
                "audio_kwargs",
            )?,
            limit_mm_per_prompt,
        )
    }

    /// Resolve multimodal support from an assembled context and parsed
    /// preprocessor configs.
    fn from_loaded(
        context: MultimodalModelContext,
        model_spec: &'static dyn ModelProcessorSpec,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: PreProcessorConfig,
        audio_preprocessor_config: PreProcessorConfig,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<Self>> {
        let (image, video) = Self::resolve_vision_lanes(
            &context,
            model_spec,
            preprocessor_config,
            video_preprocessor_config,
        )?;
        let audio = Self::resolve_audio_lane(&context, model_spec, &audio_preprocessor_config)?;

        // The engine accepts the modalities the spec declares for this
        // checkpoint (read from its config), except those limited to 0.
        let mut served = false;
        let mut unserved_modalities = Vec::new();
        for modality in model_spec.modality_limits(&context.metadata())?.into_keys() {
            let (resolved, modality) = match modality {
                Modality::Image => (image.is_some(), MmModality::Image),
                Modality::Video => (video.is_some(), MmModality::Video),
                Modality::Audio => (audio.is_some(), MmModality::Audio),
                Modality::ImageEmbeds => continue,
            };
            if limited_to_zero(&limit_mm_per_prompt, modality) {
                continue;
            }
            if resolved {
                served = true;
            } else {
                unserved_modalities.push(modality);
            }
        }
        if !served && !unserved_modalities.is_empty() {
            return Err(Error::MultimodalModelUnsupported {
                model_id: context.model_id,
                model_type: context.model_type.unwrap_or_default(),
            });
        }
        for modality in &unserved_modalities {
            warn!(
                model_id = context.model_id,
                "the Rust frontend does not support `{modality}` input for this model, and \
                 requests carrying it will be rejected; pass --limit-mm-per-prompt \
                 '{{\"{modality}\": 0}}' to disable it, or set VLLM_USE_RUST_FRONTEND=0 to use \
                 the Python frontend",
                modality = modality.as_str(),
            );
        }

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

        let item_limits = [MmModality::Image, MmModality::Video, MmModality::Audio]
            .into_iter()
            .map(|modality| {
                let limit = (limit_mm_per_prompt.get(&modality))
                    .and_then(MmLimitSpec::count)
                    .or_else(|| models::default_item_limit(&context.metadata(), modality))
                    .unwrap_or(DEFAULT_MM_ITEMS_PER_PROMPT);
                (modality, limit)
            })
            .collect();

        Ok(Some(Self {
            context,
            placeholder_token_ids,
            image,
            video,
            audio,
            unserved_modalities,
            media_connector,
            item_limits,
        }))
    }

    /// The error for a request carrying `modality`, which no lane serves.
    fn unsupported_modality(&self, modality: MmModality) -> Error {
        if self.unserved_modalities.contains(&modality) {
            Error::RustFrontendUnsupportedModality {
                modality: modality.as_str(),
            }
        } else {
            Error::UnsupportedModality {
                modality: modality.as_str().to_owned(),
            }
        }
    }

    /// Resolve the spec-backed vision lanes (image and video).
    ///
    /// A lane whose placeholder or vision processor does not resolve is
    /// disabled here; [`Self::from_loaded`] decides whether that is an error.
    /// Processor construction errors propagate to model loading.
    fn resolve_vision_lanes(
        context: &MultimodalModelContext,
        raw_spec: &'static dyn ModelProcessorSpec,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: PreProcessorConfig,
    ) -> Result<(Option<VisionModalitySupport>, Option<VisionModalitySupport>)> {
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
        raw_spec: &'static dyn ModelProcessorSpec,
        preprocessor_config: &PreProcessorConfig,
    ) -> Result<Option<AudioModalitySupport>> {
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

    /// Rewrite the text a chat template writes for each media item to the
    /// modality's placeholder token, for specs that declare one (Nemotron's
    /// `<video>`). Such text cannot be matched as a multi-token marker: its
    /// tokens depend on the text around it. Like Python, at most one
    /// occurrence per item is rewritten.
    fn rewrite_rendered_markers(
        &self,
        mut prompt: String,
        media_parts: &[MediaContentPart],
    ) -> String {
        let lanes = [
            (
                self.image.as_ref().map(|support| (&support.spec, &support.placeholder)),
                MmModality::Image,
            ),
            (
                self.video.as_ref().map(|support| (&support.spec, &support.placeholder)),
                MmModality::Video,
            ),
            (
                self.audio.as_ref().map(|support| (&support.spec, &support.placeholder)),
                MmModality::Audio,
            ),
        ];
        for (lane, modality) in lanes {
            let Some((spec, placeholder)) = lane else {
                continue;
            };
            let Some(marker) = spec.raw.rendered_marker_for(spec.modality) else {
                continue;
            };
            let items = media_parts
                .iter()
                .filter(|part| media_part_limit_modality(part) == Some(modality))
                .count();
            prompt = prompt.replacen(marker, &placeholder.token, items);
        }
        prompt
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
            .encode(
                &info.rewrite_rendered_markers(prompt, &media_parts),
                request.add_special_tokens,
            )
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
            ChatContentPart::Text { .. } | ChatContentPart::ToolReference { .. } => {
                bail_multimodal!("renderer reported a non-media part as multimodal content")
            }
            ChatContentPart::ImageUrl {
                image_url,
                detail,
                uuid,
            } => Ok(MediaContentPart::ImageUrl {
                url: Arc::new(image_url.clone()),
                detail: *detail,
                uuid: uuid.clone(),
                max_long_side_pixel: None,
            }),
            ChatContentPart::VideoUrl { video_url, uuid } => Ok(MediaContentPart::VideoUrl {
                url: Arc::new(video_url.clone()),
                uuid: uuid.clone(),
                fps: None,
                max_long_side_pixel: None,
            }),
            ChatContentPart::InputAudio { data, format, uuid } => Ok(MediaContentPart::AudioUrl {
                url: Arc::new(input_audio_data_url(data, format.as_deref())?),
                uuid: uuid.clone(),
            }),
            ChatContentPart::AudioUrl { audio_url, uuid } => Ok(MediaContentPart::AudioUrl {
                url: Arc::new(audio_url.clone()),
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
        Some(format) => {
            return Err(Error::UnsupportedInputAudioFormat {
                format: format.to_owned(),
            });
        }
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
        | MediaContentPart::ImageBase64 { .. }
        | MediaContentPart::ImageData { .. }
        | MediaContentPart::ImageEmbeds { .. } => Some(MmModality::Image),
        MediaContentPart::AudioUrl { .. }
        | MediaContentPart::AudioBase64 { .. }
        | MediaContentPart::AudioData { .. } => Some(MmModality::Audio),
        MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
            Some(MmModality::Video)
        }
    }
}

/// Default per-modality item limit applied when `--limit-mm-per-prompt` does
/// not specify a count for a modality. Matches the Python frontend's default.
const DEFAULT_MM_ITEMS_PER_PROMPT: usize = 999;

impl MultimodalModelInfo {
    /// Reject requests exceeding `--limit-mm-per-prompt`'s configured
    /// per-modality item count, before any fetch/decode work is spent on them.
    ///
    /// Modalities without a configured count default to the model's default
    /// count, else 999 items.
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
            let limit = self.item_limits[&modality];
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
                return Err(self.unsupported_modality(feature.modality));
            }
        }
        self.validate_modality_limits(features.iter().map(|feature| feature.modality))?;
        Ok(features)
    }

    /// Prepare raw media placed after already-expanded `features` and append
    /// its features.
    ///
    /// Only the prompt tokens after the last range in `features` are expanded,
    /// so the runs before it are left as they are, like Python's `/generate`
    /// with both `features` and `content_parts`. `features` must lie within
    /// `prompt_token_ids`, as [`Self::prepare_preprocessed`] checks.
    pub(crate) async fn append_multimodal(
        &self,
        features: &mut MmFeatures,
        media_parts: Vec<MediaContentPart>,
        prompt_token_ids: &mut Vec<u32>,
        model_dtype: ModelDtype,
        max_model_len: Option<usize>,
    ) -> Result<()> {
        let start = features
            .iter()
            .map(|feature| feature.mm_position.offset + feature.mm_position.length)
            .max()
            .unwrap_or(0);
        // Item limits count every item of the prompt, not each input kind alone.
        self.validate_modality_limits(
            features
                .iter()
                .map(|feature| feature.modality)
                .chain(media_parts.iter().filter_map(media_part_limit_modality)),
        )?;

        let mut tail = prompt_token_ids.split_off(start);
        // The expanded prefix takes up context the new media cannot use.
        let prepared = self
            .prepare_multimodal(
                media_parts,
                &mut tail,
                model_dtype,
                max_model_len.map(|limit| limit.saturating_sub(start)),
            )
            .await;
        prompt_token_ids.append(&mut tail);
        let appended = prepared.map_err(|source| match start {
            0 => source,
            offset => Error::MediaAfterPreprocessed {
                offset,
                source: Box::new(source),
            },
        })?;

        preprocessed::validate_batching(features.iter().chain(&appended))?;

        features.extend(appended.into_iter().map(|mut feature| {
            feature.mm_position.offset += start;
            feature
        }));
        Ok(())
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
                // Like Python, tag the cache key with its source, so a client
                // `uuid` spelled like another item's content hash cannot hit
                // that item's entries in the engine's caches.
                let cache_key = match item.uuid {
                    Some(uuid) => format!("uuid:{uuid}"),
                    None => format!("content:{}", item.hash),
                };
                features.push(MmFeatureSpec {
                    data: Some(item.data),
                    modality: match media.modality {
                        Modality::Image | Modality::ImageEmbeds => MmModality::Image,
                        Modality::Video => MmModality::Video,
                        Modality::Audio => MmModality::Audio,
                    },
                    identifier: cache_key.clone(),
                    mm_position: range,
                    mm_hash: Some(cache_key),
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
            video.spec.raw.video_frame_sampling_with(config),
            video.spec.raw.default_video_sample_fps_with(config),
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
        // Python's image loader composites transparent pixels over white
        // (`rgba_background_color`), except for a processor that handles the
        // alpha channel itself (Kimi-K3, whose Python renderer keeps it).
        let transparency = match &self.image {
            Some(image) if image.processor.keeps_alpha() => TransparencyPolicy::KeepAlpha,
            _ => TransparencyPolicy::CompositeOverWhite,
        };
        let mut tracker = AsyncMultiModalTracker::new(Arc::clone(&self.media_connector))
            .with_image_transparency(transparency);
        let video_sampling_policy = self.video_sampling_policy();
        if let Some((sampling, default_fps)) = video_sampling_policy {
            tracker = tracker
                .with_video_frame_sampling(sampling)
                .with_default_video_sample_fps(default_fps);
        }
        for mut part in media_parts {
            apply_video_fps(
                &mut part,
                video_sampling_policy.map(|(sampling, _)| sampling),
            )?;
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

/// Check a video part's requested sampling rate against the model's frame
/// sampling, and scope a client uuid by the rate: as in Python, the uuid alone
/// does not identify frames sampled at a per-request rate.
fn apply_video_fps(part: &mut MediaContentPart, sampling: Option<FrameSampling>) -> Result<()> {
    let MediaContentPart::VideoUrl {
        uuid,
        fps: Some(fps),
        ..
    } = part
    else {
        return Ok(());
    };
    if let Some(FrameSampling::UpTo { .. }) = sampling {
        return Err(Error::UnsupportedMultimodalContent(
            "video fps for a model that samples a fixed number of frames",
        ));
    }
    if let Some(uuid) = uuid {
        *uuid = format!("{uuid}@fps={fps}");
    }
    Ok(())
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

    use expect_test::expect;
    use llm_multimodal::TokenId;
    use serde_json::json;
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

    /// Resolve multimodal support as model loading does after its spec lookup
    /// succeeded.
    pub(super) fn from_context(
        context: MultimodalModelContext,
        preprocessor_config: PreProcessorConfig,
        video_preprocessor_config: PreProcessorConfig,
        audio_preprocessor_config: PreProcessorConfig,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<MultimodalModelInfo>> {
        let model_spec = context.resolve_model_spec().expect("model spec should resolve");
        MultimodalModelInfo::from_loaded(
            context,
            model_spec,
            preprocessor_config,
            video_preprocessor_config,
            audio_preprocessor_config,
            limit_mm_per_prompt,
        )
    }

    fn test_context_info(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<MultimodalModelInfo>> {
        let context = MultimodalModelContext {
            model_id: format!("{model_type}-test"),
            model_type: Some(model_type.to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        from_context(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            limit_mm_per_prompt,
        )
    }

    fn test_info_with_limits(
        model_type: &str,
        config: serde_json::Value,
        tokenizer: TestTokenizer,
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> MultimodalModelInfo {
        test_context_info(model_type, config, tokenizer, limit_mm_per_prompt)
            .unwrap()
            .unwrap_or_else(|| panic!("{model_type} multimodal support should resolve"))
    }

    pub(super) fn test_info(
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
        from_context(
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
    fn request_video_fps_scopes_the_uuid_and_needs_a_rate_based_sampling() {
        let part = |fps| MediaContentPart::VideoUrl {
            url: "https://example.com/clip.mp4".to_owned().into(),
            uuid: Some("clip".to_owned()),
            fps,
            max_long_side_pixel: None,
        };
        let results = [
            (part(None), Some(FrameSampling::EvenRounded)),
            (part(Some(1.0)), Some(FrameSampling::EvenRounded)),
            (
                part(Some(1.0)),
                Some(FrameSampling::UpTo { max_frames: 32 }),
            ),
        ]
        .map(|(mut part, sampling)| {
            apply_video_fps(&mut part, sampling)
                .map(|()| part)
                .map_err(|error| error.to_report_string())
        });
        expect_test::expect![[r#"
            [
                Ok(
                    VideoUrl {
                        url: "https://example.com/clip.mp4",
                        uuid: Some(
                            "clip",
                        ),
                        fps: None,
                        max_long_side_pixel: None,
                    },
                ),
                Ok(
                    VideoUrl {
                        url: "https://example.com/clip.mp4",
                        uuid: Some(
                            "clip@fps=1",
                        ),
                        fps: Some(
                            1.0,
                        ),
                        max_long_side_pixel: None,
                    },
                ),
                Err(
                    "unsupported multimodal content: video fps for a model that samples a fixed number of frames",
                ),
            ]
        "#]].assert_debug_eq(&results);
    }

    #[test]
    fn video_sampling_policy_comes_from_the_video_spec() {
        // GLM-5.3-Flash: the GLM-5 family's own loader rule walks the clip
        // at its fixed rate (`RateWalk`); the processor config's `fps` does
        // not change the loader's sampling rate.
        let info = glm53_flash_info(glm53_flash_video_config());
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

    /// A transparent pixel reaches the encoder as white, the background
    /// Python's image loader composites it over.
    #[tokio::test]
    async fn transparent_pixels_reach_the_encoder_as_white() {
        let info = qwen3_vl_info();
        let encoder_input = async |pixel: [u8; 4]| {
            let mut png = Vec::new();
            ::image::RgbaImage::from_pixel(32, 32, ::image::Rgba(pixel))
                .write_to(
                    &mut std::io::Cursor::new(&mut png),
                    ::image::ImageFormat::Png,
                )
                .unwrap();
            let media = MediaContentPart::ImageData {
                data: png,
                mime_type: Some("image/png".to_string()),
                uuid: None,
                detail: None,
            };
            let mut tokens = vec![11, QWEN3_IMAGE_PAD_ID, 12];
            let mut features = info
                .prepare_multimodal(vec![media], &mut tokens, ModelDtype::Float32, None)
                .await
                .unwrap();
            features.remove(0).data
        };

        let transparent = encoder_input([0, 0, 0, 0]).await;
        assert_eq!(transparent, encoder_input([255, 255, 255, 255]).await);
        assert_ne!(transparent, encoder_input([0, 0, 0, 255]).await);
    }

    /// A processor that handles the alpha channel itself gets the image as
    /// decoded; every other one gets it composited over white.
    #[tokio::test]
    async fn image_transparency_follows_the_processor() {
        let mut png = Vec::new();
        ::image::RgbaImage::from_pixel(8, 8, ::image::Rgba([0, 0, 0, 0]))
            .write_to(
                &mut std::io::Cursor::new(&mut png),
                ::image::ImageFormat::Png,
            )
            .unwrap();
        let kimi_k3_info = test_info(
            "kimi_k3",
            serde_json::json!({"model_type": "kimi_k3", "media_placeholder_token_id": 163605}),
            TestTokenizer::new()
                .with_regular_token("<|media_begin|>", 163602)
                .with_regular_token("<|media_content|>", 163603)
                .with_regular_token("<|media_end|>", 163604)
                .with_regular_token("<|media_pad|>", 163605),
        );
        for (info, expected) in [
            (qwen3_vl_info(), TransparencyPolicy::CompositeOverWhite),
            (kimi_k3_info, TransparencyPolicy::KeepAlpha),
        ] {
            let media = MediaContentPart::ImageData {
                data: png.clone(),
                mime_type: Some("image/png".to_string()),
                uuid: None,
                detail: None,
            };
            let fetched = info.fetch_media(vec![media]).await.unwrap();
            assert_eq!(fetched.images[0].transparency(), expected);
        }
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

    pub(super) fn nemotron_omni_info() -> MultimodalModelInfo {
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<image>", 1018)
            .with_regular_token("<img>", 1019)
            .with_regular_token("</img>", 1020)
            .with_regular_token("<SPECIAL_30>", 1030);
        // The fields of the Nemotron 3 Nano Omni config the spec reads.
        let config = serde_json::json!({
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "force_image_size": 512,
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
        from_context(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("Nemotron-H Omni multimodal support")
    }

    const ONE_PIXEL_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    #[tokio::test]
    async fn nemotron_preprocessing_builds_engine_features() {
        use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

        let info = nemotron_omni_info();
        let media = MediaContentPart::ImageUrl {
            url: ONE_PIXEL_PNG.to_string().into(),
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
            data["pixel_values_flat"].data.as_ref(),
            Some(MmKwargValue::Tensor(tensor)) if tensor.shape == [1024, 3 * 16 * 16]
        ));
        assert!(matches!(
            data["imgs_sizes"].data.as_ref(),
            Some(MmKwargValue::List(size)) if size.len() == 2
        ));
        assert!(matches!(
            data["num_tokens_per_image"].data.as_ref(),
            Some(MmKwargValue::Int(256))
        ));
    }

    /// A 640x480 image, which keeps a 30x40 patch grid (300 tokens inside
    /// `<img>`) on Nemotron-H Omni when the token budget allows.
    fn png_640x480() -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQAAAAAdquFJAAAAPUlEQVR42u3BAQ0AAADCoPdPbQ43oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgzwCX4AABS8wbSgAAAABJRU5ErkJggg==".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
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
            let mut tokens = vec![11; 100];
            tokens.push(1018);
            let features = info
                .prepare_multimodal(
                    vec![png_640x480()],
                    &mut tokens,
                    ModelDtype::Float32,
                    max_model_len,
                )
                .await
                .unwrap();
            let feature = &features[0];
            let Some(MmKwargValue::Tensor(pixel_values)) =
                feature.data.as_ref().unwrap()["pixel_values_flat"].data.as_ref()
            else {
                panic!("pixel_values_flat should be a tensor");
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

    /// One flat preprocessing batch is split into per-image engine items,
    /// preserving each image's patch count and identifier.
    #[tokio::test]
    async fn nemotron_images_of_different_sizes_are_separate_items() {
        use vllm_engine_core_client::protocol::multimodal::MmKwargValue;

        let info = nemotron_omni_info();
        let small = MediaContentPart::ImageUrl {
            url: ONE_PIXEL_PNG.to_string().into(),
            detail: None,
            uuid: Some("small".to_string()),
            max_long_side_pixel: None,
        };
        let mut tokens = vec![1018, 11, 1018];
        let features = info
            .prepare_multimodal(
                vec![small, png_640x480()],
                &mut tokens,
                ModelDtype::Float32,
                None,
            )
            .await
            .unwrap();

        let summary = features
            .iter()
            .map(|feature| {
                let Some(MmKwargValue::Tensor(pixel_values)) =
                    feature.data.as_ref().unwrap()["pixel_values_flat"].data.as_ref()
                else {
                    panic!("pixel_values_flat should be a tensor");
                };
                let position = &feature.mm_position;
                (
                    feature.identifier == "uuid:small",
                    position.offset,
                    position.length,
                    pixel_values.shape.clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                (true, 0, 258, vec![1024, 3 * 16 * 16]),
                (false, 259, 302, vec![1200, 3 * 16 * 16]),
            ]
        );
    }

    /// The template's `<video>` text becomes the video placeholder token
    /// before tokenizing, once per video in the request.
    #[test]
    fn rewrites_one_rendered_marker_per_media_item() {
        let info = nemotron_omni_info();
        let video = MediaContentPart::VideoUrl {
            url: "https://example.com/clip.mp4".to_owned().into(),
            uuid: None,
            fps: None,
            max_long_side_pixel: None,
        };
        let prompt = "<image> <video>\nwhat is a <video> tag?".to_string();
        assert_eq!(
            info.rewrite_rendered_markers(prompt.clone(), &[video]),
            "<image> <SPECIAL_30>\nwhat is a <video> tag?"
        );
        assert_eq!(
            info.rewrite_rendered_markers(prompt.clone(), &[png_640x480()]),
            prompt
        );
    }

    /// Prepare a full-size 640x480 image after 100 text tokens, then append
    /// one more image placeholder: the prompt a later turn sends with the
    /// first image as preprocessed features.
    async fn nemotron_prompt_with_preprocessed_image(
        info: &MultimodalModelInfo,
    ) -> (Vec<u32>, MmFeatures) {
        let mut tokens = vec![11; 100];
        tokens.push(1018);
        let features = info
            .prepare_multimodal(vec![png_640x480()], &mut tokens, ModelDtype::Float32, None)
            .await
            .unwrap();
        assert_eq!(tokens.len(), 402);
        tokens.push(1018);
        (tokens, features)
    }

    #[tokio::test]
    async fn append_multimodal_budgets_new_media_after_the_preprocessed_prefix() {
        let info = nemotron_omni_info();
        let (mut tokens, mut features) = nemotron_prompt_with_preprocessed_image(&info).await;

        // 303 tokens left after the 402-token prefix, one short of the full
        // grid plus the tiler's reserve: the new image shrinks to 30x38.
        info.append_multimodal(
            &mut features,
            vec![png_640x480()],
            &mut tokens,
            ModelDtype::Float32,
            Some(402 + 303),
        )
        .await
        .unwrap();

        let positions: Vec<_> = features
            .iter()
            .map(|feature| (feature.mm_position.offset, feature.mm_position.length))
            .collect();
        assert_eq!(positions, [(100, 302), (402, 287)]);
        assert_eq!(tokens.len(), 402 + 287);
    }

    #[tokio::test]
    async fn append_multimodal_rejects_media_batched_unlike_the_preprocessed_features() {
        use vllm_engine_core_client::protocol::multimodal::MmField;

        let info = nemotron_omni_info();
        let (mut tokens, mut features) = nemotron_prompt_with_preprocessed_image(&info).await;
        let data = features[0].data.as_mut().unwrap();
        match &mut data.get_mut("pixel_values_flat").unwrap().field {
            MmField::Flat(field) => field.dim = 1,
            other => panic!("unexpected pixel_values_flat batching: {other:?}"),
        }

        let error = info
            .append_multimodal(
                &mut features,
                vec![png_640x480()],
                &mut tokens,
                ModelDtype::Float32,
                None,
            )
            .await
            .unwrap_err();

        assert_eq!(
            error.to_report_string(),
            r#"invalid inline multimodal features: incompatible batching for image field "pixel_values_flat""#
        );
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

    // Token IDs from the Muse Spark Open checkpoint's `tokenizer_config.json`.
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
    fn model_specific_values_survive_scoped_config_merging() {
        let kwargs = serde_json::from_value(json!({
            "custom_options": {"width": 512, "height": 256},
            "images_kwargs": {"custom_options": {"width": 1024}},
            "audio_kwargs": {"max_samples": 16000},
        }))
        .unwrap();
        let checkpoint = CheckpointProcessorConfig {
            source: "preprocessor_config.json".to_owned(),
            config: serde_json::from_value(json!({"custom_options": {"checkpoint": true}}))
                .unwrap(),
        };
        let config = |scope| {
            resolve_processor_config_overrides(checkpoint.clone(), &kwargs, scope)
                .unwrap()
                .extra
        };
        let configs = json!({
            "audio": config("audio_kwargs"),
            "image": config("images_kwargs"),
            "video": config("videos_kwargs"),
        });
        assert_eq!(
            configs,
            json!({
                "audio": {
                    "custom_options": {"checkpoint": true, "height": 256, "width": 512},
                    "max_samples": 16000,
                },
                "image": {
                    "custom_options": {"checkpoint": true, "height": 256, "width": 1024},
                },
                "video": {
                    "custom_options": {"checkpoint": true, "height": 256, "width": 512},
                },
            })
        );
    }

    /// Options use the HF key names the checkpoint config uses, including
    /// ones the parsed config renames (`resample` → `resampling`).
    #[test]
    fn processor_kwargs_override_checkpoint_keys_by_hf_name() {
        let kwargs = serde_json::from_value(json!({"resample": 3})).unwrap();
        for checkpoint in [json!({"resample": 2}), json!({})] {
            let checkpoint = CheckpointProcessorConfig {
                source: "preprocessor_config.json".to_owned(),
                config: serde_json::from_value(checkpoint).unwrap(),
            };
            let config =
                resolve_processor_config_overrides(checkpoint, &kwargs, "images_kwargs").unwrap();
            assert_eq!(config.resampling, Some(3));
        }
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
            assert_eq!(config.config["sampling_rate"], 24000);
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
        assert_eq!(config.config["sampling_rate"], 8000);
    }

    #[test]
    fn processor_configs_accept_python_json_non_finite_literals() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("preprocessor_config.json");
        let combined = dir.path().join("processor_config.json");
        fs::write(&legacy, r#"{"sampling_rate":8000,"max_value":Infinity}"#).unwrap();
        fs::write(
            &combined,
            r#"{"image_processor":{"sampling_rate":24000,"min_value":-Infinity},"eps":NaN}"#,
        )
        .unwrap();

        let config = load_preprocessor_config(
            Some(&legacy),
            "preprocessor_config.json",
            Some(&combined),
            "image_processor",
        )
        .unwrap()
        .unwrap();
        assert_eq!(config.config["sampling_rate"], 24000);

        let config = load_preprocessor_config(Some(&legacy), "preprocessor_config.json", None, "")
            .unwrap()
            .unwrap();
        assert_eq!(config.config["sampling_rate"], 8000);
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
            &std::collections::HashMap::new(),
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
        let config = serde_json::json!({"model_type": "step5"});
        let info =
            load_from_files(config, &[], HashMap::new()).expect("resolution should not error");

        assert!(info.is_none());
    }

    /// Load multimodal support from a temporary model directory holding
    /// `config.json` and the given `(file name, content)` media config files.
    fn load_from_files(
        config: serde_json::Value,
        media_files: &[(&str, &str)],
        limit_mm_per_prompt: MmLimitPerPrompt,
    ) -> Result<Option<MultimodalModelInfo>> {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        fs::write(&config_path, config.to_string()).unwrap();
        let path_of = |name: &str| {
            let (name, content) = media_files.iter().find(|(file, _)| *file == name)?;
            let path = dir.path().join(name);
            fs::write(&path, content).unwrap();
            Some(path)
        };
        let preprocessor_config = path_of("preprocessor_config.json");
        let processor_config = path_of("processor_config.json");
        MultimodalModelInfo::from_paths(
            "test-model".to_string(),
            config["model_type"].as_str().map(str::to_string),
            MultimodalConfigFiles {
                config: Some(&config_path),
                preprocessor_config: preprocessor_config.as_deref(),
                processor_config: processor_config.as_deref(),
                ..Default::default()
            },
            Arc::new(qwen3_vl_tokenizer()),
            limit_mm_per_prompt,
            &HashMap::new(),
        )
    }

    #[test]
    fn multimodal_model_without_spec_fails_launch_unless_media_is_disabled() {
        let config = serde_json::json!({"model_type": "gemma3", "image_token_index": 262144});
        // Each media section the Rust processors read, standalone or inside
        // the combined processor config.
        let cases = [
            (
                "preprocessor_config.json",
                r#"{"image_processor_type": "Gemma3ImageProcessor"}"#,
            ),
            (
                "processor_config.json",
                r#"{"image_processor": {"image_processor_type": "Gemma3ImageProcessor"}}"#,
            ),
            (
                "processor_config.json",
                r#"{"video_processor": {"video_processor_type": "Qwen2VLVideoProcessor"}}"#,
            ),
            (
                "processor_config.json",
                r#"{"feature_extractor": {"feature_extractor_type": "WhisperFeatureExtractor"}}"#,
            ),
        ];
        // Every modality limited to 0 is `--language-model-only`.
        let disabled = [MmModality::Image, MmModality::Video, MmModality::Audio]
            .map(|modality| (modality, MmLimitSpec::Count(0)));

        for media_file in cases {
            let error = load_from_files(config.clone(), &[media_file], HashMap::new())
                .err()
                .unwrap_or_else(|| panic!("{media_file:?} without a spec should fail to load"));
            expect!["the Rust frontend cannot serve multimodal input for model `test-model` (model_type `gemma3`); pass --language-model-only to serve it as a text-only model, or set VLLM_USE_RUST_FRONTEND=0 to use the Python frontend"].assert_eq(&error.to_report_string());

            let info = load_from_files(
                config.clone(),
                &[media_file],
                HashMap::from(disabled.clone()),
            )
            .unwrap_or_else(|error| panic!("{media_file:?} with all media disabled: {error}"));
            assert!(info.is_none());
        }
    }

    #[test]
    fn processor_config_without_media_sections_is_text_only() {
        let config = serde_json::json!({"model_type": "qwen3"});
        let media_files = [(
            "processor_config.json",
            r#"{"processor_class": "TextOnlyProcessor"}"#,
        )];

        let info = load_from_files(config, &media_files, HashMap::new())
            .expect("a text-only processor config should load");
        assert!(info.is_none());
    }

    #[tokio::test]
    async fn unsupported_declared_modality_is_rejected_per_request() {
        // The checkpoint declares video (`video_token_id`), but the tokenizer
        // has no `<|video_pad|>` placeholder, so only the image lane resolves.
        let tokenizer =
            TestTokenizer::new().with_regular_token("<|image_pad|>", QWEN3_IMAGE_PAD_ID);
        let info = test_info_with_limits("qwen3_vl", qwen3_vl_config(), tokenizer, HashMap::new());
        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<|image_pad|>")
        );

        let error = info
            .prepare_videos(Vec::new(), Vec::new(), ModelDtype::Float32)
            .await
            .err()
            .expect("video input should be rejected");
        expect![[r#"`video` input is not supported by the Rust frontend for this model; set VLLM_USE_RUST_FRONTEND=0 to serve it with the Python frontend, or pass --limit-mm-per-prompt '{"video": 0}' to disable it"#]].assert_eq(&error.to_report_string());

        // The checkpoint does not declare audio, so the model itself does not
        // take it.
        let error = info
            .prepare_audios(Vec::new(), Vec::new())
            .await
            .err()
            .expect("audio input should be rejected");
        expect!["`audio` input is not supported by this model"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn spec_without_resolved_modality_fails_launch_unless_limited_to_zero() {
        // Neither placeholder is in the tokenizer, so no declared modality
        // resolves.
        let load = |limit_mm_per_prompt| {
            test_context_info(
                "qwen3_vl",
                qwen3_vl_config(),
                TestTokenizer::new(),
                limit_mm_per_prompt,
            )
        };

        let error = load(HashMap::new())
            .err()
            .expect("a spec serving no declared modality should fail to load");
        expect!["the Rust frontend cannot serve multimodal input for model `qwen3_vl-test` (model_type `qwen3_vl`); pass --language-model-only to serve it as a text-only model, or set VLLM_USE_RUST_FRONTEND=0 to use the Python frontend"].assert_eq(&error.to_report_string());

        let disabled = [MmModality::Image, MmModality::Video]
            .map(|modality| (modality, MmLimitSpec::Count(0)));
        let info = load(HashMap::from(disabled)).expect("disabled media should load text-only");
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
    fn kimi_k31_resolves_kimi_k3_image_support_by_model_type() {
        const KIMI_MEDIA_PAD_ID: u32 = 163605;
        let config = serde_json::json!({
            "model_type": "kimi_k31",
            "media_placeholder_token_id": KIMI_MEDIA_PAD_ID,
        });
        let tokenizer = TestTokenizer::new().with_regular_token("<|media_pad|>", KIMI_MEDIA_PAD_ID);
        // The test model ID (`kimi_k31-test`) does not contain `kimi-k3`, so
        // only the model type can select the K3 spec.
        let info = test_info("kimi_k31", config, tokenizer);

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some("<|media_pad|>")
        );
        assert_eq!(info.placeholder_token(Modality::Video), None);
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
        assert!(
            input_audio_data_url("AAAA", Some("flac"))
                .unwrap_err()
                .kind()
                .is_invalid_request()
        );
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
            url: "https://example.com/image.png".to_string().into(),
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
    fn validate_mm_limits_enforces_default_limit_for_unconfigured_modalities() {
        let info = qwen3_vl_info();

        let at_limit: Vec<_> = std::iter::repeat_with(image_url_part).take(999).collect();
        assert!(info.validate_mm_limits(&at_limit).is_ok());

        let over_limit: Vec<_> = std::iter::repeat_with(image_url_part).take(1_000).collect();
        let error = info.validate_mm_limits(&over_limit).unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "At most 999 image(s) may be provided in one prompt."
        );
    }

    #[test]
    fn validate_mm_limits_explicit_limit_overrides_default() {
        let info = qwen3_vl_info_with_limits(HashMap::from([(
            MmModality::Image,
            MmLimitSpec::Count(2000),
        )]));
        let parts: Vec<_> = std::iter::repeat_with(image_url_part).take(2000).collect();
        assert!(info.validate_mm_limits(&parts).is_ok());

        let over: Vec<_> = std::iter::repeat_with(image_url_part).take(2001).collect();
        let error = info.validate_mm_limits(&over).unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "At most 2000 image(s) may be provided in one prompt."
        );
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
        assert!(error.kind().is_invalid_request());
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

    /// An options object without a `count` falls back to the default limit
    /// (999), so small counts still pass.
    #[test]
    fn validate_mm_limits_count_less_options_object_uses_default_limit() {
        let info = qwen3_vl_info_with_limits(HashMap::from([(
            MmModality::Image,
            MmLimitSpec::Options {
                count: None,
                extra: BTreeMap::from([("width".to_string(), serde_json::json!(512))]),
            },
        )]));

        assert!(info.validate_mm_limits(&[image_url_part(), image_url_part()]).is_ok());

        let over_default: Vec<_> = std::iter::repeat_with(image_url_part).take(1_000).collect();
        let error = info.validate_mm_limits(&over_default).unwrap_err();
        assert_eq!(
            error.to_report_string(),
            "At most 999 image(s) may be provided in one prompt."
        );
    }

    fn video_url_part() -> MediaContentPart {
        MediaContentPart::VideoUrl {
            url: "https://example.com/clip.mp4".to_string().into(),
            uuid: None,
            fps: None,
            max_long_side_pixel: None,
        }
    }

    /// A model's default item cap applies to a modality that
    /// `--limit-mm-per-prompt` gives no count, as in Python.
    #[test]
    fn validate_mm_limits_applies_the_model_default_without_a_configured_count() {
        let videos = |count| std::iter::repeat_with(video_url_part).take(count).collect::<Vec<_>>();
        let count_less_options = MmLimitSpec::Options {
            count: None,
            extra: BTreeMap::from([("num_frames".to_string(), serde_json::json!(4))]),
        };

        for info in [
            qwen3_vl_info(),
            qwen3_vl_info_with_limits(HashMap::from([(MmModality::Video, count_less_options)])),
        ] {
            assert!(info.validate_mm_limits(&videos(32)).is_ok());
            let error = info.validate_mm_limits(&videos(33)).unwrap_err();
            assert_eq!(
                error.to_report_string(),
                "At most 32 video(s) may be provided in one prompt."
            );
        }

        let configured =
            qwen3_vl_info_with_limits(HashMap::from([(MmModality::Video, MmLimitSpec::Count(64))]));
        assert!(configured.validate_mm_limits(&videos(64)).is_ok());

        // The default follows the checkpoint's architecture, not a path that
        // happens to select the `qwen3_vl` spec.
        let mut config = qwen3_vl_config();
        config["model_type"] = "qwen2_5_vl".into();
        let context = MultimodalModelContext {
            model_id: "qwen3-eval/Qwen2.5-VL-3B-Instruct".to_string(),
            model_type: Some("qwen2_5_vl".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(qwen3_vl_tokenizer())),
        };
        assert_eq!(context.resolve_model_spec().unwrap().name(), "qwen3_vl");
        let path_matched = from_context(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .unwrap();
        assert!(path_matched.validate_mm_limits(&videos(999)).is_ok());
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

    // Structural token ids from the real HY4-VL checkpoint's `config.json`.
    pub(super) const HY4_IMAGE_START_ID: u32 = 120118;
    pub(super) const HY4_IMAGE_END_ID: u32 = 120119;
    pub(super) const HY4_IMAGE_TOKEN_ID: u32 = 120120;
    pub(super) const HY4_VIDEO_START_ID: u32 = 120122;
    pub(super) const HY4_VIDEO_END_ID: u32 = 120123;
    pub(super) const HY4_VIDEO_TOKEN_ID: u32 = 120124;

    const HY4_IMAGE_START_TOKEN: &str = "<｜hy_image▁start:opensource｜>";
    const HY4_IMAGE_PAD_TOKEN: &str = "<｜hy_image▁pad:opensource｜>";
    const HY4_IMAGE_END_TOKEN: &str = "<｜hy_image▁end:opensource｜>";
    const HY4_VIDEO_START_TOKEN: &str = "<｜hy_video▁start:opensource｜>";
    const HY4_VIDEO_PAD_TOKEN: &str = "<｜hy_video▁pad:opensource｜>";
    const HY4_VIDEO_END_TOKEN: &str = "<｜hy_video▁end:opensource｜>";
    pub(super) const HY4_IMAGE_TOKEN: &str =
        "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>";
    pub(super) const HY4_VIDEO_TOKEN: &str =
        "<｜hy_video▁start:opensource｜><｜hy_video▁pad:opensource｜><｜hy_video▁end:opensource｜>";

    fn hy4_vl_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token(HY4_IMAGE_START_TOKEN, HY4_IMAGE_START_ID)
            .with_regular_token(HY4_IMAGE_PAD_TOKEN, HY4_IMAGE_TOKEN_ID)
            .with_regular_token(HY4_IMAGE_END_TOKEN, HY4_IMAGE_END_ID)
            .with_regular_token(HY4_VIDEO_START_TOKEN, HY4_VIDEO_START_ID)
            .with_regular_token(HY4_VIDEO_PAD_TOKEN, HY4_VIDEO_TOKEN_ID)
            .with_regular_token(HY4_VIDEO_END_TOKEN, HY4_VIDEO_END_ID)
    }

    fn hy4_vl_info() -> MultimodalModelInfo {
        // Structural ids spelled the way the real checkpoint's config.json
        // carries them (they coincide with the spec's class defaults, so this
        // also exercises the integer config-field resolution path).
        let config = serde_json::json!({
            "model_type": "hy_v4_vl",
            "image_start_token_id": HY4_IMAGE_START_ID,
            "image_end_token_id": HY4_IMAGE_END_ID,
            "image_token_id": HY4_IMAGE_TOKEN_ID,
            "video_start_token_id": HY4_VIDEO_START_ID,
            "video_end_token_id": HY4_VIDEO_END_ID,
            "video_token_id": HY4_VIDEO_TOKEN_ID,
        });
        test_info("hy_v4_vl", config, hy4_vl_tokenizer())
    }

    #[test]
    fn hy4_vl_resolves_image_and_video_support_with_checkpoint_ids() {
        let info = hy4_vl_info();

        assert_eq!(
            info.placeholder_token(Modality::Image),
            Some(HY4_IMAGE_TOKEN)
        );
        assert_eq!(
            info.placeholder_token(Modality::Video),
            Some(HY4_VIDEO_TOKEN)
        );
        let image = info.image.as_ref().expect("image support should resolve");
        assert_eq!(
            image.placeholder.marker_token_ids,
            vec![HY4_IMAGE_START_ID, HY4_IMAGE_TOKEN_ID, HY4_IMAGE_END_ID]
        );
        assert_eq!(image.placeholder.embed_token_id, HY4_IMAGE_TOKEN_ID);
        let video = info.video.as_ref().expect("video support should resolve");
        assert_eq!(
            video.placeholder.marker_token_ids,
            vec![HY4_VIDEO_START_ID, HY4_VIDEO_TOKEN_ID, HY4_VIDEO_END_ID]
        );
        assert_eq!(video.placeholder.embed_token_id, HY4_VIDEO_TOKEN_ID);
        // The HY4 engine reads its forward kwargs under the reference
        // processor's names, with the grids kept on CPU.
        assert_eq!(image.spec.primary_key(), "pixel_values");
        assert_eq!(video.spec.primary_key(), "pixel_values_videos");
        assert_eq!(
            info.video_sampling_policy(),
            Some((FrameSampling::EvenRounded, Some(2.0)))
        );
    }

    #[tokio::test]
    async fn hy4_vl_preprocessing_builds_engine_features() {
        use vllm_engine_core_client::protocol::multimodal::{MmField, MmKwargValue};

        let info = hy4_vl_info();
        let media = MediaContentPart::ImageUrl {
            url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQAAAAAdquFJAAAAPUlEQVR42u3BAQ0AAADCoPdPbQ43oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgzwCX4AABS8wbSgAAAABJRU5ErkJggg==".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        };
        let mut tokens = vec![11; 100];
        // The chat template frames its embed marker with structural tokens;
        // expansion must consume the complete marker without duplicating them.
        tokens.extend([HY4_IMAGE_START_ID, HY4_IMAGE_TOKEN_ID, HY4_IMAGE_END_ID]);

        let features = info
            .prepare_multimodal(vec![media], &mut tokens, ModelDtype::Float32, None)
            .await
            .unwrap();

        // The 640x480 image is already a 32-multiple inside the 512^2..2048^2
        // pixel budget, so smart_resize keeps it: the 30x40 patch grid merges
        // to 15 rows x (20 + 1 newline) cols = 315 embed tokens wrapped in
        // image_start/image_end.
        assert_eq!(features.len(), 1);
        let feature = &features[0];
        assert_eq!(
            (feature.mm_position.offset, feature.mm_position.length),
            (100, 317)
        );
        assert_eq!(
            feature.mm_position.is_embed.as_ref().expect("is_embed mask").shape,
            [317]
        );
        assert_eq!(tokens.len(), 417);
        assert_eq!(tokens[100], HY4_IMAGE_START_ID);
        assert!(tokens[101..416].iter().all(|&token| token == HY4_IMAGE_TOKEN_ID));
        assert_eq!(tokens[416], HY4_IMAGE_END_ID);

        let data = feature.data.as_ref().unwrap();
        // Normalized f32 patches flat over the image's 30x40 grid: the
        // frontend rescales (1/255) and CLIP-normalizes in the processor
        // because the engine runs with `mm_device_do_normalize=False` (an
        // IdentityInputNorm pass-through).
        let pixels = &data["pixel_values"];
        assert!(matches!(&pixels.field, MmField::Flat(field) if !field.keep_on_cpu));
        let Some(MmKwargValue::Tensor(pixel_values)) = pixels.data.as_ref() else {
            panic!("pixel_values should be a tensor");
        };
        assert_eq!(pixel_values.dtype.as_str(), "float32");
        assert_eq!(pixel_values.shape, [1200, 3 * 16 * 16]);
        let fused = |byte: u8, mean: f64, std: f64| {
            (f32::from(byte) - mean as f32 * 255.0) / (std as f32 * 255.0)
        };
        let stats = PreProcessorConfig::CLIP_MEAN.iter().zip(PreProcessorConfig::CLIP_STD);
        let min = stats.clone().map(|(&m, s)| fused(0, m, s)).fold(f32::INFINITY, f32::min);
        let max = stats.map(|(&m, s)| fused(255, m, s)).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            pixel_values
                .data
                .as_raw_view()
                .unwrap()
                .as_ref()
                .chunks_exact(4)
                .all(|bytes| (min..=max).contains(&f32::from_ne_bytes(bytes.try_into().unwrap())))
        );
        let grid = &data["image_grid_thw"];
        assert!(matches!(&grid.field, MmField::Batched(field) if field.keep_on_cpu));
        let Some(MmKwargValue::Tensor(grid_tensor)) = grid.data.as_ref() else {
            panic!("image_grid_thw should be a tensor");
        };
        assert_eq!(grid_tensor.shape, [3]);
        assert_eq!(
            grid_tensor.data.as_raw_view().unwrap().as_ref(),
            [1_i64, 30, 40].into_iter().flat_map(i64::to_ne_bytes).collect::<Vec<_>>()
        );
        // The flat primary's per-item slicing sizes ride along as a
        // keep-on-CPU companion tensor the engine ignores.
        let counts = &data["image_patch_counts"];
        assert!(matches!(&counts.field, MmField::Batched(field) if field.keep_on_cpu));
    }

    /// Render a HY4-VL replacement compactly: stamp-text bytes (ids in the
    /// tokenizer's byte range) as their string, runs of a repeated token as
    /// `id xN`, structural ids bare.
    fn hy4_signature(tokens: &[TokenId]) -> String {
        let mut parts = Vec::new();
        let mut text = String::new();
        let mut index = 0;
        while index < tokens.len() {
            let token = tokens[index];
            if (0..=255).contains(&token) {
                text.push(char::from(token as u8));
                index += 1;
                continue;
            }
            if !text.is_empty() {
                parts.push(format!("{text:?}"));
                text.clear();
            }
            let mut end = index + 1;
            while end < tokens.len() && tokens[end] == token {
                end += 1;
            }
            if end - index > 1 {
                parts.push(format!("{token} x{}", end - index));
            } else {
                parts.push(token.to_string());
            }
            index = end;
        }
        if !text.is_empty() {
            parts.push(format!("{text:?}"));
        }
        parts.join(" ")
    }

    /// The connector's container decode needs a real video file, so the
    /// in-process harness feeds `prepare_videos` a synthesized clip directly
    /// (the same reach the muse_spark and GLM video tests use) and then runs
    /// the shared prompt-expansion step, covering everything but the fetch.
    #[tokio::test]
    async fn hy4_vl_video_preprocessing_stamps_frames_and_builds_engine_features() {
        use ::image::{DynamicImage, Rgb, RgbImage};
        use llm_multimodal::VideoSource;
        use vllm_engine_core_client::protocol::multimodal::{MmField, MmKwargValue};

        let info = hy4_vl_info();
        let frames =
            vec![DynamicImage::ImageRgb8(RgbImage::from_pixel(512, 512, Rgb([120, 30, 200]))); 2];
        let clip = Arc::new(VideoClip::new_with_sample_fps(
            frames,
            bytes::Bytes::from_static(b"video"),
            VideoSource::InlineBytes,
            "clip".to_string(),
            2.0,
        ));
        let prepared =
            info.prepare_videos(vec![clip], vec![None], ModelDtype::Float32).await.unwrap();

        // 512x512 frames survive adaptive_resize; each frame's 32x32 patch
        // grid merges to 16 x (16 + 1 newline) = 272 embed tokens, and the
        // 2 fps clip stamps its frames at 0.0s and 0.5s.
        let stamp = |text: &str| {
            info.context
                .tokenizer()
                .encode(text, false)
                .unwrap()
                .into_iter()
                .map(|id| id as TokenId)
                .collect::<Vec<_>>()
        };
        let mut expected = vec![HY4_VIDEO_START_ID as TokenId];
        for text in ["<0.0 seconds>", "<0.5 seconds>"] {
            expected.extend(stamp(text));
            expected.push(HY4_IMAGE_START_ID as TokenId);
            expected.extend(std::iter::repeat_n(HY4_VIDEO_TOKEN_ID as TokenId, 272));
            expected.push(HY4_IMAGE_END_ID as TokenId);
        }
        expected.push(HY4_VIDEO_END_ID as TokenId);
        assert_eq!(prepared.replacements.len(), 1);
        let replacement = &prepared.replacements[0];
        assert_eq!(replacement.modality, Modality::Video);
        assert_eq!(replacement.placeholder_token, HY4_VIDEO_TOKEN);
        assert_eq!(replacement.tokens, expected);
        expect_test::expect![[r#"120122 "<0.0 seconds>" 120118 120124 x272 120119 "<0.5 seconds>" 120118 120124 x272 120119 120123"#]]
            .assert_eq(&hy4_signature(&replacement.tokens));

        let mut prompt = vec![11; 100];
        prompt.extend([HY4_VIDEO_START_ID, HY4_VIDEO_TOKEN_ID, HY4_VIDEO_END_ID]);
        let mut ranges =
            expand_prompt_token_ids(&mut prompt, std::slice::from_ref(&prepared)).unwrap();
        let video_ranges = ranges.remove(&Modality::Video).unwrap_or_default();
        assert_eq!(video_ranges.len(), 1);
        assert_eq!(
            (video_ranges[0].offset, video_ranges[0].length),
            (100, expected.len())
        );
        assert_eq!(prompt.len(), 100 + expected.len());

        assert_eq!(prepared.items.len(), 1);
        let item = &prepared.items[0];
        let pixels = &item.data["pixel_values_videos"];
        assert!(matches!(&pixels.field, MmField::Flat(field) if !field.keep_on_cpu));
        let Some(MmKwargValue::Tensor(pixel_values)) = pixels.data.as_ref() else {
            panic!("pixel_values_videos should be a tensor");
        };
        assert_eq!(pixel_values.dtype.as_str(), "float32");
        assert_eq!(pixel_values.shape, [2 * 32 * 32, 3 * 16 * 16]);

        let grid = &item.data["video_grid_thw"];
        assert!(matches!(&grid.field, MmField::Batched(field) if field.keep_on_cpu));
        let Some(MmKwargValue::Tensor(grid_tensor)) = grid.data.as_ref() else {
            panic!("video_grid_thw should be a tensor");
        };
        assert_eq!(grid_tensor.shape, [3]);
        assert_eq!(
            grid_tensor.data.as_raw_view().unwrap().as_ref(),
            [2_i64, 32, 32].into_iter().flat_map(i64::to_ne_bytes).collect::<Vec<_>>()
        );

        // One f32 timestamp per merged frame, shipped flat and kept on CPU
        // like the reference's `flat_from_sizes` field config.
        let timestamps = &item.data["video_timestamps"];
        assert!(matches!(&timestamps.field, MmField::Flat(field) if field.keep_on_cpu));
        let Some(MmKwargValue::Tensor(timestamp_tensor)) = timestamps.data.as_ref() else {
            panic!("video_timestamps should be a tensor");
        };
        assert_eq!(timestamp_tensor.shape, [2]);
        assert_eq!(
            timestamp_tensor.data.as_raw_view().unwrap().as_ref(),
            [0.0_f32, 0.5].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>()
        );

        for key in ["video_patch_counts", "video_timestamp_counts"] {
            assert!(
                matches!(&item.data[key].field, MmField::Batched(field) if field.keep_on_cpu),
                "{key}"
            );
        }
    }
}
