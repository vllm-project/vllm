// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python's media fetching and decoding settings, which it reads from
//! environment variables. The Rust frontend fetches and decodes media through
//! llm-multimodal with its own settings, so each type accepts only the values
//! that match what it does.

use std::str::FromStr;

use clap::builder::TypedValueParser as _;
use clap::{Args, ValueEnum};
use serde::{Deserialize, Deserializer};

use super::unsupported::{AtMost, Fixed, Unsupported};
use super::{parse_int, parse_int_flag};

/// Python's media settings, each set only through its environment variable,
/// shared by the frontend and the render server, which both fetch media.
#[derive(Clone, Debug, Default, Args, PartialEq, Eq, Deserialize)]
pub struct MediaArgs {
    /// Directory where Python caches downloaded media.
    #[arg(long, hide = true, env = "VLLM_MEDIA_CACHE")]
    #[serde(default)]
    pub media_cache: Option<NoMediaCache>,

    /// Total fetch attempts per media URL (Python's default is 3). The Rust
    /// frontend makes one attempt, so only up to 1 is accepted.
    #[arg(
        long,
        hide = true,
        env = "VLLM_MEDIA_FETCH_MAX_RETRIES",
        value_parser = parse_int.try_map(AtMost::<1>::try_from)
    )]
    #[serde(default)]
    pub media_fetch_max_retries: Option<AtMost<1>>,

    /// Whether media URL fetches follow redirects (on by default). The Rust
    /// frontend always follows them, so only on is accepted.
    #[arg(
        long,
        hide = true,
        env = "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        value_parser = parse_int_flag.try_map(Fixed::<true>::try_from)
    )]
    #[serde(default)]
    pub media_url_allow_redirects: Option<Fixed<true>>,

    /// Maximum decoded pixels of an image or a video frame. llm-multimodal
    /// applies its own decode limits, so no value is accepted.
    #[arg(long, hide = true, env = "VLLM_MAX_IMAGE_PIXELS")]
    #[serde(default)]
    pub max_image_pixels: Option<Unsupported>,

    /// Maximum decoded duration of an audio clip, in seconds.
    #[arg(long, hide = true, env = "VLLM_MAX_AUDIO_DECODE_DURATION_S")]
    #[serde(default)]
    pub max_audio_decode_duration_s: Option<Unsupported>,

    /// Maximum decoded size of an audio clip, in bytes.
    #[arg(long, hide = true, env = "VLLM_MAX_AUDIO_DECODE_BYTES")]
    #[serde(default)]
    pub max_audio_decode_bytes: Option<Unsupported>,

    /// Python's video loader, which picks the frames to decode (default
    /// `opencv`).
    #[arg(long, hide = true, env = "VLLM_VIDEO_LOADER_BACKEND")]
    #[serde(default)]
    pub video_loader_backend: Option<VideoLoaderBackend>,

    /// Python's media connector plugin (default `http`).
    #[arg(long, hide = true, env = "VLLM_MEDIA_CONNECTOR")]
    #[serde(default)]
    pub media_connector: Option<MediaConnector>,
}

/// `VLLM_MEDIA_CACHE`, Python's media cache directory. The Rust frontend has
/// no media cache, so only the empty value, with which Python caches nothing
/// either, is accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoMediaCache;

impl FromStr for NoMediaCache {
    type Err = String;

    fn from_str(dir: &str) -> Result<Self, Self::Err> {
        dir.is_empty().then_some(Self).ok_or_else(|| {
            "a media cache is not implemented in Rust frontend; only an empty value is".to_string()
        })
    }
}

impl<'de> Deserialize<'de> for NoMediaCache {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?.parse().map_err(serde::de::Error::custom)
    }
}

/// The video loaders the Rust frontend implements: it decodes with ffmpeg and
/// picks frames like Python's default `opencv` loader (`numpy.linspace` over
/// the clip), whose `cv2.VideoCapture` decodes through FFmpeg too. Python's
/// other loaders are plugins or sample frames differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoLoaderBackend {
    Opencv,
}

/// The media connectors the Rust frontend implements: its own HTTP fetching,
/// like Python's default `http`. Python's other connectors are plugins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaConnector {
    Http,
}
