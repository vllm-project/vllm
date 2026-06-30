//! Shared Harmony encoding helper for the GPT-OSS renderer and output parser.

use std::sync::LazyLock;

use anyhow::Context as _;
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};

/// Lazily load the shared GPT-OSS Harmony encoding once per process.
pub(crate) fn harmony_encoding() -> Result<&'static HarmonyEncoding> {
    static ENCODING: LazyLock<anyhow::Result<HarmonyEncoding>> = LazyLock::new(|| {
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .context("failed to load harmony encoding for gpt-oss")
    });

    ENCODING.as_ref().map_err(|error| Error::HarmonyOutputParsing {
        error: error.to_report_string().into(),
    })
}
