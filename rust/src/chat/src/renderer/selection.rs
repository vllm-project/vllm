use std::fmt;
use std::str::FromStr;

use itertools::Itertools;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{EnumIter, IntoEnumIterator};

/// Specify which chat renderer implementation to use.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay, EnumIter,
)]
pub enum RendererSelection {
    /// Use model-based auto-detection.
    #[default]
    Auto,
    /// Force the generic Hugging Face chat-template renderer.
    Hf,
    /// Force the DeepSeek V3.2 renderer.
    DeepSeekV32,
    /// Force the DeepSeek V4 renderer.
    DeepSeekV4,
    /// Force the GPT-OSS Harmony renderer.
    Harmony,
}

impl RendererSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const DEEPSEEK_V32_LITERAL: &str = "deepseek_v32";
    pub const DEEPSEEK_V4_LITERAL: &str = "deepseek_v4";
    pub const GPT_OSS_MODEL_TYPE: &str = "gpt_oss";
    pub const HARMONY_LITERAL: &str = "harmony";
    pub const HF_LITERAL: &str = "hf";

    /// Resolve the renderer selection using the given model type string, if
    /// it's `Auto`.
    pub fn resolve(self, model_type: &str) -> Self {
        match self {
            Self::Auto => match model_type {
                Self::DEEPSEEK_V32_LITERAL => Self::DeepSeekV32,
                Self::DEEPSEEK_V4_LITERAL => Self::DeepSeekV4,
                Self::GPT_OSS_MODEL_TYPE => Self::Harmony,
                _ => Self::Hf,
            },
            selection => selection,
        }
    }
}

impl FromStr for RendererSelection {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::HF_LITERAL) {
            Ok(Self::Hf)
        } else if value.eq_ignore_ascii_case(Self::DEEPSEEK_V32_LITERAL) {
            Ok(Self::DeepSeekV32)
        } else if value.eq_ignore_ascii_case(Self::DEEPSEEK_V4_LITERAL) {
            Ok(Self::DeepSeekV4)
        } else if value.eq_ignore_ascii_case(Self::HARMONY_LITERAL) {
            Ok(Self::Harmony)
        } else {
            Err(format!(
                "unknown renderer `{value}` (expected one of: {})",
                Self::iter().join(", ")
            ))
        }
    }
}

impl fmt::Display for RendererSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::Hf => f.write_str(Self::HF_LITERAL),
            Self::DeepSeekV32 => f.write_str(Self::DEEPSEEK_V32_LITERAL),
            Self::DeepSeekV4 => f.write_str(Self::DEEPSEEK_V4_LITERAL),
            Self::Harmony => f.write_str(Self::HARMONY_LITERAL),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use strum::IntoEnumIterator;

    use super::RendererSelection;

    #[test]
    fn renderer_selection_display_round_trips() {
        for selection in RendererSelection::iter() {
            assert_eq!(
                selection.to_string().parse::<RendererSelection>().unwrap(),
                selection
            );
        }
    }

    #[test]
    fn renderer_selection_expected_error_message() {
        let err = RendererSelection::from_str("unknown").unwrap_err();
        expect_test::expect![
            "unknown renderer `unknown` (expected one of: auto, hf, deepseek_v32, deepseek_v4, harmony)"
        ]
        .assert_eq(&err);
    }
}
