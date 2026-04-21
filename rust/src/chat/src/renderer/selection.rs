use std::fmt;
use std::str::FromStr;

use serde_with::DeserializeFromStr;

/// Specify which chat renderer implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr)]
pub enum RendererSelection {
    /// Use model-based auto-detection.
    #[default]
    Auto,
    /// Force the generic Hugging Face chat-template renderer.
    Hf,
    /// Force the DeepSeek V3.2 renderer.
    DeepSeekV32,
}

impl RendererSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const DEEPSEEK_V32_LITERAL: &str = "deepseek_v32";
    pub const HF_LITERAL: &str = "hf";

    /// Resolve the renderer selection using the given model type string, if it's `Auto`.
    pub fn resolve(self, model_type: &str) -> Self {
        match self {
            Self::Auto => match model_type {
                Self::DEEPSEEK_V32_LITERAL => Self::DeepSeekV32,
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
        } else {
            Err(format!(
                "unknown renderer `{value}` (expected one of: auto, hf, deepseek_v32)"
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RendererSelection;

    #[test]
    fn renderer_selection_parses_known_values() {
        assert_eq!(
            "auto".parse::<RendererSelection>().unwrap(),
            RendererSelection::Auto
        );
        assert_eq!(
            "hf".parse::<RendererSelection>().unwrap(),
            RendererSelection::Hf
        );
        assert_eq!(
            "deepseek_v32".parse::<RendererSelection>().unwrap(),
            RendererSelection::DeepSeekV32
        );
    }

    #[test]
    fn renderer_selection_display_round_trips() {
        for selection in [
            RendererSelection::Auto,
            RendererSelection::Hf,
            RendererSelection::DeepSeekV32,
        ] {
            assert_eq!(
                selection.to_string().parse::<RendererSelection>().unwrap(),
                selection
            );
        }
    }
}
