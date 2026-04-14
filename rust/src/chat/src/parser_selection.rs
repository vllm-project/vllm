use std::fmt;

use serde::Deserialize;

/// Specify which reasoning or tool-call parser implementation to use.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ParserSelection {
    /// Use model-based auto-detection.
    #[default]
    Auto,
    /// Disable the parser entirely.
    None,
    /// Force one specific parser implementation by name.
    Explicit(String),
}

impl ParserSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const NONE_LITERAL: &str = "none";
}

impl From<String> for ParserSelection {
    fn from(value: String) -> Self {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Self::Auto
        } else if value.eq_ignore_ascii_case(Self::NONE_LITERAL) {
            Self::None
        } else {
            Self::Explicit(value)
        }
    }
}

impl fmt::Display for ParserSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::None => f.write_str(Self::NONE_LITERAL),
            Self::Explicit(name) => f.write_str(name),
        }
    }
}

impl<'de> Deserialize<'de> for ParserSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Option::<String>::deserialize(deserializer)?;
        Ok(value.map_or(Self::Auto, Self::from))
    }
}
