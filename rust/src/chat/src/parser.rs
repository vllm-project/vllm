use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

use serde_with::DeserializeFromStr;

/// Specify which reasoning or tool-call parser implementation to use.
#[derive(Debug, Clone, PartialEq, Eq, Default, DeserializeFromStr)]
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

impl FromStr for ParserSelection {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Self::Auto
        } else if value.eq_ignore_ascii_case(Self::NONE_LITERAL) {
            Self::None
        } else {
            Self::Explicit(value.to_owned())
        })
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

/// Registry and model matcher for reasoning and tool parsers.
#[derive(Clone)]
pub struct ParserFactory<C> {
    creators: HashMap<String, C>,
    patterns: Vec<(String, String)>,
}

impl<C> Default for ParserFactory<C> {
    fn default() -> Self {
        Self {
            creators: HashMap::new(),
            patterns: Vec::new(),
        }
    }
}

impl<C> ParserFactory<C> {
    /// Register a creator for a parser by an exact name.
    pub fn register_creator(&mut self, name: &str, creator: C) -> &mut Self {
        self.creators.insert(name.to_string(), creator);
        self
    }

    /// Add a case-insensitive substring match from model ID to parser name.
    pub fn register_pattern(&mut self, pattern: &str, parser_name: &str) -> &mut Self {
        self.patterns
            .push((pattern.to_lowercase(), parser_name.to_string()));
        self
    }

    /// Return the first registered parser name matching the given model ID.
    pub fn resolve_name_for_model(&self, model_id: &str) -> Option<&str> {
        let model_lower = model_id.to_lowercase();
        self.patterns
            .iter()
            .find(|(pattern, _)| model_lower.contains(pattern))
            .map(|(_, parser_name)| parser_name.as_str())
    }

    /// Return true if the exact parser name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.creators.contains_key(name)
    }

    /// Return all registered parser names sorted for stable display.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<_> = self.creators.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Get the constructor for a parser by its exact registered name, or return None if not found.
    pub fn creator(&self, name: &str) -> Option<&C> {
        self.creators.get(name)
    }
}
