use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};

/// Read an optional JSON file into `T`, returning `T::default()` when absent.
pub fn read_json_file<T>(path: Option<&Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let Some(path) = path else {
        return Ok(T::default());
    };
    let content = fs::read_to_string(path).map_err(|error| {
        Error::new(format!(
            "failed to read {}: {}",
            path.display(),
            error.as_report()
        ))
    })?;
    serde_json::from_str(&content).map_err(|error| {
        Error::new(format!(
            "failed to parse {}: {}",
            path.display(),
            error.as_report()
        ))
    })
}
