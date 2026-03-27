#![feature(iterator_try_collect)]

mod client;
mod error;
mod metrics;
pub mod protocol;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod transport;

pub use client::{
    EngineCoreClient, EngineCoreClientConfig, EngineCoreOutputStream, EngineCoreStreamOutput,
};
pub use error::{Error, Result};
pub use transport::{ENGINE_CORE_DEAD_SENTINEL, EngineId};

#[cfg(test)]
mod tests;
