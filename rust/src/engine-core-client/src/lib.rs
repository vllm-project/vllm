mod client;
mod coordinator;
mod error;
mod metrics;
pub mod protocol;
#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;
mod transport;

pub use client::{
    AbortCause, CoordinatorMode, EngineCoreClient, EngineCoreClientConfig, EngineCoreOutputStream,
    EngineCoreStreamOutput, TransportMode,
};
pub use error::{Error, Result};
pub use transport::{ENGINE_CORE_DEAD_SENTINEL, EngineId};

#[cfg(test)]
mod tests;
