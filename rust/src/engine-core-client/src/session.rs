// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::transport::{ConnectedEngine, ConnectedTransport, EngineId};

#[derive(Debug, Serialize, Deserialize)]
struct EngineSession {
    input_address: String,
    output_address: String,
    engines: Vec<SessionEngine>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionEngine {
    engine_id: Vec<u8>,
    ready_response: EngineCoreReadyResponse,
}

impl EngineSession {
    fn from_connected(path: &Path, connected: &ConnectedTransport) -> Result<Self> {
        if connected.coordinator.is_some() {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: "reattach sessions do not support an in-process coordinator".to_string(),
            });
        }
        if connected.engines.len() != 1 {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: format!(
                    "reattach sessions currently require exactly one engine, found {}",
                    connected.engines.len()
                ),
            });
        }

        Ok(Self {
            input_address: connected.input_address.clone(),
            output_address: connected.output_address.clone(),
            engines: connected
                .engines
                .iter()
                .map(|engine| SessionEngine {
                    engine_id: engine.engine_id.to_vec(),
                    ready_response: engine.ready_response.clone(),
                })
                .collect(),
        })
    }

    fn into_parts(self, path: &Path) -> Result<(String, String, Vec<ConnectedEngine>)> {
        if self.engines.len() != 1 {
            return Err(Error::EngineSession {
                path: path.to_path_buf(),
                message: format!(
                    "reattach sessions currently require exactly one engine, found {}",
                    self.engines.len()
                ),
            });
        }

        Ok((
            self.input_address,
            self.output_address,
            self.engines
                .into_iter()
                .map(|engine| ConnectedEngine {
                    engine_id: EngineId::from(engine.engine_id),
                    ready_response: engine.ready_response,
                })
                .collect(),
        ))
    }
}

pub(crate) fn write(path: &Path, connected: &ConnectedTransport) -> Result<()> {
    let session = EngineSession::from_connected(path, connected)?;
    let bytes = serde_json::to_vec_pretty(&session).map_err(|error| Error::EngineSession {
        path: path.to_path_buf(),
        message: error.to_report_string(),
    })?;
    let temporary_path = temporary_path(path)?;

    let result = fs::write(&temporary_path, bytes)
        .and_then(|()| fs::rename(&temporary_path, path))
        .map_err(|error| Error::EngineSession {
            path: path.to_path_buf(),
            message: error.to_report_string(),
        });
    if result.is_err() {
        let _ = fs::remove_file(temporary_path);
    }
    result
}

pub(crate) fn read(path: &Path) -> Result<(String, String, Vec<ConnectedEngine>)> {
    let bytes = fs::read(path).map_err(|error| Error::EngineSession {
        path: path.to_path_buf(),
        message: error.to_report_string(),
    })?;
    let session: EngineSession =
        serde_json::from_slice(&bytes).map_err(|error| Error::EngineSession {
            path: path.to_path_buf(),
            message: error.to_report_string(),
        })?;
    session.into_parts(path)
}

fn temporary_path(path: &Path) -> Result<PathBuf> {
    let Some(file_name) = path.file_name() else {
        return Err(Error::EngineSession {
            path: path.to_path_buf(),
            message: "session path must name a file".to_string(),
        });
    };
    let temporary_name = format!(
        ".{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id()
    );
    Ok(path.with_file_name(temporary_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_engine::default_ready_response;
    use crate::transport::connect_reattach;

    #[tokio::test]
    async fn session_round_trip_restores_transport_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine-session.json");
        let input_address = format!("ipc://{}", dir.path().join("input.sock").display());
        let output_address = format!("ipc://{}", dir.path().join("output.sock").display());
        let connected = connect_reattach(
            &input_address,
            &output_address,
            vec![ConnectedEngine {
                engine_id: EngineId::from_engine_index(0),
                ready_response: default_ready_response(),
            }],
        )
        .await
        .unwrap();

        write(&path, &connected).unwrap();
        let (actual_input, actual_output, engines) = read(&path).unwrap();

        assert_eq!(actual_input, input_address);
        assert_eq!(actual_output, output_address);
        assert_eq!(engines.len(), 1);
        assert_eq!(engines[0].engine_id.engine_index(), Some(0));
        assert_eq!(
            engines[0].ready_response.max_model_len,
            default_ready_response().max_model_len
        );
    }
}
