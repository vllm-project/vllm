// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;

use hf_hub::Repo;
use hf_hub::api::tokio::{ApiBuilder, ApiRepo};

/// A handle to a HuggingFace Hub repo, downloading via hf-hub's on-disk cache.
pub struct HubRepo {
    repo: ApiRepo,
}

impl HubRepo {
    pub fn model(model_id: String) -> Result<Self, String> {
        Self::new(Repo::model(model_id))
    }

    pub fn dataset(repo_id: String) -> Result<Self, String> {
        Self::new(Repo::dataset(repo_id))
    }

    fn new(repo: Repo) -> Result<Self, String> {
        let mut builder = ApiBuilder::from_env();
        if let Ok(token) = std::env::var("HF_TOKEN") {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build().map_err(|e| format!("Failed to init HF API: {e}"))?;
        Ok(Self {
            repo: api.repo(repo),
        })
    }

    /// Download (or fetch from cache) a single file from the repo.
    /// Auth is handled by hf-hub via HF_TOKEN / the cached login token.
    pub async fn get(&self, filename: &str) -> Result<PathBuf, String> {
        self.repo.get(filename).await.map_err(|e| format!("{e}"))
    }
}
