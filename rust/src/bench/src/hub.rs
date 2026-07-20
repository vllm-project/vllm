// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Sync facade over the async `hf_hub` API.
//!
//! The workspace bans rustls (`rust/deny.toml`), but hf-hub's sync `ureq`
//! backend unconditionally pulls ureq's default rustls feature. So we use the
//! reqwest/native-tls tokio API instead, and bridge blocking callers (dataset
//! loaders, tokenizer fallback in rayon threads) by running each download on a
//! dedicated thread with its own single-threaded runtime.

use std::path::PathBuf;

/// A handle to a HuggingFace Hub repo, downloading via hf-hub's on-disk cache.
pub struct HubRepo {
    repo: hf_hub::Repo,
}

impl HubRepo {
    pub fn model(model_id: String) -> Self {
        Self {
            repo: hf_hub::Repo::model(model_id),
        }
    }

    pub fn dataset(repo_id: String) -> Self {
        Self {
            repo: hf_hub::Repo::dataset(repo_id),
        }
    }

    /// Download (or fetch from cache) a single file from the repo.
    /// Auth is handled by hf-hub via HF_TOKEN / the cached login token.
    pub fn get(&self, filename: &str) -> Result<PathBuf, String> {
        let repo = self.repo.clone();
        let filename = filename.to_string();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("Failed to build download runtime: {e}"))?;
            rt.block_on(async move {
                let mut builder = hf_hub::api::tokio::ApiBuilder::from_env();
                if let Ok(token) = std::env::var("HF_TOKEN") {
                    builder = builder.with_token(Some(token));
                }
                let api = builder.build().map_err(|e| format!("Failed to init HF API: {e}"))?;
                api.repo(repo).get(&filename).await.map_err(|e| format!("{e}"))
            })
        })
        .join()
        .map_err(|_| "HF Hub download thread panicked".to_string())?
    }
}
