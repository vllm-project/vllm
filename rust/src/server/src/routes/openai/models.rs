use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject};
use crate::state::AppState;

/// Return all configured served model names in OpenAI `list models` format.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: state
            .served_model_names()
            .iter()
            .map(|name| ModelObject {
                id: name.clone(),
                object: "model".to_string(),
                created: 0,
                owned_by: "vllm-frontend-rs".to_string(),
            })
            .collect(),
    })
}
