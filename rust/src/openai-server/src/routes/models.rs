use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use openai_protocol::models::{ListModelsResponse, ModelObject};

use crate::state::AppState;

/// Return the single configured model in OpenAI `list models` format.
pub(super) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "vllm-frontend-rs".to_string(),
        }],
    })
}
