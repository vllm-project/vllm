//! Validated JSON extractor for automatic request validation.

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;
use validator::Validate;

use super::types::Normalizable;
use crate::error::{ApiError, invalid_request};

/// A JSON extractor that automatically validates and normalizes the request body.
///
/// This extractor deserializes the request body and automatically calls `.validate()`
/// on types that implement the `Validate` trait. If validation fails, it returns
/// [`ApiError::InvalidRequest`] with details about the validation errors.
pub struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(mut data) = Json::<T>::from_request(req, state)
            .await
            .map_err(|err: JsonRejection| ApiError::json_parse_error(err.body_text()))?;

        data.normalize();

        data.validate()
            .map_err(|validation_errors| invalid_request!("{}", validation_errors))?;

        Ok(ValidatedJson(data))
    }
}

impl<T> std::ops::Deref for ValidatedJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ValidatedJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
