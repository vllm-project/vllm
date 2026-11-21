// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Validated JSON extractor for automatic request validation.
//! Variation of https://github.com/lightseekorg/smg/blob/main/crates/protocols/src/validated.rs

use std::borrow::Cow;

use axum::Json;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;
use validator::{Validate, ValidationErrors};

use super::types::Normalizable;
use crate::error::{ApiError, invalid_request};

/// A JSON extractor that automatically validates and normalizes the request
/// body.
///
/// This extractor deserializes the request body and automatically calls
/// `.validate()` on types that implement the `Validate` trait. If validation
/// fails, it returns [`ApiError::InvalidRequest`] with details about the
/// validation errors.
pub struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(mut data) = Json::<T>::from_request(req, state).await?;

        data.normalize();

        data.validate().map_err(|validation_errors| {
            invalid_request!(
                param = error_param(&validation_errors),
                "{}",
                validation_errors
            )
        })?;

        Ok(ValidatedJson(data))
    }
}

/// The request field to report as `param`, like Python: the field a field
/// check failed on, or for a cross-field check, the `<field>` of a
/// `<field>.<reason>` error code. `None` when several fields failed.
fn error_param(errors: &ValidationErrors) -> Option<&'static str> {
    let mut fields = errors.field_errors().into_iter();
    let (field, field_errors) = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    match field {
        Cow::Borrowed("__all__") => match &field_errors.first()?.code {
            Cow::Borrowed(code) => code.split_once('.').map(|(field, _)| field),
            Cow::Owned(_) => None,
        },
        Cow::Borrowed(field) => Some(field),
        Cow::Owned(_) => None,
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
