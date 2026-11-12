// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The typed result of a utility call.

use easy_ext::ext;
use rmpv::Value;
use thiserror_ext::AsReport;

use crate::protocol::utility::UtilityOutput;
use crate::{Error, Result};

#[ext(UtilityOutputExt)]
pub(crate) impl UtilityOutput {
    /// Decode the typed result of a utility call.
    fn into_typed_result<T>(self, method: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        if let Some(message) = self.failure_message {
            return Err(Error::UtilityCallFailed {
                method: method.to_string(),
                call_id: self.call_id,
                message,
            });
        }

        let result = self.result.map(|e| e.into_result()).unwrap_or(Value::Nil);

        rmpv::ext::from_value(result).map_err(|error| Error::UtilityResultDecode {
            method: method.to_string(),
            call_id: self.call_id,
            message: error.to_report_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;
    use serde::Serialize;

    use super::UtilityOutputExt;
    use crate::Error;
    use crate::protocol::utility::{UtilityOutput, UtilityResultEnvelope};

    fn utility_result_value<T>(value: T) -> UtilityResultEnvelope
    where
        T: Serialize,
    {
        UtilityResultEnvelope::without_type_info(rmpv::ext::to_value(value).unwrap())
    }

    #[test]
    fn utility_output_decodes_typed_result() {
        let output = UtilityOutput {
            call_id: 9_u64.into(),
            failure_message: None,
            result: Some(utility_result_value(true)),
        };

        assert!(output.into_typed_result::<bool>("is_sleeping").unwrap());
    }

    #[test]
    fn utility_output_reports_failure_message() {
        let error = UtilityOutput {
            call_id: 9_u64.into(),
            failure_message: Some("boom".to_string()),
            result: None,
        }
        .into_typed_result::<bool>("is_sleeping")
        .unwrap_err();

        assert!(matches!(
            error,
            Error::UtilityCallFailed {
                method,
                call_id,
                message
            } if method == "is_sleeping" && call_id == 9 && message == "boom"
        ));
    }

    #[test]
    fn utility_output_decodes_missing_result_as_unit() {
        UtilityOutput {
            call_id: 3_u64.into(),
            failure_message: None,
            result: None,
        }
        .into_typed_result::<()>("reset_mm_cache")
        .unwrap();
    }

    #[test]
    fn utility_output_decodes_nil_result_as_unit() {
        UtilityOutput {
            call_id: 4_u64.into(),
            failure_message: None,
            result: Some(UtilityResultEnvelope::without_type_info(Value::Nil)),
        }
        .into_typed_result::<()>("sleep")
        .unwrap();
    }
}
