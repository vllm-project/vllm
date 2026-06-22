use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Number of log probabilities requested for a token position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogprobsCount {
    /// Return the full model vocabulary.
    All,
    /// Return the top-N tokens by probability.
    Top(u32),
}

impl LogprobsCount {
    /// Expands the count to the actual number of logprobs to return, given the vocabulary size.
    pub fn expanded(self, vocab_size: usize) -> usize {
        match self {
            Self::All => vocab_size,
            Self::Top(count) => count as usize,
        }
    }
}

impl TryFrom<i32> for LogprobsCount {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::All),
            value if value < -1 => Err(format!("must be non-negative or -1, got {value}")),
            value => Ok(Self::Top(value as u32)),
        }
    }
}

impl TryFrom<LogprobsCount> for i32 {
    type Error = String;

    fn try_from(value: LogprobsCount) -> Result<Self, Self::Error> {
        match value {
            LogprobsCount::All => Ok(-1),
            LogprobsCount::Top(count) => {
                i32::try_from(count).map_err(|_| format!("must fit within i32, got {count}"))
            }
        }
    }
}

impl FromStr for LogprobsCount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = s
            .parse::<i32>()
            .map_err(|e| format!("must be an i32 integer, got {s:?}: {e}"))?;
        Self::try_from(value)
    }
}

impl fmt::Display for LogprobsCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => (-1).fmt(f),
            Self::Top(count) => count.fmt(f),
        }
    }
}

impl Serialize for LogprobsCount {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value: i32 = (*self).try_into().map_err(serde::ser::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LogprobsCount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    #[test]
    fn logprobs_count_serializes_as_wire_integer() {
        assert_eq!(serde_json::to_value(LogprobsCount::All).unwrap(), -1);
        assert_eq!(serde_json::to_value(LogprobsCount::Top(3)).unwrap(), 3);
    }

    #[test]
    fn logprobs_count_deserializes_wire_integer() {
        assert_eq!(
            serde_json::from_value::<LogprobsCount>(serde_json::json!(-1)).unwrap(),
            LogprobsCount::All
        );
        assert_eq!(
            serde_json::from_value::<LogprobsCount>(serde_json::json!(3)).unwrap(),
            LogprobsCount::Top(3)
        );
        assert!(serde_json::from_value::<LogprobsCount>(serde_json::json!(-2)).is_err());
        assert!(
            serde_json::from_value::<LogprobsCount>(serde_json::json!(i64::from(i32::MAX) + 1))
                .is_err()
        );
    }

    #[test]
    fn logprobs_count_decodes_msgpack_signed_and_unsigned() {
        let mut encoded = Vec::new();
        rmpv::encode::write_value(&mut encoded, &Value::from(-1)).unwrap();
        assert_eq!(
            decode_msgpack::<LogprobsCount>(&encoded).unwrap(),
            LogprobsCount::All
        );

        let encoded = encode_msgpack(&LogprobsCount::Top(7)).unwrap();
        assert_eq!(
            decode_msgpack::<LogprobsCount>(&encoded).unwrap(),
            LogprobsCount::Top(7)
        );
    }
}
