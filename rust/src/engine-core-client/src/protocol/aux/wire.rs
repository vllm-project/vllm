use rmpv::Value;
use serde::{Deserialize, Deserializer};
use serde_tuple::Deserialize_tuple;

/// Tensors and ndarrays are encoded with this extension type in Python.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/serial_utils.py#L42>
const CUSTOM_TYPE_RAW_VIEW: i8 = 3;

/// Python wire representation of `LogprobsLists` / `LogprobsTensors` before aux-frame
/// references and raw-view payloads are resolved.
///
/// This keeps the tuple shape emitted by Python engine-core intact so the outer DTO can still
/// be deserialized through serde.
#[derive(Debug, Clone, PartialEq, Deserialize_tuple)]
pub struct WireLogprobs {
    pub logprob_token_ids: WireNdArray,
    pub logprobs: WireNdArray,
    pub token_ranks: WireNdArray,
    #[serde(default)]
    pub cu_num_generated_tokens: Option<Vec<usize>>,
}

/// Python ndarray/tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This matches the custom msgpack representation built by Python `serial_utils.encode_ndarray`
/// / `encode_tensor`.
#[derive(Debug, Clone, PartialEq, Deserialize_tuple)]
pub struct WireNdArray {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: WireArrayData,
}

/// Python array payload reference inside [`WireNdArray`].
///
/// The data can be either an inline msgpack raw-view extension or an index into the multipart
/// aux-frame list carried alongside the primary msgpack frame.
#[derive(Debug, Clone, PartialEq)]
pub enum WireArrayData {
    /// The index of the aux frame where the raw bytes of this array/tensor are stored.
    AuxIndex(usize),
    /// The raw bytes of this array/tensor.
    RawView(Vec<u8>),
}

impl<'de> Deserialize<'de> for WireArrayData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => Ok(Self::RawView(bytes)),
            Value::Ext(tag, _) => Err(serde::de::Error::custom(format!(
                "unsupported extension type code {tag}"
            ))),
            Value::Integer(index) => index
                .as_u64()
                .map(|index| Self::AuxIndex(index as usize))
                .ok_or_else(|| {
                    serde::de::Error::custom("aux frame index must be a non-negative integer")
                }),
            other => Err(serde::de::Error::custom(format!(
                "expected raw-view ext or aux frame index, got {other:?}"
            ))),
        }
    }
}
