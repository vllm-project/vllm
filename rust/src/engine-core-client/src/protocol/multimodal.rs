use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::tensor_wire::WireTensor;

/// Multimodal feature payload accepted from higher-level frontend code.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/engine/__init__.py#L88>
pub type MultiModalFeatures = Vec<MultiModalFeatureSpec>;

/// Represents a single multimodal input with its processed data and metadata.
///
/// Used to track multimodal data through processing and caching. A request
/// containing multiple multimodal items will have one `MultiModalFeatureSpec`
/// per item.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L301-L332>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiModalFeatureSpec {
    /// Represents multimodal data for this feature.
    ///
    /// Can be `None` if the item is cached, to skip IPC between API server
    /// and engine core processes.
    pub data: Option<MultiModalKwargsItem>,

    /// The input modality, e.g., `"image"`, `"audio"`, `"video"`.
    pub modality: String,

    /// The hash for caching encoder outputs (with LoRA prefix if applicable).
    pub identifier: String,

    /// The location of the `modality` tokens corresponding to this item
    /// in the prompt, e.g., `PlaceholderRange(offset=2, length=336)`.
    pub mm_position: PlaceholderRange,

    /// The hash for caching processor outputs (without LoRA prefix).
    #[serde(default)]
    pub mm_hash: Option<String>,
}

/// Placeholder location information for multi-modal data.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L118-L145>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaceholderRange {
    /// The start index of the placeholder in the prompt.
    pub offset: usize,

    /// The length of the placeholder.
    pub length: usize,

    /// A boolean mask of shape `(length,)` indicating which positions
    /// between `offset` and `offset + length` to assign embeddings to.
    #[serde(default)]
    pub is_embed: Option<WireTensor>,
}

/// A dictionary of processed keyword arguments to pass to the model,
/// corresponding to a single item in `MultiModalDataItems`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L854-L871>
pub type MultiModalKwargsItem = BTreeMap<String, MultiModalFieldElem>;

/// Represents a processed keyword argument to pass to a model for a
/// `MultiModalKwargsItem`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L348-L369>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiModalFieldElem {
    /// The tensor data of this field in `MultiModalKwargsItem`, i.e. the value
    /// of the keyword argument to be passed to the model.
    ///
    /// It may be set to `None` if it is determined that the item is cached
    /// in `EngineCore`.
    pub data: Option<NestedTensorValue>,

    /// Defines how to combine the tensor data of this field with others
    /// in order to batch multi-modal items together for model inference.
    pub field: MultiModalField,
}

/// Nested tensor payload used by multimodal keyword arguments.
///
/// Original Python type alias and wire encoding:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L218-L226>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L292-L299>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L456-L465>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NestedTensorValue {
    Tensor(WireTensor),
    Int(i64),
    Float(f64),
    List(Vec<NestedTensorValue>),
}

/// Defines how to interpret tensor data belonging to a keyword argument for
/// `MultiModalKwargsItems`, and vice versa.
///
/// Original Python definitions and wire encoding:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L385-L630>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L301-L310>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L440-L454>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MultiModalFieldWire", into = "MultiModalFieldWire")]
pub enum MultiModalField {
    Batched(MultiModalBatchedField),
    Flat(MultiModalFlatField),
    Shared(MultiModalSharedField),
}

/// Info: `MultiModalFieldConfig.batched`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L385-L502>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiModalBatchedField {
    /// If `True`, then this field is excluded from being moved to the
    /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Info: `MultiModalFieldConfig.flat` and
/// `MultiModalFieldConfig.flat_from_sizes`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L385-L397>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L505-L603>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiModalFlatField {
    /// For each multi-modal item, a slice (`dim=0`) or a tuple of slices
    /// (`dim>0`) that is used to extract the data corresponding to it.
    pub slices: Vec<MultiModalSlice>,

    /// The dimension to extract data, default to 0.
    pub dim: i32,

    /// If `True`, then this field is excluded from being moved to the
    /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Info: `MultiModalFieldConfig.shared`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L385-L397>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/multimodal/inputs.py#L606-L630>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiModalSharedField {
    pub batch_size: usize,

    /// If `True`, then this field is excluded from being moved to the
    /// accelerator when multimodal items are grouped and batched.
    pub keep_on_cpu: bool,
}

/// Python slice encoded as `(start, stop, step)`.
///
/// Original Python wire encoding:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L199-L204>
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: Option<isize>,
}

/// A single slice or a tuple of slices used by `MultiModalFlatField`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MultiModalSlice {
    Slice(SliceSpec),
    Slices(Vec<SliceSpec>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
struct MultiModalFieldWire {
    name: String,
    inner: MultiModalFieldWireInner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum MultiModalFieldWireInner {
    Batched(MultiModalBatchedField),
    Flat(MultiModalFlatField),
    Shared(MultiModalSharedField),
}

impl TryFrom<MultiModalFieldWire> for MultiModalField {
    type Error = String;

    fn try_from(value: MultiModalFieldWire) -> Result<Self, Self::Error> {
        match (value.name.as_str(), value.inner) {
            ("batched", MultiModalFieldWireInner::Batched(kwargs)) => Ok(Self::Batched(kwargs)),
            ("flat", MultiModalFieldWireInner::Flat(kwargs)) => Ok(Self::Flat(kwargs)),
            ("shared", MultiModalFieldWireInner::Shared(kwargs)) => Ok(Self::Shared(kwargs)),
            (name, _) => Err(format!(
                "mismatched or unknown multimodal field factory {name:?}"
            )),
        }
    }
}

impl From<MultiModalField> for MultiModalFieldWire {
    fn from(value: MultiModalField) -> Self {
        match value {
            MultiModalField::Batched(kwargs) => Self {
                name: "batched".to_string(),
                inner: MultiModalFieldWireInner::Batched(kwargs),
            },
            MultiModalField::Flat(kwargs) => Self {
                name: "flat".to_string(),
                inner: MultiModalFieldWireInner::Flat(kwargs),
            },
            MultiModalField::Shared(kwargs) => Self {
                name: "shared".to_string(),
                inner: MultiModalFieldWireInner::Shared(kwargs),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use rmpv::Value;

    use super::*;

    fn encode_value<T: Serialize>(value: &T) -> Value {
        let bytes = rmp_serde::to_vec_named(value).expect("encode value");
        rmpv::decode::read_value(&mut Cursor::new(bytes)).expect("decode value")
    }

    #[test]
    fn multimodal_field_serializes_to_python_factory_tuple() {
        let field = MultiModalField::Flat(MultiModalFlatField {
            slices: vec![MultiModalSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(1200),
                step: None,
            })],
            dim: 0,
            keep_on_cpu: false,
        });

        let value = encode_value(&field);
        let Value::Array(items) = value else {
            panic!("field should encode as tuple array");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_str(), Some("flat"));

        let Value::Map(kwargs) = &items[1] else {
            panic!("field kwargs should encode as map");
        };
        assert!(kwargs.iter().any(|(key, _)| key.as_str() == Some("slices")));
        assert!(kwargs.iter().any(|(key, _)| key.as_str() == Some("dim")));
        assert!(kwargs.iter().any(|(key, _)| key.as_str() == Some("keep_on_cpu")));
    }

    #[test]
    fn multimodal_field_round_trips_python_factory_tuple() {
        let field = MultiModalField::Batched(MultiModalBatchedField { keep_on_cpu: true });
        let encoded = rmp_serde::to_vec_named(&field).expect("encode field");
        let decoded: MultiModalField = rmp_serde::from_slice(&encoded).expect("decode field");
        assert_eq!(decoded, field);
    }
}
