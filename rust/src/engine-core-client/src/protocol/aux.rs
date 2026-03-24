mod array;
#[cfg(test)]
mod tests;
mod wire;

use std::ops::{Deref, DerefMut};

use enum_as_inner::EnumAsInner;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Deserializer, Serialize};

use self::wire::*;
use super::{EngineCoreOutput, EngineCoreOutputs};
use crate::error::{Error, Result};

/// Decoded logprobs payload for one engine-core output.
///
/// This is the shared Rust representation for both Python `LogprobsLists` and `LogprobsTensors`.
/// The Python encoder emits the same `(dtype, shape, data)` wire shape for both, so the Rust client
/// keeps one owned array form after resolving the msgpack + aux-frame payload.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/v1/outputs.py#L23-L56>
#[derive(Debug, Clone, PartialEq)]
pub struct Logprobs {
    /// `[num_reqs x num_generated_tokens, max_num_logprobs + 1]`
    pub logprob_token_ids: Array2<i32>,
    /// `[num_reqs x num_generated_tokens, max_num_logprobs + 1]`
    pub logprobs: Array2<f32>,
    /// `[num_reqs x num_generated_tokens]`
    ///
    /// Python uses the field name `sampled_token_ranks` for sample logprobs and
    /// `selected_token_ranks` for prompt logprobs. Rust keeps one neutral field because both
    /// payloads share the same wire representation.
    pub token_ranks: Array1<i32>,
    /// `[num_reqs]`
    ///
    /// Used for slicing outputs when different requests generated different numbers of tokens in
    /// the current engine step.
    pub cu_num_generated_tokens: Option<Vec<usize>>,
}

/// Output field wrapper that is initially deserialized from the Python wire shape, then resolved
/// into [`Logprobs`] before the decoded message is returned to callers.
#[derive(Clone, PartialEq, Debug, EnumAsInner)]
pub enum MaybeWireLogprobs {
    /// The logprobs are still in the wire format and need to be resolved by looking up aux frames
    /// and decoding raw views. Should only be used internally during deserialization.
    Wire(WireLogprobs),
    /// The actual decoded logprobs value,
    Direct(Logprobs),
}

impl Deref for MaybeWireLogprobs {
    type Target = Logprobs;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Wire(_) => panic!("Logprobs is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl DerefMut for MaybeWireLogprobs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Wire(_) => panic!("Logprobs is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for MaybeWireLogprobs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // When deserializing, it's always in the wire form.
        WireLogprobs::deserialize(deserializer).map(Self::Wire)
    }
}

impl Serialize for MaybeWireLogprobs {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Err(serde::ser::Error::custom(
            "MaybeWireLogprobs is decode-only and cannot be serialized",
        ))
    }
}

impl MaybeWireLogprobs {
    /// Resolve the wire representation into decoded logprobs by looking up aux frames and decoding
    /// raw views as needed.
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

impl EngineCoreOutputs {
    /// Resolve all wire-format fields in-place by looking up aux frames and decoding raw-view
    /// payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        for output in &mut self.outputs {
            output.resolve_in_place(frames)?;
        }
        Ok(())
    }
}

impl EngineCoreOutput {
    /// Resolve all wire-format fields in-place by looking up aux frames and decoding raw-view
    /// payloads as needed.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        self.new_logprobs = (self.new_logprobs.take())
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = (self.new_prompt_logprobs_tensors.take())
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        Ok(())
    }
}

impl WireLogprobs {
    /// Resolve the wire-format logprobs into decoded [`Logprobs`] by looking up aux frames and
    /// decoding raw views as needed.
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Logprobs>
    where
        Frame: AsRef<[u8]>,
    {
        Ok(Logprobs {
            logprob_token_ids: array::decode_array2_i32(
                self.logprob_token_ids,
                &format!("{field_prefix}.logprob_token_ids"),
                frames,
            )?,
            logprobs: array::decode_array2_f32(
                self.logprobs,
                &format!("{field_prefix}.logprobs"),
                frames,
            )?,
            token_ranks: array::decode_array1_i32(
                self.token_ranks,
                &format!("{field_prefix}.token_ranks"),
                frames,
            )?,
            cu_num_generated_tokens: self.cu_num_generated_tokens,
        })
    }
}

/// Decode one ordinary or multipart engine-core output message into the strong typed public
/// protocol shape.
pub fn decode_engine_core_outputs<Frame>(frames: &[Frame]) -> Result<EngineCoreOutputs>
where
    Frame: AsRef<[u8]>,
{
    let first_frame = frames
        .first()
        .ok_or_else(|| Error::ValueDecodeExt("missing output frame".to_string()))?;

    let mut outputs: EngineCoreOutputs = rmp_serde::from_slice(first_frame.as_ref())?;
    outputs.resolve_in_place(frames)?;
    Ok(outputs)
}
