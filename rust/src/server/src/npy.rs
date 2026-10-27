// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `numpy2base64`-compatible ndarray serialization shared by response routes.

use std::result::Result;

use base64::Engine as _;
use vllm_engine_core_client::protocol::dtype::Endianness;
use vllm_engine_core_client::protocol::tensor::WireNdArray;

use crate::error::{ApiError, bail_server_error};

/// Encode an ndarray as Python `numpy2base64` (`np.save`, then base64) does.
///
/// The dtype descriptor, byte order, shape, and row order are preserved.
pub(crate) fn npy_base64(array: &WireNdArray) -> Result<String, ApiError> {
    let Some(data) = array.data.as_raw_view() else {
        bail_server_error!("ndarray data was not resolved");
    };
    // NumPy writes `|` for single-byte types; vLLM hosts are little-endian.
    let byte_order = match array.dtype.endianness {
        _ if array.dtype.scalar.element_size() == 1 => '|',
        Endianness::Big => '>',
        Endianness::Little | Endianness::Native => '<',
    };
    let code = &array.dtype.to_string()[1..];
    let dims: Vec<_> = array.shape.iter().map(ToString::to_string).collect();
    let shape = match dims.as_slice() {
        [dim] => format!("({dim},)"),
        dims => format!("({})", dims.join(", ")),
    };
    let mut header =
        format!("{{'descr': '{byte_order}{code}', 'fortran_order': False, 'shape': {shape}, }}");
    // NumPy reserves room to grow the first axis in place.
    if let Some(first) = dims.first() {
        header.push_str(&" ".repeat(21 - first.len()));
    }
    header.push_str(&" ".repeat(64 - (11 + header.len()) % 64));
    header.push('\n');

    let mut npy = Vec::with_capacity(10 + header.len() + data.len());
    npy.extend_from_slice(b"\x93NUMPY\x01\x00");
    npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
    npy.extend_from_slice(header.as_bytes());
    npy.extend_from_slice(data);
    Ok(base64::engine::general_purpose::STANDARD.encode(npy))
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::dtype::TensorDtype;

    use super::*;

    #[test]
    fn routed_experts_match_numpy2base64() {
        // Expected values are numpy2base64 of the same arrays.
        let cases = [
            (
                WireNdArray::from_i32(vec![2, 1, 2], vec![8, -1, 2, 0]).unwrap(),
                "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDEsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoIAAAA/////wIAAAAAAAAA",
            ),
            (
                // np.arange(12, dtype=np.uint8).reshape(2, 3, 2)
                WireNdArray::from_raw(TensorDtype::U8, vec![2, 3, 2], (0..12).collect()),
                "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQIDBAUGBwgJCgs=",
            ),
            (
                // np.arange(6, dtype="<i4").reshape(3, 1, 2)
                WireNdArray::from_i32(vec![3, 1, 2], (0..6).collect()).unwrap(),
                "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsIDEsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAQAAAAIAAAADAAAABAAAAAUAAAA=",
            ),
            (
                // np.arange(3, dtype="<u2")
                WireNdArray::from_raw(TensorDtype::U16, vec![3], vec![0, 0, 1, 0, 2, 0]),
                "k05VTVBZAQB2AHsnZGVzY3InOiAnPHUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAEAAgA=",
            ),
            (
                // np.zeros((0, 2, 2), dtype=np.uint8)
                WireNdArray::from_raw(TensorDtype::U8, vec![0, 2, 2], Vec::new()),
                "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsIDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=",
            ),
        ];
        for (array, expected) in cases {
            assert_eq!(npy_base64(&array).unwrap(), expected, "{array:?}");
        }
    }

    #[test]
    fn prompt_token_id_logprobs_match_numpy2base64() {
        let scores =
            WireNdArray::from_f32(vec![2, 3], vec![-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]).unwrap();
        // numpy2base64(np.array([[-0.5, -1.5, -2.5], [-3.5, -4.5, -5.5]], np.float32))
        assert_eq!(
            npy_base64(&scores).unwrap(),
            "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAC/AADAvwAAIMAAAGDAAACQwAAAsMA="
        );
    }
}
