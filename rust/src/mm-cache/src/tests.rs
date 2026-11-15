// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Interop tests for the shm cache against a real Python decoder.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use super::{MmProcessorShmCache, MmShmCacheConfig};
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};
use vllm_engine_core_client::protocol::tensor::WireTensor;

/// Build an item with two inline tensors (raw-view ext, no aux frames) and a
/// plain int field.
fn interop_item() -> MmKwargsItem {
    let batched = || MmField::Batched(MmBatchedField { keep_on_cpu: false });
    BTreeMap::from([
        (
            "offset".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Int(7)),
                field: batched(),
            },
        ),
        (
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(
                    WireTensor::from_f32(vec![128], (0..128).map(|value| value as f32).collect())
                        .unwrap(),
                )),
                field: batched(),
            },
        ),
        (
            "small".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(
                    WireTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap(),
                )),
                field: batched(),
            },
        ),
    ])
}

fn int_field(item: &MmKwargsItem, key: &str) -> i64 {
    match item[key].data.as_ref() {
        Some(MmKwargValue::Int(value)) => *value,
        other => panic!("expected int field {key:?}, got {other:?}"),
    }
}

/// The 32-byte handle signature of an address item, hex-encoded for the
/// Python validation script.
fn signature_hex(item: &MmKwargsItem) -> String {
    match item["signature"].data.as_ref() {
        Some(MmKwargValue::List(bytes)) => bytes
            .iter()
            .map(|value| match value {
                MmKwargValue::Int(byte) => u8::try_from(*byte).expect("signature byte"),
                other => panic!("expected int signature byte, got {other:?}"),
            })
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        other => panic!("expected list field \"signature\", got {other:?}"),
    }
}

#[test]
fn python_decodes_rust_written_shm_item() {
    // Skip gracefully when the uv-run Python toolchain is unavailable.
    if Command::new("uv")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping python shm compat test: `uv` is not available");
        return;
    }

    let data_buffer_size = 1 << 20;
    let cache = MmProcessorShmCache::create(MmShmCacheConfig {
        shm_name: MmShmCacheConfig::unique_shm_name(),
        data_buffer_size,
        max_object_size: 1 << 16,
    })
    .unwrap();
    cache.set_n_readers(1);

    let address_item = cache.put_or_inline("hash-interop", interop_item());
    let address = int_field(&address_item, "address");
    let monotonic_id = int_field(&address_item, "monotonic_id");
    let signature = signature_hex(&address_item);

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/mm_shm_compat.py");
    let output = Command::new(&script)
        .arg(format!("/dev/shm/{}", cache.shm_name()))
        .arg(address.to_string())
        .arg(monotonic_id.to_string())
        .arg(data_buffer_size.to_string())
        .arg("hash-interop")
        .arg(signature)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {script:?}: {error}"));
    assert!(
        output.status.success(),
        "python shm compat script failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "OK");
}
