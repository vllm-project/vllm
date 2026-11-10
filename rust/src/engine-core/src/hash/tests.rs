// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Fixed vectors generated with CPython 3.12 `pickle`, `cbor2` 6.2 and
//! `get_request_block_hasher`. `tests/v1/core/test_rust_core_hash.py` tests
//! the bindings differentially against Python.

use expect_test::expect;

use super::*;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `(b"ab", (1, 300, 70000, -1, 2**40, -(2**70)), None, True, "é", (x, x),
/// ())` with `x` one object.
fn sample_value() -> Value {
    let x = Value::Ident(1, Box::new(Value::str("x")));
    Value::tuple([
        Value::Bytes(b"ab".to_vec()),
        Value::tuple([1, 300, 70000, -1, 1 << 40, -(1 << 70)].map(Value::Int)),
        Value::None,
        Value::Bool(true),
        Value::str("é"),
        Value::tuple([x.clone(), x]),
        Value::tuple([]),
    ])
}

#[test]
fn pickle_matches_python() {
    expect![[
        "8005953e00000000000000284302616294284b014d2c014a701101004affffffff8a060000000000018a090000000000000000c074944e888c02c3a9948c017894680386942974942e"
    ]]
    .assert_eq(&hex(&HashAlgo::Sha256.serialize_value(&sample_value())));
}

#[test]
fn cbor_matches_python() {
    expect!["87426162860119012c1a00011170201b0000010000000000c3493ffffffffffffffffff6f562c3a9826178617880"]
        .assert_eq(&hex(&HashAlgo::Sha256Cbor.serialize_value(&sample_value())));
}

/// The frame closes only at the start of the next `save()`, so the tuple's
/// closing opcodes stay in a frame that already reached 64 KiB.
#[test]
fn pickle_frame_boundary_matches_python() {
    let value = Value::Tuple(vec![Value::Int(70000); 13107]);
    let bytes = HashAlgo::Sha256.serialize_value(&value);
    assert_eq!(bytes.len(), 65550);
    expect!["800595030001000000000028"].assert_eq(&hex(&bytes[..12]));
    expect!["0074942e"].assert_eq(&hex(&bytes[bytes.len() - 4..]));
}

/// Prompt block hashes of each algorithm with `NONE_HASH = bytes(range(32))`,
/// block size 4, and either no extra keys or a lookahead of 2 with a cache
/// salt and LoRA.
#[test]
fn request_block_hasher_matches_python() {
    let tokens = [0, 1, 23, 24, 255, 256, 65535, 65536, 151645, 2147483647].repeat(2);
    let none_hash = BlockHash::from_slice(&(0..32).collect::<Vec<u8>>());
    let extra_keys = RequestExtraKeys {
        lora: Some((Value::str("name"), Value::str("/path"))),
        cache_salt: Some(Value::str("salt")),
    };
    let mut out = String::new();
    for algo in [
        HashAlgo::Sha256,
        HashAlgo::Sha256Cbor,
        HashAlgo::Xxhash,
        HashAlgo::XxhashCbor,
    ] {
        let plain = RequestBlockHasher::new(algo, none_hash.clone(), 4, 0);
        let hashes = plain.new_block_hashes(&tokens, 0, None, &RequestExtraKeys::default());
        out += &format!("{algo:?} 0 {hashes:?}\n");

        let lookahead = RequestBlockHasher::new(algo, none_hash.clone(), 4, 2);
        let hashes = lookahead.new_block_hashes(&tokens, 0, None, &extra_keys);
        // Resuming after existing hashes chains onto the last one.
        let resumed = lookahead.new_block_hashes(&tokens, 2, Some(&hashes[1]), &extra_keys);
        assert_eq!(resumed, hashes[2..]);
        out += &format!("{algo:?} 2 {hashes:?}\n");
    }
    expect![[r#"
        Sha256 0 [70f798233be8503c, ebe92f43acf62a5d, 133204a2c07f9f30, 8d2abae6f258cc3d, 213d9864b73613d0]
        Sha256 2 [cf083cc08794fa17, e853c9abc91e4ada, 3e0ca28efdd2a161, e5b68e78a43335b0]
        Sha256Cbor 0 [7b8c5200e68a7321, 592b29f15c98db94, bc6e63217db179a2, 9b745f1f4433027a, 2789e201dd0a6e25]
        Sha256Cbor 2 [ea91ef4232c2312b, ccf492edb1348987, 0c156229533fc1ef, e3f0bd59f8595830]
        Xxhash 0 [7319501b52b67234, 8cb8172d27ced054, 3da2586a3a8e228a, 8177786d86e4b927, d1fe606c2048619e]
        Xxhash 2 [1d147bc5211c1311, 19447d9f17f09e37, 2b69308311514a10, 6cb73fe75d8051ad]
        XxhashCbor 0 [0ce11990f3fefa8e, fe7b6bf106e04e37, 8eea59f7b8eb9b04, 38357e6e0f7fda95, 6af1d1a80d55d7fe]
        XxhashCbor 2 [5d239b802b8ac7dd, be8598c5261a333a, eac413797b843f85, e69a85b8b3c9209d]
    "#]]
    .assert_eq(&out);
}

#[test]
fn request_block_hasher_waits_for_full_blocks() {
    let hasher = RequestBlockHasher::new(HashAlgo::Sha256, BlockHash::from_slice(&[0; 32]), 4, 2);
    let no_extra_keys = RequestExtraKeys::default();
    assert!(hasher.new_block_hashes(&[1; 5], 0, None, &no_extra_keys).is_empty());
    assert_eq!(
        hasher.new_block_hashes(&[1; 6], 0, None, &no_extra_keys).len(),
        1
    );
}
