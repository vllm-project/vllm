use std::collections::BTreeSet;

use bytes::Bytes;
use rmpv::Value;

use super::{Logprobs, PositionLogprobs, TokenLogprob, decode_engine_core_outputs};
use crate::protocol::EngineCoreFinishReason;

fn encode_value(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).unwrap();
    out
}

fn output_wire_with_custom_fields(
    new_logprobs: Option<Value>,
    prompt_logprobs: Option<Value>,
) -> Value {
    Value::Array(vec![
        Value::from(0),
        Value::Array(vec![Value::Array(vec![
            Value::from("req-1"),
            Value::Array(vec![Value::from(7), Value::from(8)]),
            new_logprobs.unwrap_or(Value::Nil),
            prompt_logprobs.unwrap_or(Value::Nil),
            Value::Nil,
            Value::from(EngineCoreFinishReason::Length as u8),
        ])]),
        Value::Nil,
        Value::from(0.0),
        Value::Nil,
        Value::Array(vec![Value::from("req-1")]),
    ])
}

fn ndarray_value(dtype: &str, shape: &[usize], data: Value) -> Value {
    Value::Array(vec![
        Value::from(dtype),
        Value::Array(shape.iter().copied().map(Value::from).collect()),
        data,
    ])
}

fn inline_logprobs_value() -> Value {
    let ids = Value::Ext(
        3,
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let probs = Value::Ext(
        3,
        vec![
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
        ],
    );
    let ranks = Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    Value::Array(vec![
        ndarray_value("<i8", &[2, 3], ids),
        ndarray_value("<f4", &[2, 3], probs),
        ndarray_value("<i8", &[2], ranks),
        Value::Nil,
    ])
}

fn inline_prompt_logprobs_value() -> Value {
    let ids = Value::Ext(
        3,
        vec![
            10, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0,
            0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let probs = Value::Ext(
        3,
        vec![
            0, 0, 32, 65, 0, 0, 48, 65, 0, 0, 64, 65, 0, 0, 80, 65, 0, 0, 96, 65, 0, 0, 112, 65,
        ],
    );
    let ranks = Value::Ext(3, vec![3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
    Value::Array(vec![
        ndarray_value("int64", &[2, 3], ids),
        ndarray_value("float32", &[2, 3], probs),
        ndarray_value("int64", &[2], ranks),
        Value::Nil,
    ])
}

fn expected_sample_logprobs() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 1,
                        logprob: 1.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 2,
                        logprob: 2.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 3,
                        logprob: 3.0,
                        rank: 2,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 4,
                        logprob: 4.0,
                        rank: 2,
                    },
                    TokenLogprob {
                        token_id: 5,
                        logprob: 5.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 6,
                        logprob: 6.0,
                        rank: 2,
                    },
                ],
            },
        ],
    }
}

fn expected_prompt_logprobs() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 10,
                        logprob: 10.0,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 11,
                        logprob: 11.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 12,
                        logprob: 12.0,
                        rank: 2,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 13,
                        logprob: 13.0,
                        rank: 4,
                    },
                    TokenLogprob {
                        token_id: 14,
                        logprob: 14.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 15,
                        logprob: 15.0,
                        rank: 2,
                    },
                ],
            },
        ],
    }
}

#[test]
fn decodes_inline_new_logprobs() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(inline_logprobs_value()),
        None,
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap();

    let logprobs = decoded.outputs[0]
        .new_logprobs
        .clone()
        .unwrap()
        .into_direct()
        .unwrap();
    assert_eq!(logprobs, expected_sample_logprobs());
    assert_eq!(
        decoded.finished_requests,
        Some(BTreeSet::from(["req-1".to_string()]))
    );
}

#[test]
fn decodes_multipart_new_logprobs() {
    let frames = vec![
        Bytes::from(encode_value(&output_wire_with_custom_fields(
            Some(Value::Array(vec![
                ndarray_value("<i8", &[2, 3], Value::from(1)),
                ndarray_value("<f4", &[2, 3], Value::from(2)),
                ndarray_value("<i8", &[2], Value::from(3)),
                Value::Nil,
            ])),
            None,
        ))),
        Bytes::from_static(&[
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ]),
        Bytes::from_static(&[
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
        ]),
        Bytes::from_static(&[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let decoded = decode_engine_core_outputs(&frames).unwrap();

    let logprobs = decoded.outputs[0]
        .new_logprobs
        .clone()
        .unwrap()
        .into_direct()
        .unwrap();
    assert_eq!(logprobs, expected_sample_logprobs());
}

#[test]
fn decodes_inline_prompt_logprobs() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        None,
        Some(inline_prompt_logprobs_value()),
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap();

    let logprobs = decoded.outputs[0]
        .new_prompt_logprobs_tensors
        .clone()
        .unwrap()
        .into_direct()
        .unwrap();
    assert_eq!(logprobs, expected_prompt_logprobs());
}

#[test]
fn decodes_big_endian_payloads() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(Value::Array(vec![
            ndarray_value(">i4", &[1, 2], Value::Ext(3, vec![0, 0, 0, 1, 0, 0, 0, 2])),
            ndarray_value(
                ">f4",
                &[1, 2],
                Value::Ext(3, vec![63, 128, 0, 0, 64, 0, 0, 0]),
            ),
            ndarray_value(">i4", &[1], Value::Ext(3, vec![0, 0, 0, 3])),
            Value::Nil,
        ])),
        None,
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap();
    let logprobs = decoded.outputs[0]
        .new_logprobs
        .clone()
        .unwrap()
        .into_direct()
        .unwrap();
    assert_eq!(
        logprobs,
        Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 1,
                        logprob: 1.0,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 2,
                        logprob: 2.0,
                        rank: 1,
                    },
                ],
            }],
        }
    );
}

#[test]
fn rejects_non_none_cu_num_generated_tokens() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(Value::Array(vec![
            ndarray_value("<i8", &[1, 1], Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0])),
            ndarray_value("<f4", &[1, 1], Value::Ext(3, vec![0, 0, 128, 63])),
            ndarray_value("<i8", &[1], Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0])),
            Value::Array(vec![Value::from(0usize), Value::from(1usize)]),
        ])),
        None,
    )))];

    let error = decode_engine_core_outputs(&frames).unwrap_err();
    assert_eq!(error.to_string(), "messagepack ext value decode failed");
    let crate::error::Error::ValueDecodeExt(message) = error else {
        panic!("expected ValueDecodeExt");
    };
    assert_eq!(
        message,
        "new_logprobs.cu_num_generated_tokens: expected None for per-request engine-core logprobs payload, got [0, 1]"
    );
}
