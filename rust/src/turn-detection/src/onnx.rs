// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Just enough of the ONNX protobuf to pull weights out of a model file.
//!
//! The pinned models are read as they are published (same bytes, same sha256), so
//! nothing is converted or redistributed. Only tensors are extracted: graph
//! initializers and the `value` of `Constant` nodes, including those inside `If`
//! subgraphs (Silero keeps its weights in the 16 kHz / 8 kHz branches). The graph
//! itself is not interpreted; the forward pass is written by hand.

use std::collections::HashMap;

#[derive(Debug)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: TensorData,
}

#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl Tensor {
    pub fn f32(&self) -> Result<&[f32], String> {
        match &self.data {
            TensorData::F32(v) => Ok(v),
            other => Err(format!("expected f32 tensor, got {}", other.kind())),
        }
    }
}

impl TensorData {
    fn kind(&self) -> &'static str {
        match self {
            TensorData::F32(_) => "f32",
            TensorData::I8(_) => "i8",
            TensorData::U8(_) => "u8",
            TensorData::I32(_) => "i32",
            TensorData::I64(_) => "i64",
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("varint too long".into())
    }
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or("truncated field")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    /// Next (field number, wire type); the caller reads or skips the value.
    fn key(&mut self) -> Result<(u32, u8), String> {
        let k = self.varint()?;
        Ok(((k >> 3) as u32, (k & 7) as u8))
    }
    fn skip(&mut self, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.pos += 4,
            w => return Err(format!("unsupported wire type {w}")),
        }
        if self.pos > self.buf.len() {
            return Err("truncated fixed field".into());
        }
        Ok(())
    }
}

/// Every named tensor in the model: initializers by name, `Constant` nodes by output name.
pub fn load_tensors(model: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    Ok(load_graph(model)?.0)
}

/// A top-level graph node: operator and input names (enough to find anonymous weights).
#[derive(Debug)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
}

/// Named tensors plus the top-level nodes in graph order.
pub fn load_graph(model: &[u8]) -> Result<(HashMap<String, Tensor>, Vec<Node>), String> {
    let mut out = HashMap::new();
    let mut nodes = Vec::new();
    let mut r = Reader::new(model);
    while !r.done() {
        let (field, wire) = r.key()?;
        if field == 7 && wire == 2 {
            let g = r.bytes()?;
            graph(g, &mut out)?;
            let mut gr = Reader::new(g);
            while !gr.done() {
                let (f, w) = gr.key()?;
                if f == 1 && w == 2 {
                    let mut nr = Reader::new(gr.bytes()?);
                    let mut node = Node {
                        op: String::new(),
                        inputs: Vec::new(),
                    };
                    while !nr.done() {
                        let (nf, nw) = nr.key()?;
                        match (nf, nw) {
                            (1, 2) => {
                                node.inputs.push(String::from_utf8_lossy(nr.bytes()?).into_owned())
                            }
                            (4, 2) => node.op = String::from_utf8_lossy(nr.bytes()?).into_owned(),
                            _ => nr.skip(nw)?,
                        }
                    }
                    nodes.push(node);
                } else {
                    gr.skip(w)?;
                }
            }
        } else {
            r.skip(wire)?;
        }
    }
    Ok((out, nodes))
}

fn graph(buf: &[u8], out: &mut HashMap<String, Tensor>) -> Result<(), String> {
    let mut r = Reader::new(buf);
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => node(r.bytes()?, out)?,
            (5, 2) => {
                let (name, t) = tensor(r.bytes()?)?;
                out.insert(name, t);
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

fn node(buf: &[u8], out: &mut HashMap<String, Tensor>) -> Result<(), String> {
    let mut r = Reader::new(buf);
    let mut outputs = Vec::new();
    let mut op = String::new();
    let mut value = None;
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (2, 2) => outputs.push(String::from_utf8_lossy(r.bytes()?).into_owned()),
            (4, 2) => op = String::from_utf8_lossy(r.bytes()?).into_owned(),
            (5, 2) => {
                let mut a = Reader::new(r.bytes()?);
                let mut name = String::new();
                let mut t = None;
                let mut graphs = Vec::new();
                while !a.done() {
                    let (f, w) = a.key()?;
                    match (f, w) {
                        (1, 2) => name = String::from_utf8_lossy(a.bytes()?).into_owned(),
                        (5, 2) => t = Some(tensor(a.bytes()?)?.1),
                        (6, 2) => graphs.push(a.bytes()?),
                        _ => a.skip(w)?,
                    }
                }
                for g in graphs {
                    graph(g, out)?;
                }
                if name == "value" {
                    value = t;
                }
            }
            _ => r.skip(wire)?,
        }
    }
    if op == "Constant"
        && let (Some(name), Some(t)) = (outputs.into_iter().next(), value)
    {
        out.insert(name, t);
    }
    Ok(())
}

fn tensor(buf: &[u8]) -> Result<(String, Tensor), String> {
    let mut r = Reader::new(buf);
    let mut dims = Vec::new();
    let mut dtype = 0u64;
    let mut name = String::new();
    let mut raw: Option<&[u8]> = None;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut int64s = Vec::new();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 0) => dims.push(r.varint()? as usize),
            (1, 2) => {
                let mut p = Reader::new(r.bytes()?);
                while !p.done() {
                    dims.push(p.varint()? as usize);
                }
            }
            (2, 0) => dtype = r.varint()?,
            (4, 2) => floats.extend(
                r.bytes()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            (5, 2) => {
                let mut p = Reader::new(r.bytes()?);
                while !p.done() {
                    ints.push(p.varint()? as i64);
                }
            }
            (7, 2) => {
                let mut p = Reader::new(r.bytes()?);
                while !p.done() {
                    int64s.push(p.varint()? as i64);
                }
            }
            (8, 2) => name = String::from_utf8_lossy(r.bytes()?).into_owned(),
            (9, 2) => raw = Some(r.bytes()?),
            _ => r.skip(wire)?,
        }
    }
    let n: usize = dims.iter().product();
    // TensorProto.DataType: 1 FLOAT, 2 UINT8, 3 INT8, 6 INT32, 7 INT64.
    let data = match (dtype, raw) {
        (1, Some(b)) => TensorData::F32(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        (1, None) => TensorData::F32(floats),
        (2, Some(b)) => TensorData::U8(b.to_vec()),
        (2, None) => TensorData::U8(ints.iter().map(|&v| v as u8).collect()),
        (3, Some(b)) => TensorData::I8(b.iter().map(|&v| v as i8).collect()),
        (3, None) => TensorData::I8(ints.iter().map(|&v| v as i8).collect()),
        (6, Some(b)) => TensorData::I32(
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        (6, None) => TensorData::I32(ints.iter().map(|&v| v as i32).collect()),
        (7, Some(b)) => TensorData::I64(
            b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        ),
        (7, None) => TensorData::I64(int64s),
        (t, _) => return Err(format!("tensor {name}: unsupported data type {t}")),
    };
    let t = Tensor { dims, data };
    let got = match &t.data {
        TensorData::F32(v) => v.len(),
        TensorData::I8(v) => v.len(),
        TensorData::U8(v) => v.len(),
        TensorData::I32(v) => v.len(),
        TensorData::I64(v) => v.len(),
    };
    if got != n {
        return Err(format!("tensor {name}: {got} values for dims {:?}", t.dims));
    }
    Ok((name, t))
}
