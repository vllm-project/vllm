// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The native output path: the scheduler's outputs converted to the Rust
//! frontend's wire types and sent by the output thread of
//! `vllm-engine-core-io`, instead of returned as Python `EngineCoreOutputs`.
//!
//! Payloads the step's `ModelRunnerOutput` holds as arrays are read through
//! the buffer protocol: the batch's logprobs and sampling-mask arrays once per
//! step, then sliced per request without a copy. Rare opaque values
//! (connector params, trace headers, Python-side stats) are converted by
//! encoding them with msgspec.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyBytes;
use vllm_engine_core::sched::scheduler::{EngineCoreOutput, UpdateOutput, monotonic};
use vllm_engine_core_client::protocol::dtype::{NumpyDtype, TensorDtype};
use vllm_engine_core_client::protocol::logprobs::{MaybeWireLogprobs, WireLogprobs};
use vllm_engine_core_client::protocol::output::{self as wire, RequestBatchOutputs};
use vllm_engine_core_client::protocol::sampling_mask::{MaybeWireSamplingMask, WireSamplingMask};
use vllm_engine_core_client::protocol::tensor::WireNdArray;
use vllm_engine_core_client::protocol::{decode_msgpack, decode_value};
use vllm_engine_core_io::output::{Destination, OutputConfig, OutputWriter};

use super::output::{ExtraStats, ModelOutput, kv_transfer_params_to_py, opaque_obj};
use super::request::RequestTable;

/// The engine core's output socket thread
/// (`EngineCoreProc.process_output_sockets`), for the native output path.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct EngineOutputWriter {
    pub(super) inner: Arc<OutputWriter>,
}

static LOG_ERROR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[pymethods]
impl EngineOutputWriter {
    /// Connect the output sockets (`addresses.outputs`, and the DP
    /// coordinator's `addresses.coordinator_output`) and start the thread.
    #[new]
    fn new(
        py: Python<'_>,
        client_addresses: Vec<String>,
        coordinator_address: Option<String>,
        engine_index: u32,
        aux_frame_threshold: usize,
    ) -> PyResult<Self> {
        let log_error = LOG_ERROR
            .get_or_try_init(py, || -> PyResult<_> {
                let logger = py
                    .import("vllm.logger")?
                    .call_method1("init_logger", ("vllm.v1.engine.core",))?;
                Ok(logger.getattr("error")?.unbind())
            })?
            .clone_ref(py);
        let config = OutputConfig {
            client_addresses,
            coordinator_address,
            engine_index,
            aux_frame_threshold,
        };
        let on_error = Box::new(move |error: String| {
            Python::attach(|py| {
                let _ =
                    log_error.bind(py).call1(("The engine core output thread failed: %s", error));
            });
        });
        let inner = py
            .detach(|| OutputWriter::start(config, on_error))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Send a message Python encoded; client index -1 is the DP coordinator.
    fn send_encoded(&self, client_index: i64, frames: Vec<PyBackedBytes>) -> PyResult<()> {
        let destination = match client_index {
            -1 => Destination::Coordinator,
            index => Destination::Client(
                usize::try_from(index)
                    .map_err(|_| PyValueError::new_err(format!("bad client index {index}")))?,
            ),
        };
        let frames = frames.into_iter().map(Bytes::from_owner).collect();
        self.inner.send_encoded(destination, frames);
        Ok(())
    }

    /// Send `ENGINE_CORE_DEAD` to every client, then stop the thread.
    fn send_dead(&self) {
        self.inner.send_dead();
    }

    /// `threading.Thread.join`.
    #[pyo3(signature = (timeout = None))]
    fn join(&self, py: Python<'_>, timeout: Option<f64>) {
        let timeout = timeout.map(Duration::from_secs_f64);
        py.detach(|| self.inner.join(timeout));
    }

    /// `threading.Thread.is_alive`.
    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }
}

static MSGPACK_ENCODE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The msgpack encoding of a Python object.
fn msgpack<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let py = obj.py();
    let encode = MSGPACK_ENCODE.import(py, "msgspec.msgpack", "encode")?;
    Ok(encode.call1((obj,))?.cast_into::<PyBytes>()?)
}

/// A Python object as the wire type it encodes to.
fn decode_py<T: for<'de> serde::Deserialize<'de>>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    decode_msgpack(msgpack(obj)?.as_bytes())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

static ASCONTIGUOUSARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A numpy array or CPU tensor, as the C-contiguous numpy array whose bytes
/// Python's encoder sends.
fn contiguous_ndarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let array = if obj.hasattr(intern!(py, "numpy"))? && !obj.hasattr(intern!(py, "ndim"))? {
        obj.call_method0(intern!(py, "numpy"))?
    } else {
        obj.clone()
    };
    ASCONTIGUOUSARRAY.import(py, "numpy", "ascontiguousarray")?.call1((array,))
}

/// The bytes of a C-contiguous array.
fn ndarray_bytes(array: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    let py = array.py();
    let flat = array.call_method1(intern!(py, "reshape"), (-1,))?;
    let flat = flat.call_method1(intern!(py, "view"), ("u1",))?;
    let buffer = PyBuffer::<u8>::get(&flat)?;
    Ok(Bytes::from(buffer.to_vec(py)?))
}

/// An array of a step, sliced along its first axis per request.
struct BatchArray {
    dtype: NumpyDtype,
    shape: Vec<usize>,
    data: Bytes,
}

impl BatchArray {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let array = contiguous_ndarray(obj)?;
        let dtype: String =
            array.getattr(intern!(py, "dtype"))?.getattr(intern!(py, "str"))?.extract()?;
        Ok(Self {
            dtype: dtype.parse().map_err(PyValueError::new_err)?,
            shape: array.getattr(intern!(py, "shape"))?.extract()?,
            data: ndarray_bytes(&array)?,
        })
    }

    fn whole(&self) -> WireNdArray {
        WireNdArray::from_raw_bytes(self.dtype, self.shape.clone(), self.data.clone())
    }

    /// Rows `start..end`, without a copy.
    fn rows(&self, start: usize, end: usize) -> WireNdArray {
        let num_rows = self.shape.first().copied().unwrap_or(0);
        let row_bytes = self.data.len().checked_div(num_rows).unwrap_or(0);
        let mut shape = self.shape.clone();
        shape[0] = end - start;
        WireNdArray::from_raw_bytes(
            self.dtype,
            shape,
            self.data.slice(start * row_bytes..end * row_bytes),
        )
    }
}

/// `cu_num_generated_tokens`: where each request's rows start, or `None` when
/// request `i` starts at row `i`.
fn cu_num_generated_tokens(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    let cu = obj.getattr(intern!(obj.py(), "cu_num_generated_tokens"))?;
    if cu.is_none() {
        Ok(None)
    } else {
        Ok(Some(cu.extract()?))
    }
}

/// The step's `LogprobsLists`.
struct BatchLogprobs {
    token_ids: BatchArray,
    logprobs: BatchArray,
    ranks: BatchArray,
    cu_num_generated_tokens: Option<Vec<usize>>,
}

impl BatchLogprobs {
    fn from_py(lists: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = lists.py();
        Ok(Self {
            token_ids: BatchArray::from_py(&lists.getattr(intern!(py, "logprob_token_ids"))?)?,
            logprobs: BatchArray::from_py(&lists.getattr(intern!(py, "logprobs"))?)?,
            ranks: BatchArray::from_py(&lists.getattr(intern!(py, "sampled_token_ranks"))?)?,
            cu_num_generated_tokens: cu_num_generated_tokens(lists)?,
        })
    }

    /// `LogprobsLists.slice_request`.
    fn slice_request(&self, req_index: usize, num_positions: usize) -> MaybeWireLogprobs {
        let start = self.cu_num_generated_tokens.as_ref().map_or(req_index, |cu| cu[req_index]);
        let end = start + num_positions;
        MaybeWireLogprobs::Wire(Box::new(WireLogprobs {
            logprob_token_ids: self.token_ids.rows(start, end),
            logprobs: self.logprobs.rows(start, end),
            token_ranks: self.ranks.rows(start, end),
            cu_num_generated_tokens: None,
            cu_num_generated_tokens_tensor: None,
        }))
    }
}

/// A `LogprobsTensors` of prompt logprobs, which Python sends whole.
fn prompt_logprobs(tensors: &Bound<'_, PyAny>) -> PyResult<MaybeWireLogprobs> {
    let py = tensors.py();
    let array = |name| -> PyResult<WireNdArray> {
        Ok(BatchArray::from_py(&tensors.getattr(name)?)?.whole())
    };
    Ok(MaybeWireLogprobs::Wire(Box::new(WireLogprobs {
        logprob_token_ids: array(intern!(py, "logprob_token_ids"))?,
        logprobs: array(intern!(py, "logprobs"))?,
        token_ranks: array(intern!(py, "selected_token_ranks"))?,
        cu_num_generated_tokens: None,
        cu_num_generated_tokens_tensor: None,
    })))
}

/// The step's `SamplingMaskLists` (CSR).
struct BatchSamplingMasks {
    token_ids: BatchArray,
    offsets: Vec<i64>,
    offsets_dtype: TensorDtype,
    cu_num_generated_tokens: Option<Vec<usize>>,
}

impl BatchSamplingMasks {
    fn from_py(lists: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = lists.py();
        let offsets = lists.getattr(intern!(py, "offsets"))?;
        let offsets_dtype = BatchArray::from_py(&offsets)?.dtype.scalar;
        Ok(Self {
            token_ids: BatchArray::from_py(&lists.getattr(intern!(py, "token_ids"))?)?,
            offsets: offsets.call_method0(intern!(py, "tolist"))?.extract()?,
            offsets_dtype,
            cu_num_generated_tokens: cu_num_generated_tokens(lists)?,
        })
    }

    /// `SamplingMaskLists.slice_request`.
    fn slice_request(
        &self,
        req_index: usize,
        num_positions: usize,
    ) -> PyResult<MaybeWireSamplingMask> {
        let start = self.cu_num_generated_tokens.as_ref().map_or(req_index, |cu| cu[req_index]);
        let end = start + num_positions;
        let (lo, hi) = (self.offsets[start], self.offsets[end]);
        let token_ids = self.token_ids.rows(lo as usize, hi as usize);
        let offsets = if num_positions == 1 {
            None
        } else {
            let shape = vec![num_positions + 1];
            let values = self.offsets[start..=end].iter().map(|o| o - lo);
            let array = match self.offsets_dtype {
                TensorDtype::I64 => WireNdArray::from_i64(shape, values.collect()),
                TensorDtype::I32 => {
                    WireNdArray::from_i32(shape, values.map(|o| o as i32).collect())
                }
                dtype => {
                    return Err(PyValueError::new_err(format!(
                        "{dtype:?} sampling-mask offsets"
                    )));
                }
            };
            Some(array.map_err(PyValueError::new_err)?)
        };
        Ok(MaybeWireSamplingMask::Wire(Box::new(WireSamplingMask {
            token_ids,
            offsets,
            cu_num_generated_tokens: None,
        })))
    }
}

/// The step's batch arrays, read on first use.
struct StepPayloads<'a, 'py> {
    mo: &'a ModelOutput<'py>,
    logprobs: Option<BatchLogprobs>,
    sampling_masks: Option<BatchSamplingMasks>,
}

impl<'py> StepPayloads<'_, 'py> {
    fn logprobs(&mut self) -> PyResult<&BatchLogprobs> {
        if self.logprobs.is_none() {
            self.logprobs = Some(BatchLogprobs::from_py(&self.mo.logprobs)?);
        }
        Ok(self.logprobs.as_ref().unwrap())
    }

    fn sampling_masks(&mut self) -> PyResult<&BatchSamplingMasks> {
        if self.sampling_masks.is_none() {
            self.sampling_masks = Some(BatchSamplingMasks::from_py(&self.mo.sampling_masks)?);
        }
        Ok(self.sampling_masks.as_ref().unwrap())
    }

    /// The output of one request with the payloads Python objects hold.
    fn output(
        &mut self,
        py: Python<'py>,
        table: &RequestTable,
        output: &EngineCoreOutput,
    ) -> PyResult<wire::EngineCoreOutput> {
        let mut converted = output.to_wire();
        let num_positions = output.new_token_ids.len();
        if let Some(req_index) = output.req_index {
            if output.new_logprobs {
                converted.new_logprobs =
                    Some(self.logprobs()?.slice_request(req_index, num_positions));
            }
            if output.new_sampling_mask {
                converted.new_sampling_mask =
                    Some(self.sampling_masks()?.slice_request(req_index, num_positions)?);
            }
            if output.pooling_output {
                return Err(PyNotImplementedError::new_err(
                    "pooling outputs on the native output path",
                ));
            }
            if let Some(dict) = &self.mo.prompt_logprobs
                && let Some(tensors) = dict.get_item(&output.request_id)?.filter(|t| !t.is_none())
            {
                converted.new_prompt_logprobs_tensors = Some(prompt_logprobs(&tensors)?);
            }
            if let Some(dict) = &self.mo.prompt_token_id_logprobs
                && let Some(tensor) = dict.get_item(&output.request_id)?.filter(|t| !t.is_none())
            {
                converted.prompt_token_id_logprobs =
                    Some(Box::new(BatchArray::from_py(&tensor)?.whole()));
            }
        }
        if let Some(rows) = &output.routed_experts {
            converted.routed_experts = Some(Box::new(
                BatchArray::from_py(&opaque_obj(py, rows))?.whole(),
            ));
        }
        if let Some(params) = &output.kv_transfer_params {
            converted.kv_transfer_params =
                Some(Box::new(decode_py(&kv_transfer_params_to_py(py, params)?)?));
        }
        if let Some(params) = &output.ec_transfer_params {
            converted.ec_transfer_params = Some(Box::new(decode_py(&opaque_obj(py, params))?));
        }
        if let Some(headers) = table.get(&output.request_id).and_then(|e| e.trace_headers.as_ref())
        {
            let encoded = msgpack(headers.bind(py))?;
            let value = decode_value(encoded.as_bytes())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            converted.trace_headers = Some(Box::new(value));
        }
        Ok(converted)
    }
}

/// The scheduler's update of a step in the Rust frontend's wire types, a
/// batch per client index.
pub(super) fn native_outputs(
    py: Python<'_>,
    update: &UpdateOutput,
    table: &RequestTable,
    mo: &ModelOutput<'_>,
    mut extra: ExtraStats<'_>,
    weight_version: Option<&str>,
) -> PyResult<Vec<(usize, RequestBatchOutputs)>> {
    let mut batches = Vec::with_capacity(update.engine_core_outputs.len());
    let mut payloads = StepPayloads {
        mo,
        logprobs: None,
        sampling_masks: None,
    };
    let weight_version: Option<Arc<str>> = weight_version.map(Arc::from);
    for (&client_index, eco) in &update.engine_core_outputs {
        let outputs = (eco.outputs.iter())
            .map(|o| payloads.output(py, table, o))
            .collect::<PyResult<Vec<_>>>()?;
        let scheduler_stats = match &eco.scheduler_stats {
            Some(stats) => {
                let mut converted = stats.to_wire();
                let extra = std::mem::take(&mut extra);
                if let Some(stats) = extra.kv_connector_stats {
                    let dict = stats.call_method0(intern!(py, "to_dict"))?;
                    converted.kv_connector_stats = Some(decode_py(&dict)?);
                }
                if let Some(stats) = extra.cudagraph_stats {
                    converted.cudagraph_stats = Some(decode_py(&stats)?);
                }
                if let Some(stats) = extra.perf_stats {
                    converted.perf_stats = Some(decode_py(&stats)?);
                }
                // The Rust frontend reads no `ec_connector_stats`.
                Some(Box::new(converted))
            }
            None => None,
        };
        let finished_requests = (eco.finished_requests.as_ref())
            .map(|ids| ids.iter().cloned().collect::<BTreeSet<_>>());
        batches.push((
            client_index,
            RequestBatchOutputs {
                engine_index: 0,
                outputs,
                scheduler_stats,
                timestamp: monotonic(),
                finished_requests,
                weight_version: weight_version.clone(),
            },
        ));
    }
    Ok(batches)
}
