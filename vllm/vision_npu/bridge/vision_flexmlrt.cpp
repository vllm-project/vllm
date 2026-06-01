// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// vision_flexmlrt.cpp — MODIFIED VERSION for CPU preprocessing
//
// This version accepts CPU-preprocessed [1073, 4, 1280] input instead of raw
// pixel_values

#include <FlexMLClient.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Debug logging gated by VLLM_LOGGING_LEVEL=DEBUG
inline bool is_vllm_debug() {
  static int debug_enabled = -1;
  if (debug_enabled == -1) {
    const char* level = std::getenv("VLLM_LOGGING_LEVEL");
    debug_enabled = (level && std::strcmp(level, "DEBUG") == 0) ? 1 : 0;
  }
  return debug_enabled == 1;
}

// Use stderr (not PySys_WriteStdout) so logging is safe while the GIL is
// released during model_->forward().
#define DEBUG_LOG(expr)               \
  do {                                \
    if (is_vllm_debug()) {            \
      std::ostringstream oss;         \
      oss << "[FlexMLRT] " << expr;   \
      std::cerr << oss.str() << '\n'; \
    }                                 \
  } while (0)

// Build ErtIoTypeNew tensor descriptor
static flexmlrt::client::ErtIoTypeNew makeIO(
    const std::string& name, int index, void* data, size_t size_bytes,
    const std::string& dtype, const std::vector<int64_t>& shape) {
  flexmlrt::client::ErtIoTypeNew io;
  io.name = name;
  io.idx = index;
  io.data = data;
  io.size = size_bytes;
  io.type = dtype;
  io.shape = shape;
  return io;
}

// VisionFlexMLRTModel with CPU preprocessing support
class VisionFlexMLRTModel {
 public:
  VisionFlexMLRTModel(const std::string& model_cache,
                      const std::string& device_name)
      : device_name_(device_name) {
    DEBUG_LOG(" VisionFlexMLRTModel constructor START");
    DEBUG_LOG("   model_cache: " << model_cache);
    DEBUG_LOG("   device_name: " << device_name);

    // Create options object (will be destroyed after model creation)
    flexmlrt::client::Options opts;
    opts.modelPath = model_cache;
    opts.deviceName = device_name;
    opts.subgraphName = "0";  // Specify subgraph name explicitly
    opts.executeMode = 2;     // From test_generic line 446

    DEBUG_LOG(" Creating FlexMLRT Model object...");
    try {
      model_ = std::make_unique<flexmlrt::client::Model>(opts);
      DEBUG_LOG(" FlexMLRT Model object created");
    } catch (const std::exception& e) {
      std::cerr << "[FlexMLRT ERROR] FlexMLRT Model creation threw exception: "
                << e.what() << std::endl;
      throw std::runtime_error(
          std::string("Failed to load FlexMLRT vision model: ") + e.what());
    }
    // opts goes out of scope here - memory automatically freed

    if (!model_->good()) {
      std::cerr << "[FlexMLRT ERROR] model->good() returned false" << std::endl;
      throw std::runtime_error(
          "FlexMLRT vision model creation failed - check model cache and "
          "device availability");
    }
    DEBUG_LOG(" model->good() returned true");
    DEBUG_LOG(" VisionFlexMLRTModel constructor END (opts memory released)");
  }

  // Forward pass with CPU-preprocessed input [1073, 4, 1280]
  py::array_t<float> forward(py::array_t<float> preprocessed_input) {
    DEBUG_LOG(" forward() START (CPU-preprocessed input)");

    auto buf = preprocessed_input.request();
    DEBUG_LOG(" Input ndim: " << buf.ndim);

    if (buf.ndim != 3) {
      throw std::runtime_error(
          "preprocessed_input must be 3D array [1073, 4, 1280]");
    }

    int64_t dim0 = buf.shape[0];  // 1073
    int64_t dim1 = buf.shape[1];  // 4
    int64_t dim2 = buf.shape[2];  // 1280

    DEBUG_LOG(" Input shape: [" << dim0 << ", " << dim1 << ", " << dim2 << "]");

    if (dim0 != 1073 || dim1 != 4 || dim2 != 1280) {
      throw std::runtime_error(
          "Expected input shape [1073, 4, 1280], got [" + std::to_string(dim0) +
          ", " + std::to_string(dim1) + ", " + std::to_string(dim2) + "]");
    }

    // Build input tensors
    std::vector<flexmlrt::client::ErtIoTypeNew> ifms;

    // Input name from NPU partition ONNX: "/blocks/Gather_output_0"
    ifms.push_back(makeIO("/blocks/Gather_output_0", 0, buf.ptr,
                          dim0 * dim1 * dim2 * sizeof(float), "float32",
                          {dim0, dim1, dim2}));
    DEBUG_LOG(" Input tensor built: /blocks/Gather_output_0 [1073, 4, 1280]");

    // Output tensor
    // From NPU partition ONNX: "/merger/merger/mlp/mlp.2/Gemm_output_0" [1073,
    // 3584]
    int64_t out_dim0 = 1073;
    int64_t out_dim1 = 3584;

    std::vector<float> output_buf(out_dim0 * out_dim1);
    std::vector<flexmlrt::client::ErtIoTypeNew> ofms;
    ofms.push_back(makeIO("/merger/merger/mlp/mlp.2/Gemm_output_0", 0,
                          output_buf.data(), output_buf.size() * sizeof(float),
                          "float32", {out_dim0, out_dim1}));
    DEBUG_LOG(
        " Output tensor built: /merger/merger/mlp/mlp.2/Gemm_output_0 [1073, "
        "3584]");

    std::vector<flexmlrt::client::ErtIoTypeNew> wts;

    // Run NPU inference
    DEBUG_LOG(" Calling model->forward()...");
    DEBUG_LOG(" Releasing GIL to allow GPU parallelization...");
    try {
      // CRITICAL: Release GIL during NPU execution to allow GPU to run in
      // parallel NPU inference takes ~11 seconds - other Python threads must be
      // able to proceed
      py::gil_scoped_release release;
      model_->forward(ifms, ofms, wts);
      // GIL automatically reacquired when 'release' goes out of scope
      DEBUG_LOG(" model->forward() returned successfully (GIL reacquired)");
    } catch (const std::exception& e) {
      std::cerr << "[FlexMLRT ERROR] model->forward() threw exception: "
                << e.what() << std::endl;
      throw std::runtime_error(std::string("FlexMLRT forward failed: ") +
                               e.what());
    }

    // Copy output to numpy array
    DEBUG_LOG(" Copying output to numpy array...");
    py::array_t<float> result({out_dim0, out_dim1});
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, output_buf.data(),
                output_buf.size() * sizeof(float));

    // Explicitly clear temporary buffers (helps with memory fragmentation)
    output_buf.clear();
    output_buf.shrink_to_fit();
    ifms.clear();
    ofms.clear();

    DEBUG_LOG(" forward() END (temporary buffers released)");

    return result;
  }

  int output_dim() const {
    return 3584;  // Fixed for Qwen2.5-VL
  }

 private:
  std::unique_ptr<flexmlrt::client::Model> model_;
  std::string device_name_;
  // Removed unused members:
  // - std::unique_ptr<RaiLoader> rai_loader_; (never initialized or used)
  // - int output_dim_; (unused, output_dim() returns hardcoded 3584)
};

// pybind11 module
PYBIND11_MODULE(_vision_flexmlrt_cpu, m) {
  m.doc() = "FlexMLRT vision model with CPU preprocessing support";

  py::class_<VisionFlexMLRTModel>(m, "VisionFlexMLRTModel")
      .def(py::init<std::string, std::string>(), py::arg("model_cache"),
           py::arg("device_name") = "stx",
           "Load FlexMLRT vision model\n\n"
           "Args:\n"
           "    model_cache: Path to VAIP model cache (vaiml_par_0 directory)\n"
           "    device_name: XRT device name (default: 'stx')")
      .def("forward", &VisionFlexMLRTModel::forward,
           py::arg("preprocessed_input"),
           "Run vision encoding on NPU with CPU-preprocessed input\n\n"
           "Args:\n"
           "    preprocessed_input: [1073, 4, 1280] float32 array "
           "(CPU-preprocessed)\n\n"
           "Returns:\n"
           "    embeddings: [1073, 3584] float32 array")
      .def("output_dim", &VisionFlexMLRTModel::output_dim,
           "Get output embedding dimension");
}
