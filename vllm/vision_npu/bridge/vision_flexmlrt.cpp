// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// FlexMLRT NPU vision bridge: accepts CPU-preprocessed [1073, 4, 1280] input.
// Model weights are loaded from a compiled .rai cache (see test_generic -r).

#include <FlexMLClient.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rai_loader.h"

namespace py = pybind11;
namespace fs = std::filesystem;

// Debug logging gated by VLLM_LOGGING_LEVEL=DEBUG
inline bool is_vllm_debug() {
  static int debug_enabled = -1;
  if (debug_enabled == -1) {
    const char* level = std::getenv("VLLM_LOGGING_LEVEL");
    debug_enabled = (level && std::strcmp(level, "DEBUG") == 0) ? 1 : 0;
  }
  return debug_enabled == 1;
}

#define DEBUG_LOG(expr)               \
  do {                                \
    if (is_vllm_debug()) {            \
      std::ostringstream oss;         \
      oss << "[FlexMLRT] " << expr;   \
      std::cerr << oss.str() << '\n'; \
    }                                 \
  } while (0)

static bool pathEndsWithRai(const fs::path& path) {
  const std::string ext = path.extension().string();
  if (ext.size() != 4) {
    return false;
  }
  return (ext[0] == '.' && (ext[1] == 'r' || ext[1] == 'R') &&
          (ext[2] == 'a' || ext[2] == 'A') && (ext[3] == 'i' || ext[3] == 'I'));
}

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

class VisionFlexMLRTModel {
 public:
  VisionFlexMLRTModel(const std::string& rai_path,
                      const std::string& device_name)
      : device_name_(device_name) {
    static constexpr const char* kSubgraphName = "vaiml_par_0";

    DEBUG_LOG("VisionFlexMLRTModel constructor START");
    DEBUG_LOG("  rai_path: " << rai_path);
    DEBUG_LOG("  device_name: " << device_name);
    DEBUG_LOG("  subgraph_name: " << kSubgraphName);

    fs::path rai_file = fs::absolute(fs::path(rai_path));
    if (!pathEndsWithRai(rai_file)) {
      throw std::runtime_error(
          "VLLM_VISION_NPU_CACHE must be a .rai file path, got: " + rai_path);
    }
    if (!rai_loader_.load(rai_file)) {
      throw std::runtime_error("Failed to load RAI file: " + rai_file.string());
    }

    flexmlrt::client::Options opts;
    opts.deviceName = device_name;
    opts.subgraphName = kSubgraphName;
    opts.executeMode = 2;  // test_generic default

    opts.extOptions["fbs_buffer"] = static_cast<uint8_t*>(rai_loader_.data());
    opts.extOptions["fbs_buffer_size"] = rai_loader_.size();
    opts.extOptions["cache_dir"] = rai_file.parent_path().string();

    DEBUG_LOG("  RAI size bytes: " << rai_loader_.size());
    DEBUG_LOG("  cache_dir: " << rai_file.parent_path().string());

    DEBUG_LOG("Creating FlexMLRT Model object...");
    try {
      model_ = std::make_unique<flexmlrt::client::Model>(opts);
    } catch (const std::exception& e) {
      std::cerr << "[FlexMLRT ERROR] Model creation failed: " << e.what()
                << std::endl;
      throw std::runtime_error(std::string("Failed to load FlexMLRT vision "
                                           "model from RAI: ") +
                               e.what());
    }

    if (!model_->good()) {
      throw std::runtime_error(
          "FlexMLRT vision model creation failed - check RAI file, subgraph "
          "name, and device availability");
    }
    DEBUG_LOG("VisionFlexMLRTModel constructor END");
  }

  py::array_t<float> forward(py::array_t<float> preprocessed_input) {
    DEBUG_LOG("forward() START");

    auto buf = preprocessed_input.request();
    if (buf.ndim != 3) {
      throw std::runtime_error(
          "preprocessed_input must be 3D array [1073, 4, 1280]");
    }

    int64_t dim0 = buf.shape[0];
    int64_t dim1 = buf.shape[1];
    int64_t dim2 = buf.shape[2];

    if (dim0 != 1073 || dim1 != 4 || dim2 != 1280) {
      throw std::runtime_error(
          "Expected input shape [1073, 4, 1280], got [" + std::to_string(dim0) +
          ", " + std::to_string(dim1) + ", " + std::to_string(dim2) + "]");
    }

    std::vector<flexmlrt::client::ErtIoTypeNew> ifms;
    ifms.push_back(makeIO("/blocks/Gather_output_0", 0, buf.ptr,
                          dim0 * dim1 * dim2 * sizeof(float), "float32",
                          {dim0, dim1, dim2}));

    int64_t out_dim0 = 1073;
    int64_t out_dim1 = 3584;
    std::vector<float> output_buf(out_dim0 * out_dim1);
    std::vector<flexmlrt::client::ErtIoTypeNew> ofms;
    ofms.push_back(makeIO("/merger/merger/mlp/mlp.2/Gemm_output_0", 0,
                          output_buf.data(), output_buf.size() * sizeof(float),
                          "float32", {out_dim0, out_dim1}));
    std::vector<flexmlrt::client::ErtIoTypeNew> wts;

    DEBUG_LOG("Calling model->forward() (GIL released)...");
    try {
      py::gil_scoped_release release;
      model_->forward(ifms, ofms, wts);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("FlexMLRT forward failed: ") +
                               e.what());
    }

    py::array_t<float> result({out_dim0, out_dim1});
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, output_buf.data(),
                output_buf.size() * sizeof(float));

    DEBUG_LOG("forward() END");
    return result;
  }

  int output_dim() const { return 3584; }

 private:
  vaiml_run::RaiLoader rai_loader_;
  std::unique_ptr<flexmlrt::client::Model> model_;
  std::string device_name_;
};

PYBIND11_MODULE(_vision_flexmlrt_npu, m) {
  m.doc() = "FlexMLRT vision model for NPU inference (RAI cache)";

  py::class_<VisionFlexMLRTModel>(m, "VisionFlexMLRTModel")
      .def(py::init<std::string, std::string>(), py::arg("rai_path"),
           py::arg("device_name") = "stx",
           "Load FlexMLRT vision model from a compiled .rai cache file.\n\n"
           "Args:\n"
           "    rai_path: Path to the .rai file (VLLM_VISION_NPU_CACHE)\n"
           "    device_name: XRT device name (default: 'stx')")
      .def("forward", &VisionFlexMLRTModel::forward,
           py::arg("preprocessed_input"),
           "Run NPU vision encoding on CPU-preprocessed [1073, 4, 1280] input")
      .def("output_dim", &VisionFlexMLRTModel::output_dim,
           "Output embedding dimension");
}
