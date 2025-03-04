#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_compat.h"
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

inline void cuErrCheck_(CUresult stat, char const* file, int line) {
  if (stat != CUDA_SUCCESS) {
    char const* msg = nullptr;
    cuGetErrorName(stat, &msg);
    fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
  }
}
#define cuErrCheck(stat) \
  { cuErrCheck_((stat), __FILE__, __LINE__); }

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)
inline int getDevice() {
  int current_dev_id = 0;
  CUDACHECK(cudaGetDevice(&current_dev_id));
  return current_dev_id;
}
inline int getSMVersion() {
  int device{-1};
  CUDACHECK(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDACHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor,
                                   device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor,
                                   device));
  return sm_major * 10 + sm_minor;
}

// For xqa kernel IO
enum Data_type {
  DATA_TYPE_FP16,
  DATA_TYPE_BF16,
  DATA_TYPE_FP32,
  DATA_TYPE_E4M3,
  DATA_TYPE_UNKNOWN
};

class Log {
 public:
  // The different predefined verbosity levels.
  static constexpr int Muted = 0;
  static constexpr int Error = 10;
  static constexpr int Warning = 20;
  static constexpr int Info = 30;
  static constexpr int SchedInfo = 40;
  static constexpr int Trace = 50;

 public:
  // Ctor.
  Log();

  // Get the singleton.
  static Log* getInstance();

  // Are the warnings treated as errors?
  //  inline bool areWarningsTreatedAsErrors() const {
  //    return mAreWarningsTreatedAsErrors;
  //  }
  // The level of verbosity.
  inline int getVerbLvl() const { return mVerbLvl; }
  // Is it muted?
  //  inline bool isMuted() const { return mVerbLvl == Muted; }

  // Log error.
  template <typename... Args>
  inline std::string log(int verbLvl, char desc, char const* filePath, int line,
                         Args... args) {
    std::string fileName{std::filesystem::path{filePath}.filename().native()};
    std::string msg{toString(args...)};
    getStream(verbLvl) << "[" << desc << "] [" << fileName << ":" << line
                       << "]: ";
    getStream(verbLvl) << msg << std::endl;
    return msg;
  }

  // Create a string from a single string... Do nothing...
  static inline std::string toString(std::string msg) { return msg; }

  // Create a string for a list of arguments.
  template <typename... Args>
  static inline std::string toString(Args... args) {
    std::ostringstream stream;
    auto pushToStream = [&](auto a) { stream << a; };
    (pushToStream(args), ...);
    return stream.str();
  }

 private:
  // Helper function.

  inline std::ostream& getStream(int verbLvl) {
    if (mStream) {
      return *mStream;
    } else {
      return verbLvl >= Info ? std::cout : std::cerr;
    }
  }

 private:
  // Are the warnings treated as errors?
  //  bool mAreWarningsTreatedAsErrors;
  // The level of verbosity.
  int mVerbLvl;
  // The log stream.
  std::unique_ptr<std::ofstream> mStream;
};

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
