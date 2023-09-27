/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_utils.h"
// #include "cuda_fp8_utils.h"

/* **************************** debug tools ********************************* */

template <typename T>
void print_to_file(const T *result, const int size, const char *file,
                   cudaStream_t stream, std::ios::openmode open_mode) {
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  printf("[INFO] file: %s with size %d.\n", file, size);
  std::ofstream outFile(file, open_mode);
  if (outFile) {
    T *tmp = new T[size];
    check_cuda_error(cudaMemcpyAsync(tmp, result, sizeof(T) * size,
                                     cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < size; ++i) {
      float val = (float)(tmp[i]);
      outFile << val << std::endl;
    }
    delete[] tmp;
  } else {
    throw std::runtime_error(std::string("[FT][ERROR] Cannot open file: ") +
                             file + "\n");
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template void print_to_file(const float *result, const int size,
                            const char *file, cudaStream_t stream,
                            std::ios::openmode open_mode);
template void print_to_file(const half *result, const int size,
                            const char *file, cudaStream_t stream,
                            std::ios::openmode open_mode);

template <typename T>
void print_abs_mean(const T *buf, uint size, cudaStream_t stream,
                    std::string name) {
  if (buf == nullptr) {
    // FT_LOG_WARNING("It is an nullptr, skip!");
    return;
  }
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  T *h_tmp = new T[size];
  cudaMemcpyAsync(h_tmp, buf, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  double sum = 0.0f;
  uint64_t zero_count = 0;
  float max_val = -1e10;
  bool find_inf = false;
  for (uint i = 0; i < size; i++) {
    if (std::isinf((float)(h_tmp[i]))) {
      find_inf = true;
      continue;
    }
    sum += abs((double)h_tmp[i]);
    if ((float)h_tmp[i] == 0.0f) {
      zero_count++;
    }
    max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
  }
  printf("[INFO][FT] %20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, "
         "find inf: %s",
         name.c_str(), size, sum / size, sum, max_val,
         find_inf ? "true" : "false");
  std::cout << std::endl;
  delete[] h_tmp;
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template void print_abs_mean(const float *buf, uint size, cudaStream_t stream,
                             std::string name);
template void print_abs_mean(const half *buf, uint size, cudaStream_t stream,
                             std::string name);
template void print_abs_mean(const int *buf, uint size, cudaStream_t stream,
                             std::string name);
template void print_abs_mean(const uint *buf, uint size, cudaStream_t stream,
                             std::string name);
template void print_abs_mean(const int8_t *buf, uint size, cudaStream_t stream,
                             std::string name);

template <typename T> void print_to_screen(const T *result, const int size) {
  if (result == nullptr) {
    // FT_LOG_WARNING("It is an nullptr, skip! \n");
    return;
  }
  T *tmp = reinterpret_cast<T *>(malloc(sizeof(T) * size));
  check_cuda_error(
      cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) {
    printf("%d, %f\n", i, static_cast<float>(tmp[i]));
  }
  free(tmp);
}

template void print_to_screen(const float *result, const int size);
template void print_to_screen(const half *result, const int size);
template void print_to_screen(const int *result, const int size);
template void print_to_screen(const uint *result, const int size);
template void print_to_screen(const bool *result, const int size);


template <typename T>
void printMatrix(T *ptr, int m, int k, int stride, bool is_device_ptr) {
  T *tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    tmp = reinterpret_cast<T *>(malloc(m * stride * sizeof(T)));
    check_cuda_error(
        cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%7.3f ", (float)tmp[ii * stride + jj]);
      } else {
        printf("%7d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

template void printMatrix(float *ptr, int m, int k, int stride,
                          bool is_device_ptr);
template void printMatrix(half *ptr, int m, int k, int stride,
                          bool is_device_ptr);

void printMatrix(unsigned long long *ptr, int m, int k, int stride,
                 bool is_device_ptr) {
  typedef unsigned long long T;
  T *tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    tmp = reinterpret_cast<T *>(malloc(m * stride * sizeof(T)));
    check_cuda_error(
        cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4llu ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

void printMatrix(int *ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef int T;
  T *tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    tmp = reinterpret_cast<T *>(malloc(m * stride * sizeof(T)));
    check_cuda_error(
        cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4d ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

void printMatrix(size_t *ptr, int m, int k, int stride, bool is_device_ptr) {
  typedef size_t T;
  T *tmp;
  if (is_device_ptr) {
    // k < stride ; stride = col-dimension.
    tmp = reinterpret_cast<T *>(malloc(m * stride * sizeof(T)));
    check_cuda_error(
        cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
  } else {
    tmp = ptr;
  }

  for (int ii = -1; ii < m; ++ii) {
    if (ii >= 0) {
      printf("%02d ", ii);
    } else {
      printf("   ");
    }

    for (int jj = 0; jj < k; jj += 1) {
      if (ii >= 0) {
        printf("%4ld ", tmp[ii * stride + jj]);
      } else {
        printf("%4d ", jj);
      }
    }
    printf("\n");
  }
  if (is_device_ptr) {
    free(tmp);
  }
}

template <typename T> void check_max_val(const T *result, const int size) {
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float max_val = -100000;
  for (int i = 0; i < size; i++) {
    float val = static_cast<float>(tmp[i]);
    if (val > max_val) {
      max_val = val;
    }
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

template void check_max_val(const float *result, const int size);
template void check_max_val(const half *result, const int size);

template <typename T> void check_abs_mean_val(const T *result, const int size) {
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += abs(static_cast<float>(tmp[i]));
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

template void check_abs_mean_val(const float *result, const int size);
template void check_abs_mean_val(const half *result, const int size);

/* ***************************** common utils ****************************** */

cudaError_t getSetDevice(int i_device, int *o_device) {
  int current_dev_id = 0;
  cudaError_t err = cudaSuccess;

  if (o_device != NULL) {
    err = cudaGetDevice(&current_dev_id);
    if (err != cudaSuccess) {
      return err;
    }
    if (current_dev_id == i_device) {
      *o_device = i_device;
    } else {
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess) {
        return err;
      }
      *o_device = current_dev_id;
    }
  } else {
    err = cudaSetDevice(i_device);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

/* ************************** end of common utils ************************** */
