// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Camino de M grande para la GEMM W4A16 de gfx1100: dequantizar a denso y dejar
// la multiplicacion a rocBLAS/hipBLASLt.
//
// POR QUE SE SOBRESCRIBE EL OP EN VEZ DE SUSTITUIRLO. Tres vehiculos medidos en
// ainode2 el 20-sep, todos contra una base reproducida tres veces a 21,2 ms de paso:
//
//     cierre de Python con ramas por forma ..... decode 60,1 ms
//     torch.library.custom_op de Python ........ decode 46,9 ms
//     op propio de C++ (TORCH_LIBRARY + Meta) .. decode 34,8 ms
//
// El tercero cierra la puerta: con el umbral puesto a 100.000.000 el camino denso
// NO se ejecuta ni en el prefill -- lo confirma el TTFT, identico a la base -- y el
// decode seguia en 34,8 ms. O sea que **el coste no lo paga el kernel nuevo, lo
// paga sustituir el op**. El call site (`rdna3_w4a16.py:158`) vive dentro de una
// region de torch.compile, y ahi un op que inductor no conoce se compila como
// FallbackKernel: el CUDA graph sigue intacto y las 224 GEMM por paso son las
// mismas, pero aparecen 8,17 ms de HUECOS (24% del paso, contra 0,95 = 5%) y copias
// nuevas (aten::copy_ 92/paso, Memcpy DtoD 41, __amd_rocclr_copyBuffer 145).
//
// Ojo: el microbanco decia que el salto del dispatcher cuesta -0,9 a +0,8 us a
// M=4..24, y era verdad -- medido en EAGER. No predice nada de lo de arriba.
//
// Asi que aqui no se sustituye nada: se sobrescribe la implementacion CUDA del
// PROPIO `_rocm_C::gptq_gemm_rdna3`, y el kernel original se alcanza por su simbolo
// para poder reenviar por debajo del umbral. Desde el grafo no cambia NADA: el nodo
// sigue siendo el mismo op con el mismo esquema y el mismo fake de Python.
//
// EL DEQUANT ESTA VALIDADO, no reconstruido de memoria. La identidad
// `gptq_gemm_rdna3(I_KxK, W, ...)` devuelve los pesos densos tal y como los ve el
// kernel fusionado; contra eso el error es 0,000e+00 con la convencion v2 y
// 5,0e-04 (redondeo de fp16) con la v1. Con la convencion cruzada es 7,0e-02, o
// sea que adivinarla da un resultado plausible y equivocado.
//
// Medido el 20-sep-2026 en ainode2 (gfx1100, formas de prefill por GPU a TP4):
//
//     forma           M=2048   WMMA actual -> dequant+rocBLAS
//     mlp.gate_up              2378,8 us -> 2104,3   1,13x
//     mlp.down                 1260,3    ->  927,0   1,36x
//     gdn.qkvz                 1170,5    ->  918,2   1,27x

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <dlfcn.h>

#include <map>
#include <mutex>
#include <utility>

namespace {

// [K/8, N] uint32 -> [K, N] half.
//
// Los 8 valores de k de cada uint32 van ENTRELAZADOS (ver dequant_4bit_8_fp16 en
// qdq_4_rdna3.cuh): q[2i] en los bits 4i..4i+3 y q[2i+1] en 16+4i..16+4i+3.
// Leerlo como "nibble i = k i" da un resultado plausible y equivocado.
//
// Un hilo se come los 8 valores de k de un uint32 para una sola n, asi que
// `b_q_weight` se lee UNA vez en total; los hilos consecutivos llevan n
// consecutivas, o sea que tanto la lectura del empaquetado como los 8 stores
// salen coalescidos.
__global__ void dequant_w4a16_kernel(const uint32_t* __restrict__ b_q_weight,
                                     const uint32_t* __restrict__ b_qzeros,
                                     const half* __restrict__ b_scales,
                                     half* __restrict__ out, int size_k,
                                     int size_n, int group_size,
                                     int zero_offset) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int kb = blockIdx.y;  // bloque de 8 valores de k
  if (n >= size_n) return;

  const uint32_t packed = b_q_weight[(size_t)kb * size_n + n];
  const int zshift = (n % 8) * 4;
  const int nz = size_n / 8;

  int last_g = -1;
  float scale = 0.0f;
  float zero = 0.0f;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = kb * 8 + i;
    if (k >= size_k) break;
    const int g = k / group_size;
    // Con G32 los 8 k de un uint32 caen siempre en el mismo grupo, asi que esto
    // carga una sola vez. Se deja general para que un group_size que no sea
    // multiplo de 8 siga siendo correcto en vez de silenciosamente mal.
    if (g != last_g) {
      last_g = g;
      scale = __half2float(b_scales[(size_t)g * size_n + n]);
      const uint32_t zp = b_qzeros[(size_t)g * nz + (n / 8)];
      zero = (float)((zp >> zshift) & 0xF);
    }
    const int shift = (i / 2) * 4 + (i % 2) * 16;
    const float q = (float)((packed >> shift) & 0xF);
    out[(size_t)k * size_n + n] = __float2half((q - zero - (float)zero_offset) * scale);
  }
}

// Espacio de trabajo persistente para el denso.
//
// Sin esto se pedian y liberaban ~89 MB (gate_up: K=5120, N=8704) en CADA llamada,
// cientos de veces por chunk de prefill. El asignador cacheado lo absorbe casi
// siempre, pero cuando tiene que crecer llama a hipMalloc con el resto del modelo
// ya dentro, y ese coste cae en medio del paso.
//
// Se puede reutilizar porque todas estas GEMM salen en el MISMO flujo: el dequant
// de la llamada N+1 esta ordenado despues del matmul de la N, que ya lo leyo. La
// clave lleva el flujo justamente para no apoyarse en esa suposicion si algun dia
// hay mas de uno.
at::Tensor& espacio_denso(int64_t elementos, const torch::TensorOptions& opts,
                          hipStream_t stream) {
  static std::mutex mu;
  static std::map<std::pair<int, void*>, at::Tensor> cache;
  const int dev = (int)c10::cuda::current_device();
  std::lock_guard<std::mutex> lk(mu);
  auto& t = cache[{dev, (void*)stream}];
  if (!t.defined() || t.numel() < elementos) {
    t = torch::empty({elementos}, opts);
  }
  return t;
}

// El umbral es una variable, no una constante, para que el autotest pueda bajarlo
// un instante y validar el camino DENSO aunque en produccion este puesto a un
// valor que no se alcanza nunca. Antes el autotest lo deducia del umbral y con un
// umbral de 1e8 pedia una matriz de 47,69 GiB.
int g_min_m = [] {
  const char* e = std::getenv("VLLM_RDNA3_ROCBLAS_MIN_M");
  return e ? std::atoi(e) : 0;
}();

int min_m_umbral() { return g_min_m; }

// El kernel fusionado original, por su simbolo. Sobrescribir la entrada del
// dispatcher deja el kernel de antes inalcanzable por ahi, asi que se coge directo:
// `_rocm_C.abi3.so` lo exporta como simbolo global de texto
// (`_Z15gptq_gemm_rdna3N2at6TensorES0_S0_S0_b`). Python abre esa .so con RTLD_NOW y
// SIN RTLD_GLOBAL, asi que dlsym(RTLD_DEFAULT) no lo ve: hay que pedir un handle a
// lo ya cargado con RTLD_NOLOAD.
using FnOriginal = torch::Tensor (*)(torch::Tensor, torch::Tensor, torch::Tensor,
                                     torch::Tensor, bool);

FnOriginal original_o_nulo() {
  static FnOriginal fn = []() -> FnOriginal {
    const char* ruta = std::getenv("VLLM_RDNA3_ROCM_SO");
    const char* simbolo = std::getenv("VLLM_RDNA3_GPTQ_SYMBOL");
    if (!ruta) ruta = "/usr/local/lib/python3.12/dist-packages/vllm/_rocm_C.abi3.so";
    if (!simbolo) simbolo = "_Z15gptq_gemm_rdna3N2at6TensorES0_S0_S0_b";
    void* h = dlopen(ruta, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return nullptr;
    return (FnOriginal)dlsym(h, simbolo);
  }();
  return fn;
}

torch::Tensor llamar_original(const torch::Tensor& a, const torch::Tensor& b_q_weight,
                              const torch::Tensor& b_qzeros,
                              const torch::Tensor& b_scales, bool use_v2_format) {
  FnOriginal fn = original_o_nulo();
  TORCH_CHECK(fn != nullptr,
              "rocblas-prefill: no se pudo resolver el kernel fusionado original; "
              "sin el no hay a donde reenviar por debajo del umbral");
  return fn(a, b_q_weight, b_qzeros, b_scales, use_v2_format);
}

torch::Tensor gemm_impl(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_qzeros, torch::Tensor b_scales,
                        bool use_v2_format) {
  const int min_m = min_m_umbral();
  const bool apto = min_m > 0 && a.dim() == 2 && a.scalar_type() == torch::kHalf &&
                    a.size(0) >= min_m && b_q_weight.dim() == 2;
  if (!apto) {
    return llamar_original(a, b_q_weight, b_qzeros, b_scales, use_v2_format);
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_k = (int)a.size(1);
  const int size_n = (int)b_q_weight.size(1);
  const int groups = (int)b_scales.size(0);
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf,
              "b_scales debe ser half cuando a es half");
  TORCH_CHECK(b_q_weight.is_contiguous() && b_qzeros.is_contiguous() &&
                  b_scales.is_contiguous(),
              "los pesos empaquetados deben ser contiguos");
  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k, "b_q_weight debe ser [K/8, N]");
  TORCH_CHECK(size_n % 8 == 0, "N debe ser multiplo de 8");
  TORCH_CHECK(groups > 0 && size_k % groups == 0, "grupos incompatibles con K");
  const int group_size = size_k / groups;

  auto opts = torch::TensorOptions().dtype(torch::kHalf).device(a.device());
  at::Tensor dense =
      espacio_denso((int64_t)size_k * size_n, opts, stream)
          .narrow(0, 0, (int64_t)size_k * size_n)
          .view({size_k, size_n});

  const int hilos = 64;
  dim3 grid((size_n + hilos - 1) / hilos, (size_k + 7) / 8);
  hipLaunchKernelGGL(dequant_w4a16_kernel, grid, dim3(hilos), 0, stream,
                     (const uint32_t*)b_q_weight.data_ptr(),
                     (const uint32_t*)b_qzeros.data_ptr(),
                     (const half*)b_scales.data_ptr(), (half*)dense.data_ptr(),
                     size_k, size_n, group_size, use_v2_format ? 0 : 1);

  return at::matmul(a, dense);
}

}  // namespace

// El override. Es lo unico que hace esta extension al cargarse: la entrada CUDA del
// op de siempre pasa a ser `gemm_impl`. El esquema, el nombre y el fake de Python no
// se tocan, asi que inductor compila exactamente el mismo grafo que antes.
//
// ⚠️ Solo aridad 5 (`a, b_q_weight, b_qzeros, b_scales, use_v2_format`), que es la
// del fork y la de la imagen desplegada (comprobado leyendo el esquema registrado).
// Si algun dia hay otra, `m.impl` falla al registrar y el parche no se aplica: el
// patch_rocblas_prefill.py comprueba la aridad ANTES de cargar esto.
TORCH_LIBRARY_IMPL(_rocm_C, CUDA, m) { m.impl("gptq_gemm_rdna3", &gemm_impl); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_min_m", [] { return g_min_m; }, "umbral de filas actual");
  m.def("set_min_m", [](int v) { g_min_m = v; }, "cambia el umbral de filas");
  m.def("original_resuelto", [] { return original_o_nulo() != nullptr; },
        "si el simbolo del kernel fusionado se pudo resolver");
  m.def("original", [](torch::Tensor a, torch::Tensor w, torch::Tensor z,
                       torch::Tensor s, bool v2) {
          return llamar_original(a, w, z, s, v2);
        },
        "el kernel fusionado original, saltandose el override (para el arnes)");
}
