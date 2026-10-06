# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""⛔ RETIRADO el 21-sep-2026: el +6% de TTFT no paga el -22/27% de decode.

Lo de abajo es correcto y el camino denso funciona (TTFT -5,9% en el lab, y el arnes de
arranque hace su trabajo). Pero medido e2e a TP4 con concurrencia, este override junto
con `tiny_gemv` cuesta:

    6 sesiones (prompts de 4,3k):  44,3 ms sin ellos  ->  56,7 ms con ellos   (-22%)
    4 sesiones:                    34,4 ms            ->  47,1 ms             (-27%)

y deja UNA GPU al 100% y 120-123 W con el servidor EN REPOSO. Al quitar los dos, las
cuatro GPUs bajan a 0% y 6-13 W, y las dos cajas coinciden al 0,6%.

El trato completo, con los dos signos:
    +6% de TTFT en prefill en frio   (al quitarlo, el frio de 9k paso de 3,49-3,54 s a 3,71 s)
    -22/27% de decode en concurrencia
    120 W quemados en reposo

No esta desplegado en ninguna caja. Fleco honesto: el TTFT a 163k sin el override no se
llego a medir; la evidencia del prefill en frio de 9k apunta bien pero es INDIRECTA.

"""

"""Enchufa el camino de M grande de la GEMM W4A16 (`rdna3_rocblas_gemm.cu`).

Este fichero NO sustituye nada: solo compila la extension, que al cargarse
sobrescribe la implementacion CUDA del propio `_rocm_C::gptq_gemm_rdna3`.

⛔ Por que asi, con tres vehiculos medidos el 20-sep contra una base reproducida
tres veces a 21,2 ms de paso:

    cierre de Python con ramas por forma ..... decode 60,1 ms
    torch.library.custom_op de Python ........ decode 46,9 ms
    op propio de C++ (TORCH_LIBRARY + Meta) .. decode 34,8 ms

El tercero cierra la puerta: con el umbral a 100.000.000 el camino denso no corre
NI EN EL PREFILL (TTFT identico a la base) y el decode seguia en 34,8 ms. El coste
lo paga **sustituir el op**, no el kernel: el call site (`rdna3_w4a16.py:158`) esta
dentro de una region de torch.compile y ahi un op que inductor no conoce se compila
como FallbackKernel. Sobrescribiendo la impl, el grafo es bit a bit el de antes.

El veto de `csrc/rocm/q_gemm_rdna3.cu:699` pedia "un arnes que falle primero".
`_selftest()` es ese arnes: en el arranque compara el camino denso contra el kernel
fusionado ORIGINAL (alcanzado por su simbolo, saltandose el override) con las dos
convenciones del cero, y si no cuadra **no se parchea** y lo dice a gritos.
"""

import os
import sys

import torch

_MIN_M = int(os.environ.get("VLLM_RDNA3_ROCBLAS_MIN_M", "0"))
_SRC = os.environ.get(
    "VLLM_RDNA3_ROCBLAS_SRC",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdna3_rocblas_gemm.cu"),
)
_BUILD = os.environ.get("VLLM_RDNA3_ROCBLAS_BUILD", "/tmp/rdna3_rocblas_build")
_TOL = float(os.environ.get("VLLM_RDNA3_ROCBLAS_TOL", "2e-2"))


def _cargar():
    from torch.utils.cpp_extension import load

    os.makedirs(_BUILD, exist_ok=True)
    return load(
        name="rdna3_rocblas_ext",
        sources=[_SRC],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-ldl"],
        build_directory=_BUILD,
        verbose=False,
    )


def _selftest(ext):
    """El arnes que el veto de `q_gemm_rdna3.cu:699` pedia. Barato y en el arranque.

    La referencia es el PROPIO kernel fusionado, que es lo unico que sabe de verdad
    como estan empaquetados los pesos; se alcanza por `ext.original`, que va por el
    simbolo y por tanto se salta el override. Con la convencion del cero cruzada el
    error seria 7,0e-02 y pasaria cualquier mirada por encima: de ahi el umbral.
    """
    dev = torch.cuda.current_device()
    k, n, group = 256, 128, 32
    groups = k // group
    g = torch.Generator(device="cuda").manual_seed(1234)
    wq = torch.randint(0, 2**31 - 1, (k // 8, n), device=dev, dtype=torch.int32,
                       generator=g)
    qz = torch.randint(0, 2**31 - 1, (groups, n // 8), device=dev, dtype=torch.int32,
                       generator=g)
    sc = (torch.rand((groups, n), device=dev, dtype=torch.float16, generator=g) * 0.02
          + 0.001)
    a = torch.randn((64, k), device=dev, dtype=torch.float16, generator=g)

    previo = ext.get_min_m()
    ext.set_min_m(64)
    try:
        for v2 in (False, True):
            denso = torch.ops._rocm_C.gptq_gemm_rdna3(a, wq, qz, sc, v2)
            ref = ext.original(a, wq, qz, sc, v2)
            rel = ((denso.float() - ref.float()).abs().max().item()
                   / (ref.float().abs().max().item() + 1e-6))
            print(f"[rocblas-prefill] selftest v2={v2}: rel={rel:.3e}", file=sys.stderr)
            if rel > _TOL:
                raise RuntimeError(
                    f"el dequant NO casa con el kernel fusionado "
                    f"(v2={v2}, rel={rel:.3e} > {_TOL:.1e})"
                )
    finally:
        ext.set_min_m(previo)


def _desactivar(ext):
    """Deja el override puesto pero inerte: todo se reenvia al kernel fusionado."""
    ext.set_min_m(0)


def apply() -> None:
    if _MIN_M <= 0:
        return
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "gptq_gemm_rdna3"
    ):
        print("[rocblas-prefill] no hay _rocm_C::gptq_gemm_rdna3", file=sys.stderr)
        return

    # La aridad se comprueba ANTES de cargar, porque cargar ya registra el override:
    # `gemm_impl` esta escrito para 5 argumentos y con otro esquema `m.impl` fallaria
    # al registrar, dentro de la carga del modelo.
    esquema = torch.ops._rocm_C.gptq_gemm_rdna3.default._schema
    nargs = len(esquema.arguments)
    if nargs != 5:
        print(f"[rocblas-prefill] aridad {nargs} != 5, no se parchea: {esquema}",
              file=sys.stderr)
        return

    ext = _cargar()
    if not ext.original_resuelto():
        print("[rocblas-prefill] !!! NO se resolvio el simbolo del kernel fusionado; "
              "se deja inerte", file=sys.stderr)
        _desactivar(ext)
        return

    if os.environ.get("VLLM_RDNA3_ROCBLAS_SELFTEST", "1") == "1":
        # Falla CERRADO: si el arnes no cuadra, el override se deja inerte y el
        # servidor sigue con el kernel fusionado. El modo de fallo que vetaba este
        # camino era SILENCIOSO (el modelo escribia `!!!!` horas despues).
        try:
            _selftest(ext)
        except Exception as e:  # noqa: BLE001
            print(f"[rocblas-prefill] !!! NO SE PARCHEA: {e}", file=sys.stderr)
            _desactivar(ext)
            return

    print(f"[rocblas-prefill] impl CUDA de _rocm_C::gptq_gemm_rdna3 sobrescrita, "
          f"camino denso desde M={_MIN_M}", file=sys.stderr)


_TARGET = "vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16"


def apply_when_imported() -> None:
    if _TARGET in sys.modules:
        apply()
        return

    import importlib.abc
    import importlib.machinery

    class _PatchAfterLoad(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(module):
                inner(module)
                # Falla CERRADO: este gancho corre DENTRO de la carga del modelo, asi
                # que una excepcion aqui tumba el worker. El servidor tiene que poder
                # seguir con el kernel fusionado.
                try:
                    apply()
                except Exception as e:  # noqa: BLE001
                    print(f"[rocblas-prefill] !!! no aplicado: {e}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterLoad())


if _MIN_M > 0:
    apply_when_imported()
