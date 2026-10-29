"""Parche de ENTORNO (no toca pesos): factores RoPE por dimension sobre MRotaryEmbedding.

YaRN deriva el factor de cada dimension de una formula cerrada. Esto permite darlos a
mano (estilo LongRoPE) CONSERVANDO mRoPE (y por tanto la torre visual), y recalcularlos
EN CALIENTE sin reiniciar el servidor.

Activacion: PYTHONPATH=/app + sitecustomize.py que llame a apply().
Control:    VLLM_PERDIM_ROPE_FILE=<json> con {"factors":[f0..f31]}. Un hilo vigila el
            fichero y reconstruye la tabla al vuelo. factor=1 -> identidad.
Nota:       cos_sin_cache se actualiza con copy_ in-place, asi el puntero no cambia y
            los CUDA graphs capturados siguen siendo validos.
"""
import json, os, threading, time
import torch

_INSTANCES = []
_STATE = {"factors": None, "mtime": 0.0, "aplicado": 0}


def _read(path):
    try:
        with open(path) as fh:
            f = json.load(fh).get("factors")
        return list(map(float, f)) if f else None
    except Exception:
        return None


def _set_all(factors):
    n = 0
    for r in list(_INSTANCES):
        try:
            r.set_perdim_factors(factors)
            n += 1
        except Exception:
            pass
    _STATE["aplicado"] = n
    return n


def _watch(path, status):
    while True:
        try:
            m = os.path.getmtime(path) if os.path.exists(path) else 0.0
            if m != _STATE["mtime"]:
                _STATE["mtime"] = m
                f = _read(path)
                n = _set_all(f)
                with open(status, "w") as fh:
                    json.dump({"pid": os.getpid(), "instancias": n,
                               "factors": f, "ts": time.time()}, fh)
        except Exception:
            pass
        time.sleep(1.0)


def apply():
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    if getattr(MRotaryEmbedding, "_perdim_patched", False):
        return
    orig_inv = MRotaryEmbedding._compute_inv_freq
    orig_init = MRotaryEmbedding.__init__

    def _compute_inv_freq(self, base):
        inv = orig_inv(self, base)
        f = _STATE["factors"]
        if f is not None:
            t = torch.tensor(f, dtype=inv.dtype, device=inv.device)
            if t.numel() != inv.numel():
                raise ValueError(f"factores {t.numel()} != pares {inv.numel()}")
            inv = inv / t
        return inv

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        _INSTANCES.append(self)

    def set_perdim_factors(self, factors):
        _STATE["factors"] = list(map(float, factors)) if factors else None
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache.copy_(cache.to(self.cos_sin_cache.device,
                                          self.cos_sin_cache.dtype))
        return int(self.cos_sin_cache.numel())

    MRotaryEmbedding._compute_inv_freq = _compute_inv_freq
    MRotaryEmbedding.__init__ = __init__
    MRotaryEmbedding.set_perdim_factors = set_perdim_factors
    MRotaryEmbedding._perdim_patched = True

    path = os.environ.get("VLLM_PERDIM_ROPE_FILE")
    if path:
        _STATE["factors"] = _read(path)
        _STATE["mtime"] = os.path.getmtime(path) if os.path.exists(path) else 0.0
        status = os.environ.get("VLLM_PERDIM_ROPE_STATUS", "/app/perdim_status.json")
        threading.Thread(target=_watch, args=(path, status), daemon=True).start()
