# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatically apply hardware_profiles.yaml to EngineArgs for Cohere models.

Invoked from `vllm/engine/arg_utils.py::EngineArgs.__post_init__`
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import typing
from pathlib import Path

import yaml

from vllm.logger import init_logger

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

_COHERE_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "CohereForCausalLM",
        "Cohere2ForCausalLM",
        "Cohere2MoeForCausalLM",
        "Cohere2VisionForConditionalGeneration",
    }
)

# The hardware profiles YAML must lives next to this module in source.
_DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "hardware_profiles.yaml"


@functools.lru_cache(maxsize=128)
def detect_cohere_from_model_id(
    model: str | None,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> bool:
    """Probe the HF config for the model and check architectures."""
    if not model:
        return False
    try:
        from vllm.transformers_utils.config import get_config

        cfg = get_config(model, trust_remote_code, revision, None, "auto")
        archs = list(getattr(cfg, "architectures", None) or [])
        result = any(a in _COHERE_ARCHITECTURES for a in archs)
        logger.debug(
            "Cohere arch probe: model=%s architectures=%s -> is_cohere=%s",
            model,
            archs,
            result,
        )
        return result
    except Exception as e:
        logger.warning("Cohere arch probe failed for model=%s: %s", model, e)
        return False


@functools.cache
def _gpu_name() -> str:
    """Best-effort device-name lookup. Empty string on any failure."""
    try:
        from vllm.platforms import current_platform

        name = current_platform.get_device_name(0) or ""
        logger.debug("COHERE AUTO-CONFIG: detected device name %r", name)
        return name
    except Exception as e:
        logger.warning("get_device_name failed: %s", e)
        return ""


def _evaluate_when(when_clause: str, gpu_name: str) -> bool:
    """Evaluate a CEL `when:` clause against {server.type, gpu.name}.

    `gpu.name` is bound lowercased so the YAML's lowercase patterns
    (``"b200"``, ``"mi300x"``) match real device names. Empty clauses
    default to True; bad clauses log a warning and return False.
    """
    if not when_clause.strip():
        return True
    try:
        import celpy

        env = celpy.Environment()
        prog = env.program(env.compile(when_clause))
        result = prog.evaluate(
            {
                "server": celpy.json_to_cel({"type": "vllm"}),
                "gpu": celpy.json_to_cel({"name": gpu_name.lower()}),
            }
        )
        return bool(result)
    except Exception as e:
        logger.warning(
            "COHERE AUTO-CONFIG: when-clause %r failed: %s; skipping profile",
            when_clause,
            e,
        )
        return False


@functools.lru_cache(maxsize=8)
def _load_profiles_doc(path: Path) -> list[dict[str, object]]:
    """Return the `profiles` list from the YAML, filtering out non-dict entries."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    profiles = doc.get("profiles") if isinstance(doc, dict) else None
    if not isinstance(profiles, list):
        return []
    return [p for p in profiles if isinstance(p, dict)]


def resolve_profiles(
    *,
    gpu_name: str | None = None,
    profiles_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, str], list[str]]:
    """Return (merged_args, merged_env, applied_profile_names).

    Profiles are applied in YAML order; later wins on conflicting keys.
    Arg values keep their YAML-parsed type (str/int/float/bool/list/...);
    ``_coerce`` handles the conversion to the declared field type later.
    Env values are stringified because ``os.environ`` only accepts strings.
    """
    path = profiles_path or _DEFAULT_PROFILES_PATH
    try:
        profiles = _load_profiles_doc(path)
    except FileNotFoundError:
        logger.warning(
            "COHERE AUTO-CONFIG: hardware_profiles.yaml not found at %s; "
            "no auto-config will be applied",
            path,
        )
        return {}, {}, []

    name = gpu_name if gpu_name is not None else _gpu_name()

    args: dict[str, object] = {}
    env: dict[str, str] = {}
    applied: list[str] = []

    for p in profiles:
        prof_name = str(p.get("name") or "<unnamed>")
        when = str(p.get("when") or "")
        if not _evaluate_when(when, name):
            logger.debug(
                "COHERE AUTO-CONFIG: profile %s skipped (when=%r, gpu=%r)",
                prof_name,
                when,
                name,
            )
            continue
        prof_args = p.get("args") or {}
        prof_env = p.get("env") or {}
        if not isinstance(prof_args, dict) or not isinstance(prof_env, dict):
            logger.warning(
                "COHERE AUTO-CONFIG: profile %s has non-mapping args/env; skipping",
                prof_name,
            )
            continue
        applied.append(prof_name)
        args.update(prof_args)
        env.update({str(k): str(v) for k, v in prof_env.items()})
        logger.debug(
            "COHERE AUTO-CONFIG: profile %s matched; +%d args +%d env",
            prof_name,
            len(prof_args),
            len(prof_env),
        )

    return args, env, applied


_TRUTHY_STRINGS = frozenset({"", "1", "true", "yes", "on"})
_FALSY_STRINGS = frozenset({"0", "false", "no", "off"})


@functools.cache
def _engine_arg_kwargs() -> dict[str, dict[str, typing.Any]]:
    """Cache the EngineArgs argparse kwargs across `_coerce` calls.

    `get_kwargs` deep-copies its lru_cached output on every call so external
    callers can mutate freely.
    """
    from vllm.engine.arg_utils import EngineArgs, get_kwargs

    return get_kwargs(EngineArgs)


def _coerce(value: object, field_name: str) -> object:
    """Coerce a YAML profile value to the EngineArgs field's expected type.

    Delegates to vLLM's own argparse type-coercion machinery so the YAML
    profile inherits the same Literal / Union / Optional / JSON-dict /
    dataclass / human-readable-int handling that ``vllm serve`` supports
    for the same field. Raises ``TypeError`` on unknown fields or values
    the vLLM type fn rejects; ``_apply_args`` catches and logs a WARNING.
    """
    cfg = _engine_arg_kwargs().get(field_name)
    if cfg is None:
        raise TypeError(f"unknown EngineArgs field: {field_name!r}")

    type_fn = cfg.get("type")
    if type_fn is None:
        # BooleanOptionalAction: no string->value callable, parse here.
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUTHY_STRINGS:
                return True
            if v in _FALSY_STRINGS:
                return False
        raise TypeError(f"cannot coerce {value!r} to bool for field {field_name!r}")

    raw = value if isinstance(value, str) else str(value)
    try:
        return type_fn(raw)
    except (argparse.ArgumentTypeError, ValueError, TypeError) as e:
        raise TypeError(
            f"failed to coerce {value!r} for field {field_name!r}: {e}"
        ) from e


def _apply_env(profile_env: dict[str, str]) -> None:
    for key, val in profile_env.items():
        if key in os.environ:
            logger.info(
                "COHERE AUTO-CONFIG: env %s already set to %r; "
                "keeping user value (would have set %r)",
                key,
                os.environ[key],
                val,
            )
            continue
        os.environ[key] = val
        logger.info(
            "COHERE AUTO-CONFIG: env %s=%r (from hardware_profiles.yaml)", key, val
        )


def _field_default(field: dataclasses.Field) -> object:
    """Return the dataclass-declared default for ``field``.

    Raises ``LookupError`` if the field has neither a ``default`` nor a
    ``default_factory``; the caller treats such a field as user-set since
    we cannot tell user-supplied from default in that case.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    raise LookupError(field.name)


def _apply_args(engine_args: EngineArgs, profile_args: dict[str, object]) -> None:
    """Set EngineArgs fields from profile_args, respecting user overrides.

    "User-set" detection compares the live attribute value to the dataclass
    field's declared default (via ``f.default`` / ``f.default_factory``).
    We deliberately do NOT construct a fresh ``cls(model=...)`` sentinel
    here — that would re-enter ``__post_init__`` and recurse back into this
    function.
    """
    field_map = {f.name: f for f in dataclasses.fields(engine_args)}
    applied: list[str] = []
    skipped_user_set: list[str] = []
    skipped_unknown: list[str] = []
    skipped_coerce: list[str] = []

    for cli_key, raw_val in profile_args.items():
        attr = cli_key.replace("-", "_")
        f = field_map.get(attr)
        if f is None:
            skipped_unknown.append(cli_key)
            continue

        try:
            baseline = _field_default(f)
        except LookupError:
            skipped_user_set.append(cli_key)
            logger.info(
                "COHERE AUTO-CONFIG: --%s has no declared default; "
                "treating as user-set and skipping (profile would have set %r)",
                cli_key,
                raw_val,
            )
            continue
        except Exception as e:
            skipped_coerce.append(cli_key)
            logger.warning(
                "COHERE AUTO-CONFIG: default_factory for --%s raised %s; "
                "leaving field unchanged",
                cli_key,
                e,
            )
            continue

        current = getattr(engine_args, attr)
        if current != baseline:
            skipped_user_set.append(cli_key)
            logger.info(
                "COHERE AUTO-CONFIG: --%s already set to %r by user; "
                "keeping user value (profile would have set %r)",
                cli_key,
                current,
                raw_val,
            )
            continue

        try:
            new_val = _coerce(raw_val, attr)
        except Exception as e:
            skipped_coerce.append(cli_key)
            logger.warning(
                "COHERE AUTO-CONFIG: failed to coerce --%s=%r for field "
                "type %r: %s; leaving default",
                cli_key,
                raw_val,
                f.type,
                e,
            )
            continue

        setattr(engine_args, attr, new_val)
        applied.append(cli_key)
        logger.info(
            "COHERE AUTO-CONFIG: --%s: %r -> %r (from profile)",
            cli_key,
            baseline,
            new_val,
        )

    logger.info(
        "COHERE AUTO-CONFIG summary: applied=%d kept_user=%d "
        "unknown_field=%d coerce_failed=%d  applied_keys=%s",
        len(applied),
        len(skipped_user_set),
        len(skipped_unknown),
        len(skipped_coerce),
        applied,
    )
    if skipped_unknown:
        logger.warning(
            "COHERE AUTO-CONFIG: profile keys not recognized as EngineArgs "
            "fields (likely upstream rename): %s",
            skipped_unknown,
        )


def apply_cohere_auto_config(engine_args: EngineArgs) -> None:
    """Apply hardware_profiles.yaml to ``engine_args`` if it targets a Cohere model.

    Opt-in semantics: this function assumes the caller has already verified
    that ``VLLM_ENABLE_COHERE_AUTO_CONFIG`` is set. The check lives at the
    call site in ``vllm/engine/arg_utils.py`` so non-opted-in launches do
    not import this module at all.

    No-op (and never raises) for any of:

      - model is not a Cohere architecture
      - YAML missing or malformed
      - any internal error during resolution / coercion
    """
    try:
        model = engine_args.model
        if not detect_cohere_from_model_id(
            model,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        ):
            logger.debug(
                "COHERE AUTO-CONFIG: model=%r is not a Cohere architecture; skipping",
                model,
            )
            return

        gpu_name = _gpu_name()
        profile_args, profile_env, applied_names = resolve_profiles(gpu_name=gpu_name)

        if not applied_names:
            logger.info(
                "COHERE AUTO-CONFIG: detected Cohere model %r but no profile "
                "matched (gpu=%r); no changes applied",
                model,
                gpu_name,
            )
            return

        logger.info(
            "COHERE AUTO-CONFIG: detected Cohere model %r, gpu=%r, profiles applied=%s",
            model,
            gpu_name,
            applied_names,
        )

        _apply_env(profile_env)
        _apply_args(engine_args, profile_args)

    except Exception as e:
        logger.exception(
            "COHERE AUTO-CONFIG: unexpected error %r; falling back to "
            "user-supplied configuration",
            e,
        )
