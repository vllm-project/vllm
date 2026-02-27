#!/usr/bin/env python3
"""Tool definitions and implementations for SceneIQ verification.

Provides 4 statistical tools in OpenAI function-calling format,
their Python implementations, oracle variants, and a dispatcher.
"""

from __future__ import annotations

import json
import os
from typing import Any

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_STATS_PATH = os.path.join(_DIR, "tool_stats.json")

# ------------------------------------------------------------------
# Hardcoded fallback data (used when tool_stats.json is absent)
# ------------------------------------------------------------------
CLASS_FREQUENCIES: dict[str, float] = {
    "nominal": 0.781,
    "flagger": 0.077,
    "flooded": 0.072,
    "incident_zone": 0.037,
    "mounted_police": 0.032,
}

CONFUSION_PAIRS: dict[str, dict[str, Any]] = {
    "incident_zone": {
        "confused_with": "nominal",
        "error_rate": 0.506,
        "note": (
            "incident_zone is predicted 12.5x more often than "
            "it actually occurs. 46.8% of all predictions are "
            "incident_zone but only 3.7% of ground truth is. "
            "Most incident_zone predictions are actually "
            "nominal scenes with visual triggers (traffic "
            "cones, barriers). Look carefully: are there "
            "actual emergency vehicles, crashes, or road "
            "closures? If not, this is likely nominal."
        ),
    },
    "flooded": {
        "confused_with": "nominal",
        "error_rate": 0.264,
        "note": (
            "26.4% of flooded scenes are misclassified. "
            "Look for standing water on the road surface."
        ),
    },
    "flagger": {
        "confused_with": "incident_zone",
        "error_rate": 0.335,
        "note": (
            "33.5% of flagger scenes are confused with "
            "incident_zone. Look for a human flagging "
            "traffic."
        ),
    },
    "mounted_police": {
        "confused_with": "incident_zone",
        "error_rate": 0.287,
        "note": (
            "28.7% of mounted_police scenes are confused "
            "with incident_zone. Look for horses."
        ),
    },
}

# Keys are (long_action, lat_action) tuples
COOCCURRENCE: dict[str, dict[tuple[str, str], int]] = {
    "nominal": {("null", "null"): 6728},
    "incident_zone": {
        ("slowdown", "null"): 107,
        ("stop", "null"): 72,
        ("slowdown", "lc_left"): 46,
        ("proceed", "null"): 32,
        ("slowdown", "lc_right"): 29,
        ("stop", "lc_left"): 18,
        ("proceed", "lc_left"): 9,
        ("stop", "lc_right"): 5,
        ("proceed", "lc_right"): 4,
    },
    "flooded": {
        ("slowdown", "null"): 348,
        ("stop", "null"): 171,
        ("proceed", "null"): 99,
    },
    "flagger": {
        ("stop", "null"): 391,
        ("slowdown", "null"): 190,
        ("proceed", "null"): 79,
    },
    "mounted_police": {("null", "null"): 272},
}

# Fallback waypoint stats (mean, std, min, max) per scene+action
WAYPOINT_STATS: dict[str, dict[str, dict[str, float]]] = {
    "nominal": {
        "null": {
            "x_mean": 0.0,
            "x_std": 0.1,
            "y_mean": 0.0,
            "y_std": 0.1,
            "x_min": -0.5,
            "x_max": 0.5,
            "y_min": -0.5,
            "y_max": 0.5,
        }
    },
}

# ------------------------------------------------------------------
# Load stats from JSON if available
# ------------------------------------------------------------------
_loaded_stats: dict[str, Any] | None = None


def _load_stats() -> dict[str, Any]:
    global _loaded_stats  # noqa: PLW0603
    if _loaded_stats is not None:
        return _loaded_stats
    if os.path.exists(TOOL_STATS_PATH):
        with open(TOOL_STATS_PATH) as f:
            _loaded_stats = json.load(f)
    else:
        _loaded_stats = {}
    return _loaded_stats


def _get_class_frequencies() -> dict[str, float]:
    stats = _load_stats()
    return stats.get("class_frequencies", CLASS_FREQUENCIES)


def _get_confusion_pairs() -> dict[str, dict[str, Any]]:
    stats = _load_stats()
    return stats.get("confusion_pairs", CONFUSION_PAIRS)


def _get_cooccurrence() -> dict[str, dict[str, int]]:
    """Return co-occurrence data.

    JSON stores keys as ``"long|lat"`` strings; the hardcoded
    fallback uses ``(long, lat)`` tuples.  This helper always
    returns the JSON-style ``"long|lat"`` string keys so callers
    have a uniform interface.
    """
    stats = _load_stats()
    raw = stats.get("cooccurrence")
    if raw is not None:
        return raw
    # Convert tuple keys to pipe-separated strings
    converted: dict[str, dict[str, int]] = {}
    for scene, actions in COOCCURRENCE.items():
        converted[scene] = {
            f"{k[0]}|{k[1]}": v for k, v in actions.items()
        }
    return converted


def _get_waypoint_stats() -> dict[str, dict[str, dict[str, float]]]:
    stats = _load_stats()
    return stats.get("waypoint_stats", WAYPOINT_STATS)


# ------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ------------------------------------------------------------------
TOOL_PRIOR_CHECK: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_scene_prior",
        "description": (
            "Check how common a predicted scene type is in the "
            "training data. Returns the base rate of the "
            "predicted scene and the most common scene overall. "
            "Use this to verify whether a rare prediction is "
            "plausible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_scene": {
                    "type": "string",
                    "enum": [
                        "nominal",
                        "flooded",
                        "incident_zone",
                        "mounted_police",
                        "flagger",
                    ],
                    "description": (
                        "The predicted scene classification "
                        "to check"
                    ),
                }
            },
            "required": ["predicted_scene"],
        },
    },
}

TOOL_SCENE_ACTION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_scene_action_compatibility",
        "description": (
            "Check whether a predicted action is compatible "
            "with a predicted scene type based on historical "
            "co-occurrence data. Returns whether this "
            "combination has been observed and what actions "
            "are typical for this scene."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "enum": [
                        "nominal",
                        "flooded",
                        "incident_zone",
                        "mounted_police",
                        "flagger",
                    ],
                },
                "long_action": {
                    "type": "string",
                    "enum": [
                        "stop",
                        "slowdown",
                        "proceed",
                        "null",
                    ],
                    "description": (
                        "Longitudinal action prediction"
                    ),
                },
                "lat_action": {
                    "type": "string",
                    "enum": ["lc_left", "lc_right", "null"],
                    "description": (
                        "Lateral action prediction"
                    ),
                },
            },
            "required": [
                "scene",
                "long_action",
                "lat_action",
            ],
        },
    },
}

TOOL_WAYPOINT_CHECK: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_waypoint_feasibility",
        "description": (
            "Check whether predicted waypoint deltas are "
            "within the typical range for a given "
            "scene-action combination. Returns feasibility "
            "assessment and typical waypoint statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "enum": [
                        "nominal",
                        "flooded",
                        "incident_zone",
                        "mounted_police",
                        "flagger",
                    ],
                },
                "long_action": {
                    "type": "string",
                    "enum": [
                        "stop",
                        "slowdown",
                        "proceed",
                        "null",
                    ],
                },
                "first_waypoint_x": {
                    "type": "number",
                    "description": "First waypoint x-delta",
                },
                "first_waypoint_y": {
                    "type": "number",
                    "description": "First waypoint y-delta",
                },
            },
            "required": [
                "scene",
                "long_action",
                "first_waypoint_x",
                "first_waypoint_y",
            ],
        },
    },
}

TOOL_CONFUSION_CHECK: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_confusion_risk",
        "description": (
            "Check whether a predicted scene type is commonly "
            "confused with another class. Returns the "
            "historical confusion rate and what class it is "
            "most often confused with. High-risk predictions "
            "should be double-checked carefully."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_scene": {
                    "type": "string",
                    "enum": [
                        "nominal",
                        "flooded",
                        "incident_zone",
                        "mounted_police",
                        "flagger",
                    ],
                }
            },
            "required": ["predicted_scene"],
        },
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    TOOL_PRIOR_CHECK,
    TOOL_SCENE_ACTION,
    TOOL_WAYPOINT_CHECK,
    TOOL_CONFUSION_CHECK,
]

# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------


def check_scene_prior(
    predicted_scene: str,
) -> dict[str, Any]:
    """Return base-rate information for *predicted_scene*."""
    freqs = _get_class_frequencies()
    rate = freqs.get(predicted_scene, 0.0)
    most_common = max(freqs, key=lambda k: freqs[k])
    return {
        "predicted_scene": predicted_scene,
        "base_rate": rate,
        "most_common_scene": most_common,
        "most_common_rate": freqs[most_common],
        "is_rare": rate < 0.05,
        "warning": (
            f"{predicted_scene} is rare "
            f"(base rate {rate:.1%}). Consider whether "
            f"this could actually be {most_common}."
            if rate < 0.05
            else None
        ),
    }


def check_scene_action_compatibility(
    scene: str,
    long_action: str,
    lat_action: str,
) -> dict[str, Any]:
    """Check if a scene-action pair has been observed."""
    coocc = _get_cooccurrence()
    scene_data = coocc.get(scene, {})
    key = f"{long_action}|{lat_action}"
    count = scene_data.get(key, 0)
    total = sum(scene_data.values()) if scene_data else 0
    typical = []
    if scene_data:
        sorted_actions = sorted(
            scene_data.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        typical = [
            {
                "actions": k,
                "count": v,
                "rate": v / total if total else 0,
            }
            for k, v in sorted_actions[:3]
        ]
    return {
        "scene": scene,
        "long_action": long_action,
        "lat_action": lat_action,
        "observed": count > 0,
        "count": count,
        "rate": count / total if total else 0.0,
        "typical_actions": typical,
        "warning": (
            f"({long_action}, {lat_action}) has never been "
            f"observed with {scene}."
            if count == 0
            else None
        ),
    }


def check_waypoint_feasibility(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
) -> dict[str, Any]:
    """Check if waypoint deltas are within typical range."""
    wp = _get_waypoint_stats()
    scene_stats = wp.get(scene, {})
    action_stats = scene_stats.get(long_action)
    if action_stats is None:
        return {
            "scene": scene,
            "long_action": long_action,
            "first_waypoint_x": first_waypoint_x,
            "first_waypoint_y": first_waypoint_y,
            "feasible": None,
            "warning": (
                f"No waypoint data for {scene}+"
                f"{long_action}."
            ),
        }
    x_mean = action_stats.get("x_mean", 0.0)
    x_std = action_stats.get("x_std", 1.0)
    y_mean = action_stats.get("y_mean", 0.0)
    y_std = action_stats.get("y_std", 1.0)
    x_z = (
        abs(first_waypoint_x - x_mean) / x_std
        if x_std > 0
        else 0.0
    )
    y_z = (
        abs(first_waypoint_y - y_mean) / y_std
        if y_std > 0
        else 0.0
    )
    feasible = x_z < 3.0 and y_z < 3.0
    return {
        "scene": scene,
        "long_action": long_action,
        "first_waypoint_x": first_waypoint_x,
        "first_waypoint_y": first_waypoint_y,
        "feasible": feasible,
        "x_z_score": round(x_z, 2),
        "y_z_score": round(y_z, 2),
        "typical_x_range": [
            round(x_mean - 2 * x_std, 3),
            round(x_mean + 2 * x_std, 3),
        ],
        "typical_y_range": [
            round(y_mean - 2 * y_std, 3),
            round(y_mean + 2 * y_std, 3),
        ],
        "warning": (
            "Waypoint deltas are outside typical range."
            if not feasible
            else None
        ),
    }


def check_confusion_risk(
    predicted_scene: str,
) -> dict[str, Any]:
    """Return confusion-risk info for *predicted_scene*."""
    pairs = _get_confusion_pairs()
    info = pairs.get(predicted_scene)
    if info is None:
        return {
            "predicted_scene": predicted_scene,
            "has_confusion_risk": False,
            "note": (
                f"{predicted_scene} is not commonly "
                "confused with other classes."
            ),
        }
    return {
        "predicted_scene": predicted_scene,
        "has_confusion_risk": True,
        "confused_with": info["confused_with"],
        "error_rate": info["error_rate"],
        "note": info["note"],
    }


# ------------------------------------------------------------------
# Oracle tool variants -- return ground truth directly
# ------------------------------------------------------------------


def oracle_check_scene_prior(
    predicted_scene: str,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oracle: reveals the ground-truth scene type."""
    gt = ground_truth or {}
    gt_scene = gt.get("scene_type_gt", "unknown")
    is_correct = predicted_scene == gt_scene
    result = check_scene_prior(predicted_scene)
    result["oracle"] = True
    result["ground_truth_scene"] = gt_scene
    result["prediction_is_correct"] = is_correct
    if not is_correct:
        result["warning"] = (
            f"ORACLE: The correct scene is {gt_scene}, "
            f"not {predicted_scene}."
        )
    return result


def oracle_check_confusion_risk(
    predicted_scene: str,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oracle: reveals whether the prediction is confused."""
    gt = ground_truth or {}
    gt_scene = gt.get("scene_type_gt", "unknown")
    is_correct = predicted_scene == gt_scene
    result = check_confusion_risk(predicted_scene)
    result["oracle"] = True
    result["ground_truth_scene"] = gt_scene
    result["prediction_is_correct"] = is_correct
    if not is_correct:
        result["warning"] = (
            f"ORACLE: This prediction is WRONG. The "
            f"correct scene is {gt_scene}."
        )
    return result


def oracle_check_scene_action_compatibility(
    scene: str,
    long_action: str,
    lat_action: str,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oracle: reveals correct scene-action combination."""
    gt = ground_truth or {}
    gt_scene = gt.get("scene_type_gt", "unknown")
    gt_long = gt.get("long_action_gt", "unknown")
    gt_lat = gt.get("lat_action_gt", "unknown")
    result = check_scene_action_compatibility(
        scene, long_action, lat_action
    )
    result["oracle"] = True
    result["ground_truth_scene"] = gt_scene
    result["ground_truth_long_action"] = gt_long
    result["ground_truth_lat_action"] = gt_lat
    scene_ok = scene == gt_scene
    long_ok = long_action == gt_long
    lat_ok = lat_action == gt_lat
    if not (scene_ok and long_ok and lat_ok):
        parts = []
        if not scene_ok:
            parts.append(f"scene should be {gt_scene}")
        if not long_ok:
            parts.append(f"long_action should be {gt_long}")
        if not lat_ok:
            parts.append(f"lat_action should be {gt_lat}")
        result["warning"] = (
            "ORACLE: " + ", ".join(parts) + "."
        )
    return result


def oracle_check_waypoint_feasibility(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oracle: reveals correct scene for waypoint check."""
    gt = ground_truth or {}
    gt_scene = gt.get("scene_type_gt", "unknown")
    gt_long = gt.get("long_action_gt", "unknown")
    result = check_waypoint_feasibility(
        scene, long_action, first_waypoint_x, first_waypoint_y
    )
    result["oracle"] = True
    result["ground_truth_scene"] = gt_scene
    result["ground_truth_long_action"] = gt_long
    if scene != gt_scene:
        result["warning"] = (
            f"ORACLE: Scene should be {gt_scene}, not "
            f"{scene}. Re-evaluate waypoints."
        )
    return result


# ------------------------------------------------------------------
# Dispatcher maps
# ------------------------------------------------------------------
TOOL_MAP: dict[str, Any] = {
    "check_scene_prior": check_scene_prior,
    "check_scene_action_compatibility": (
        check_scene_action_compatibility
    ),
    "check_waypoint_feasibility": check_waypoint_feasibility,
    "check_confusion_risk": check_confusion_risk,
}

ORACLE_TOOL_MAP: dict[str, Any] = {
    "check_scene_prior": oracle_check_scene_prior,
    "check_scene_action_compatibility": (
        oracle_check_scene_action_compatibility
    ),
    "check_waypoint_feasibility": (
        oracle_check_waypoint_feasibility
    ),
    "check_confusion_risk": oracle_check_confusion_risk,
}


def get_tools_for_condition(
    condition_name: str,
) -> list[dict[str, Any]]:
    """Return the tool definitions for *condition_name*.

    Supported conditions:
      - ``prior_only``   -- only check_scene_prior
      - ``confusion_only`` -- only check_confusion_risk
      - ``all_tools`` / ``oracle`` / ``staged`` -- all 4 tools
    """
    mapping: dict[str, list[dict[str, Any]]] = {
        "prior_only": [TOOL_PRIOR_CHECK],
        "confusion_only": [TOOL_CONFUSION_CHECK],
        "all_tools": ALL_TOOLS,
        "oracle": ALL_TOOLS,
        "staged": ALL_TOOLS,
    }
    return mapping.get(condition_name, ALL_TOOLS)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a tool by name, optionally in oracle mode.

    Parameters
    ----------
    tool_name:
        One of the four tool function names.
    arguments:
        Keyword arguments for the tool function.
    ground_truth:
        If provided, use oracle variants that reveal GT.

    Returns
    -------
    dict with tool result (JSON-serializable).
    """
    if ground_truth is not None:
        fn = ORACLE_TOOL_MAP.get(tool_name)
        if fn is not None:
            return fn(**arguments, ground_truth=ground_truth)
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_MAP.keys()),
        }
    return fn(**arguments)
