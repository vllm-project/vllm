#!/usr/bin/env python3
"""Tool definitions and implementations for SceneIQ verification -- v2.

Provides 4 statistical tools at 4 levels of reasoning scaffolding:
  Level 1: Raw Data -- statistics only
  Level 2: Interpreted Data -- statistics + natural language interpretation
  Level 3: Procedural -- visual checklist and decision procedure
  Level 4: Prescriptive -- explicit if/then decision rules

All levels share the same OpenAI function-calling schemas. The level
parameter controls how much reasoning support the tool return value
provides.

Usage:
    from tools_v2 import execute_tool_v2, get_tools_for_level

    # Same schemas regardless of level
    tools = get_tools_for_level(level=3)

    # Level controls richness of return value
    result = execute_tool_v2("check_confusion_risk",
                             {"predicted_scene": "incident_zone"},
                             level=3)

Backward-compatible: execute_tool() from tools.py still works via
execute_tool_v2(..., level=1).
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
# Load stats from JSON
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


# ------------------------------------------------------------------
# Hardcoded fallback data
# ------------------------------------------------------------------
CLASS_FREQUENCIES: dict[str, float] = {
    "nominal": 0.7827,
    "flagger": 0.0766,
    "flooded": 0.0718,
    "incident_zone": 0.0374,
    "mounted_police": 0.0316,
}

CONFUSION_PAIRS: dict[str, dict[str, Any]] = {
    "incident_zone": {
        "confused_with": "nominal",
        "error_rate": 0.41,
        "error_count": 132,
        "total": 322,
        "note": "41.0% of incident_zone scenes are misclassified as nominal.",
    },
    "flooded": {
        "confused_with": "incident_zone",
        "error_rate": 0.28,
        "error_count": 173,
        "total": 618,
        "note": "28.0% of flooded scenes are misclassified as incident_zone.",
    },
    "flagger": {
        "confused_with": "incident_zone",
        "error_rate": 0.344,
        "error_count": 227,
        "total": 660,
        "note": "34.4% of flagger scenes are misclassified as incident_zone.",
    },
    "mounted_police": {
        "confused_with": "nominal",
        "error_rate": 0.507,
        "error_count": 138,
        "total": 272,
        "note": "50.7% of mounted_police scenes are misclassified as nominal.",
    },
    "nominal": {
        "confused_with": "incident_zone",
        "error_rate": 0.509,
        "error_count": 3430,
        "total": 6741,
        "note": "50.9% of nominal scenes are misclassified as incident_zone.",
    },
}

COOCCURRENCE: dict[str, dict[str, int]] = {
    "nominal": {"null|null": 6741},
    "incident_zone": {
        "proceed|lc_left": 8,
        "proceed|lc_right": 5,
        "proceed|null": 34,
        "slowdown|lc_left": 41,
        "slowdown|lc_right": 23,
        "slowdown|null": 104,
        "stop|lc_left": 22,
        "stop|lc_right": 4,
        "stop|null": 81,
    },
    "flooded": {
        "proceed|null": 103,
        "slowdown|null": 346,
        "stop|null": 169,
    },
    "flagger": {
        "proceed|null": 69,
        "slowdown|null": 207,
        "stop|null": 384,
    },
    "mounted_police": {"null|null": 272},
}

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
# Data accessors
# ------------------------------------------------------------------
def _get_class_frequencies() -> dict[str, float]:
    stats = _load_stats()
    return stats.get("class_frequencies", CLASS_FREQUENCIES)


def _get_confusion_pairs() -> dict[str, dict[str, Any]]:
    stats = _load_stats()
    return stats.get("confusion_pairs", CONFUSION_PAIRS)


def _get_cooccurrence() -> dict[str, dict[str, int]]:
    stats = _load_stats()
    raw = stats.get("cooccurrence")
    if raw is not None:
        return raw
    return COOCCURRENCE


def _get_waypoint_stats() -> dict[str, dict[str, dict[str, float]]]:
    stats = _load_stats()
    return stats.get("waypoint_stats", WAYPOINT_STATS)


# ------------------------------------------------------------------
# Domain knowledge: scene-specific visual checklists and rules
# ------------------------------------------------------------------

# Visual features that must be present to confirm a scene type
SCENE_MUST_PRESENT: dict[str, list[str]] = {
    "incident_zone": [
        "Emergency vehicles with flashing lights actively responding",
        "Crashed or damaged vehicles on the roadway",
        "Debris or wreckage scattered on the road surface",
        "Road physically blocked by accident or emergency",
        "First responders (police, fire, EMS) on foot in the roadway",
    ],
    "flooded": [
        "Standing water visibly covering the road surface",
        "Reflective water surface disrupting normal road texture",
        "Water pooling in low spots or underpasses",
        "Spray or wake from vehicles driving through water",
    ],
    "flagger": [
        "A human figure standing on or near the roadway",
        "Person wearing high-visibility vest or clothing",
        "Person holding a STOP/SLOW sign, flag, or paddle",
        "Active hand signals directing traffic flow",
    ],
    "mounted_police": [
        "Horse clearly visible on or near the roadway",
        "Rider (officer) on horseback",
        "Police markings or uniform visible on rider",
    ],
    "nominal": [
        "Normal traffic flow with no obstructions",
        "Standard road conditions (dry, clear lanes)",
        "No emergency vehicles, flaggers, horses, or standing water",
    ],
}

# Visual triggers that are commonly mistaken for a scene type
SCENE_COMMONLY_MISTAKEN: dict[str, list[str]] = {
    "incident_zone": [
        "Traffic cones used for lane guidance or construction (not accidents)",
        "Concrete barriers or jersey walls along construction zones",
        "Parked emergency vehicles with lights OFF (not actively responding)",
        "Road work signage or detour signs",
        "Heavy traffic or slow-moving congestion (not an incident)",
        "Shadows creating dark patches that look like debris",
        "Construction equipment parked on shoulder",
    ],
    "flooded": [
        "Wet road surface from recent rain (not standing water)",
        "Reflective road markings or paint that look like water",
        "Shadows on road surface creating dark patches",
        "Puddles in gutters that do not cover the driving lanes",
    ],
    "flagger": [
        "Pedestrians near the road (not directing traffic)",
        "Road signs on posts (no human holding them)",
        "Construction workers not actively flagging",
        "Mannequins or sign holders at businesses",
    ],
    "mounted_police": [
        "Large dogs or other animals near the road",
        "Motorcycles (not horses)",
        "Statues or sculptures near the roadway",
        "Pedestrians walking alongside the road",
    ],
    "nominal": [
        "Construction zone cones (could indicate flagger or incident_zone)",
        "Emergency vehicle parked with lights off (could be incident_zone)",
        "Wet road surface (could be flooded)",
    ],
}

# Distinguishing features between commonly confused pairs
CONFUSION_DISTINGUISHERS: dict[str, dict[str, list[str]]] = {
    "incident_zone": {
        "nominal": [
            "Emergency lights FLASHING vs. no emergency lights",
            "Vehicles at ABNORMAL angles vs. normal lane positions",
            "ACTIVE first responders on road vs. no emergency personnel",
            "Road BLOCKED by wreckage vs. road clear with construction cones",
            "Broken glass/debris on pavement vs. clean road surface",
            "Traffic STOPPED in all lanes vs. normal flow through work zone",
        ],
    },
    "nominal": {
        "incident_zone": [
            "Cones in orderly lane-guidance pattern vs. "
            "scattered around wreckage",
            "Construction workers in work zone vs. first responders at crash scene",
            "Traffic flowing (even if slowly) vs. traffic completely stopped",
            "No damaged vehicles visible vs. visible vehicle damage",
            "Standard work zone signage vs. emergency scene tape/barriers",
        ],
    },
    "flagger": {
        "incident_zone": [
            "Person with STOP/SLOW sign vs. first "
            "responders without flagging equipment",
            "Organized construction zone vs. chaotic emergency scene",
            "Traffic being DIRECTED through vs. traffic STOPPED by blockage",
            "High-vis vest and flag/sign visible vs. emergency uniforms",
        ],
    },
    "flooded": {
        "incident_zone": [
            "Water covering road surface vs. dry road with emergency vehicles",
            "No emergency vehicles vs. flashing lights present",
            "Road passable but wet vs. road blocked by accident",
        ],
    },
    "mounted_police": {
        "nominal": [
            "Horse clearly visible on/near road vs. no large animals present",
            "Rider in uniform on horseback vs. standard pedestrians/vehicles",
            "Horse body shape distinct from vehicles vs. only vehicles present",
        ],
    },
}

# Scene-specific decision rules (L4)
SCENE_DECISION_RULES: dict[str, str] = {
    "incident_zone": (
        "You predicted incident_zone. Follow these steps exactly:\n"
        "1. Do you see emergency vehicles with FLASHING lights? "
        "If NO, go to step 2.\n"
        "2. Do you see crashed, damaged, or wrecked vehicles? "
        "If NO, go to step 3.\n"
        "3. Is the road physically blocked by an accident or debris? "
        "If NO, go to step 4.\n"
        "4. Do you see first responders (police/fire/EMS) actively "
        "working on the roadway? If NO, go to step 5.\n"
        "5. If you answered NO to ALL of the above, this is NOT "
        "incident_zone. Reclassify as nominal. Traffic cones, "
        "barriers, and construction equipment alone do NOT make a "
        "scene incident_zone.\n"
        "6. If you answered YES to ANY of steps 1-4, confirm "
        "incident_zone.\n"
        "DEFAULT: When uncertain, classify as nominal. It is correct "
        "78.3% of the time."
    ),
    "flooded": (
        "You predicted flooded. Follow these steps exactly:\n"
        "1. Do you see standing water covering the road surface? "
        "If NO, this is NOT flooded.\n"
        "2. Is the water clearly on the driving lanes (not just in "
        "gutters or shoulders)? If NO, this is NOT flooded.\n"
        "3. Can you see water reflections or spray from vehicles? "
        "If YES, confirm flooded.\n"
        "4. If the road is only wet (shiny but no pooling), this is "
        "nominal with rain, NOT flooded.\n"
        "DEFAULT: If unsure whether water is standing or just wet "
        "pavement, classify as nominal."
    ),
    "flagger": (
        "You predicted flagger. Follow these steps exactly:\n"
        "1. Do you see a HUMAN figure on or near the road? "
        "If NO, this is NOT flagger.\n"
        "2. Is the person holding a sign, flag, or paddle? "
        "If YES, confirm flagger.\n"
        "3. Is the person wearing high-visibility clothing and "
        "actively directing traffic? If YES, confirm flagger.\n"
        "4. If you only see construction cones/barriers but NO human "
        "directing traffic, this may be incident_zone or nominal, "
        "NOT flagger.\n"
        "DEFAULT: If no clear human flagging traffic, do not classify "
        "as flagger."
    ),
    "mounted_police": (
        "You predicted mounted_police. Follow these steps exactly:\n"
        "1. Do you see a HORSE on or near the roadway? "
        "If NO, this is NOT mounted_police. Reclassify as nominal.\n"
        "2. Is there a rider (officer) on the horse? "
        "If YES, confirm mounted_police.\n"
        "3. If you see a large animal but cannot confirm it is a "
        "horse with a rider, do NOT classify as mounted_police.\n"
        "DEFAULT: When uncertain, classify as nominal."
    ),
    "nominal": (
        "You predicted nominal. This is correct 78.3% of the time, "
        "so the prior is in your favor. Verify:\n"
        "1. Do you see ANY of the following: standing water, emergency "
        "vehicles with lights, a person flagging traffic, or a horse? "
        "If YES, reconsider the specific non-nominal class.\n"
        "2. If the scene shows only normal driving conditions (standard "
        "traffic, clear road, no unusual objects), confirm nominal.\n"
        "DEFAULT: Nominal is the safest default classification."
    ),
}

# Action-specific decision rules
ACTION_DECISION_RULES: dict[str, dict[str, str]] = {
    "nominal": {
        "rule": (
            "Scene is nominal. The ONLY valid action is (null, null). "
            "If you predicted any other action, change it to (null, null). "
            "There are zero exceptions in the training data."
        ),
    },
    "mounted_police": {
        "rule": (
            "Scene is mounted_police. The ONLY valid action is (null, null). "
            "If you predicted any other action, change it to (null, null). "
            "There are zero exceptions in the training data."
        ),
    },
    "incident_zone": {
        "rule": (
            "Scene is incident_zone. Valid longitudinal actions: stop, "
            "slowdown, proceed. Valid lateral actions: null, lc_left, lc_right.\n"
            "Most common: slowdown|null (32.3%), stop|null (25.2%), "
            "slowdown|lc_left (12.7%).\n"
            "Check: Is the road blocked? -> stop. Partially blocked? -> "
            "slowdown. Clear but hazardous? -> proceed.\n"
            "Is lane change needed? Only if the obstruction blocks your "
            "current lane AND an adjacent lane is available."
        ),
    },
    "flooded": {
        "rule": (
            "Scene is flooded. Valid longitudinal: stop, slowdown, proceed. "
            "Lateral: null ONLY (no lane changes for floods).\n"
            "Most common: slowdown|null (56.0%), stop|null (27.3%), "
            "proceed|null (16.7%).\n"
            "How deep is the water? Deep/road covered -> stop. "
            "Shallow/passable -> slowdown. Minimal -> proceed."
        ),
    },
    "flagger": {
        "rule": (
            "Scene is flagger. Valid longitudinal: stop, slowdown, proceed. "
            "Lateral: null ONLY (no lane changes for flagger scenes).\n"
            "Most common: stop|null (58.2%), slowdown|null (31.4%), "
            "proceed|null (10.5%).\n"
            "What is the flagger signaling? STOP sign -> stop. "
            "SLOW sign -> slowdown. Waving through -> proceed."
        ),
    },
}

# Waypoint spatial interpretation helpers
WAYPOINT_SPATIAL_GUIDANCE: dict[str, str] = {
    "far_left": (
        "The waypoint is far to the LEFT of typical center. This could indicate:\n"
        "- A lane change to the left (valid for incident_zone with lc_left)\n"
        "- A left curve in the road (valid if road curves left)\n"
        "- An error if the road is straight and no lane change is intended\n"
        "Check: Does the road curve left? Is a left lane change appropriate?"
    ),
    "far_right": (
        "The waypoint is far to the RIGHT of typical center. This could indicate:\n"
        "- A lane change to the right (valid for incident_zone with lc_right)\n"
        "- A right curve in the road (valid if road curves right)\n"
        "- An error if the road is straight and no lane change is intended\n"
        "Check: Does the road curve right? Is a right lane change appropriate?"
    ),
    "far_forward": (
        "The waypoint suggests MORE forward motion than typical. This could indicate:\n"
        "- Higher speed than expected for this action type\n"
        "- The model expects to proceed/accelerate when slowdown is appropriate\n"
        "Check: Is the action appropriate for the scene severity?"
    ),
    "far_behind": (
        "The waypoint suggests LESS forward motion (or reversal) than typical. "
        "This could indicate:\n"
        "- Hard braking or stopping\n"
        "- The model is over-reacting to the scene\n"
        "Check: Does the scene actually require hard braking?"
    ),
    "normal": (
        "The waypoint is within the typical range for this scene-action "
        "combination. No spatial concerns."
    ),
}


# ------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format) -- same for all levels
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
                        "The predicted scene classification to check"
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
                    "enum": ["stop", "slowdown", "proceed", "null"],
                    "description": "Longitudinal action prediction",
                },
                "lat_action": {
                    "type": "string",
                    "enum": ["lc_left", "lc_right", "null"],
                    "description": "Lateral action prediction",
                },
            },
            "required": ["scene", "long_action", "lat_action"],
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
                    "enum": ["stop", "slowdown", "proceed", "null"],
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


def get_tools_for_level(level: int = 1) -> list[dict[str, Any]]:
    """Return tool definitions. Schemas are the same for all levels."""
    _ = level  # schemas do not vary by level
    return ALL_TOOLS


# ------------------------------------------------------------------
# Helper: compute co-occurrence stats for a scene
# ------------------------------------------------------------------
def _scene_action_stats(scene: str) -> tuple[dict[str, int], int, list[dict]]:
    """Return (scene_data, total_count, top_3_actions) for a scene."""
    coocc = _get_cooccurrence()
    scene_data = coocc.get(scene, {})
    total = sum(scene_data.values()) if scene_data else 0
    typical = []
    if scene_data:
        sorted_actions = sorted(
            scene_data.items(), key=lambda x: x[1], reverse=True
        )
        typical = [
            {
                "actions": k,
                "count": v,
                "rate": round(v / total, 3) if total else 0,
            }
            for k, v in sorted_actions[:5]
        ]
    return scene_data, total, typical


# ------------------------------------------------------------------
# check_scene_prior -- all 4 levels
# ------------------------------------------------------------------
def _check_scene_prior_l1(predicted_scene: str) -> dict[str, Any]:
    freqs = _get_class_frequencies()
    rate = freqs.get(predicted_scene, 0.0)
    most_common = max(freqs, key=lambda k: freqs[k])
    return {
        "predicted_scene": predicted_scene,
        "base_rate": rate,
        "most_common_scene": most_common,
        "most_common_rate": freqs[most_common],
    }


def _check_scene_prior_l2(predicted_scene: str) -> dict[str, Any]:
    result = _check_scene_prior_l1(predicted_scene)
    rate = result["base_rate"]
    most_common = result["most_common_scene"]
    mc_rate = result["most_common_rate"]

    # Compute over-prediction ratio relative to base rate
    # (the model predicts incident_zone at ~46.8%)
    prediction_rates = {
        "incident_zone": 0.468,
        "nominal": 0.296,
        "flooded": 0.108,
        "flagger": 0.087,
        "mounted_police": 0.041,
    }
    pred_rate = prediction_rates.get(predicted_scene)
    over_pred = (
        round(pred_rate / rate, 1) if pred_rate and rate > 0 else None
    )

    if rate < 0.05:
        interpretation = (
            f"{predicted_scene} occurs in only {rate:.1%} of real scenes, "
            f"making it a rare class. "
        )
        if over_pred and over_pred > 2:
            interpretation += (
                f"The model predicts {predicted_scene} at "
                f"~{pred_rate:.1%}, which is {over_pred}x more than the "
                f"true rate. "
            )
        interpretation += (
            f"The most common scene is {most_common} at "
            f"{mc_rate:.1%}. If you are not highly confident in "
            f"the visual evidence, {most_common} is the statistically "
            f"safer choice."
        )
    elif rate > 0.5:
        interpretation = (
            f"{predicted_scene} is the most common class at "
            f"{rate:.1%}. This prediction is statistically likely "
            f"to be correct based on base rate alone."
        )
    else:
        interpretation = (
            f"{predicted_scene} occurs at {rate:.1%} -- a moderately "
            f"common class. Verify that the visual evidence matches "
            f"the expected features for this scene type."
        )

    result["interpretation"] = interpretation
    return result


def _check_scene_prior_l3(predicted_scene: str) -> dict[str, Any]:
    result = _check_scene_prior_l2(predicted_scene)
    rate = result["base_rate"]

    # Classify risk level
    if rate < 0.04:
        risk = "high"
        confidence_msg = (
            f"This is a rare class ({rate:.1%} base rate). "
            f"Strong, unambiguous visual evidence is required to "
            f"justify this prediction."
        )
    elif rate < 0.08:
        risk = "moderate"
        confidence_msg = (
            f"This is an uncommon class ({rate:.1%} base rate). "
            f"Clear visual evidence should be present."
        )
    else:
        risk = "low"
        confidence_msg = (
            f"This is a common class ({rate:.1%} base rate). "
            f"Standard visual confirmation is sufficient."
        )

    result["classification_risk"] = risk
    result["confidence_required"] = confidence_msg
    result["visual_checklist"] = {
        "must_be_present": SCENE_MUST_PRESENT.get(predicted_scene, []),
        "commonly_mistaken_triggers": SCENE_COMMONLY_MISTAKEN.get(
            predicted_scene, []
        ),
    }
    result["decision_framework"] = (
        f"Count how many items from 'must_be_present' you can "
        f"identify in the image. If ZERO, reclassify as "
        f"{result['most_common_scene']}. If ONE, state which feature "
        f"you see and your confidence level. If TWO or more, confirm "
        f"{predicted_scene}. Also check 'commonly_mistaken_triggers' -- "
        f"if what you see matches those more than 'must_be_present', "
        f"reconsider your classification."
    )
    return result


def _check_scene_prior_l4(predicted_scene: str) -> dict[str, Any]:
    result = _check_scene_prior_l3(predicted_scene)

    result["decision_rule"] = SCENE_DECISION_RULES.get(
        predicted_scene,
        (
            f"No specific decision rule for {predicted_scene}. "
            f"Use the visual checklist and your best judgment."
        ),
    )
    result["final_instruction"] = (
        "Apply the decision rule above step by step. For each "
        "question, answer YES or NO based on what you see in the "
        "image. Follow the rule to its conclusion. Output ONLY the "
        "final scene classification."
    )
    return result


# ------------------------------------------------------------------
# check_confusion_risk -- all 4 levels
# ------------------------------------------------------------------
def _check_confusion_risk_l1(predicted_scene: str) -> dict[str, Any]:
    pairs = _get_confusion_pairs()
    info = pairs.get(predicted_scene)
    if info is None:
        return {
            "predicted_scene": predicted_scene,
            "has_confusion_risk": False,
        }
    return {
        "predicted_scene": predicted_scene,
        "has_confusion_risk": True,
        "confused_with": info["confused_with"],
        "error_rate": info["error_rate"],
    }


def _check_confusion_risk_l2(predicted_scene: str) -> dict[str, Any]:
    result = _check_confusion_risk_l1(predicted_scene)
    pairs = _get_confusion_pairs()
    info = pairs.get(predicted_scene)

    if info is None:
        result["interpretation"] = (
            f"{predicted_scene} does not have a significant confusion "
            f"pattern with other classes."
        )
        return result

    confused_with = info["confused_with"]
    error_rate = info["error_rate"]
    error_count = info.get("error_count", "unknown")
    total = info.get("total", "unknown")

    # Build a rich interpretation
    interpretation = (
        f"{error_rate:.1%} of {predicted_scene} predictions are "
        f"actually {confused_with} ({error_count} errors out of "
        f"{total} predictions). "
    )

    # Add context about what this means in practice
    if predicted_scene == "incident_zone":
        interpretation += (
            "incident_zone is the most over-predicted class: it is "
            "predicted 12.5x more often than it actually occurs. Most "
            "incident_zone predictions are actually nominal scenes "
            "with visual triggers like traffic cones, barriers, or "
            "construction equipment."
        )
    elif predicted_scene == "nominal" and confused_with == "incident_zone":
        interpretation += (
            "This is a critical confusion: over half of nominal "
            "predictions may actually be incident_zone. However, note "
            "that this high error rate is partly because nominal is "
            "the largest class. In absolute terms, nominal is still "
            "correct more often than any other class."
        )
    elif predicted_scene == "mounted_police":
        interpretation += (
            "mounted_police is frequently confused with nominal "
            "because horses at a distance or partial view can be "
            "missed. If you do not see a horse clearly, this is "
            "likely nominal."
        )
    elif predicted_scene == "flagger":
        interpretation += (
            "flagger scenes are often confused with incident_zone "
            "because both involve construction/work zones. The key "
            "distinguishing feature is a human actively directing "
            "traffic."
        )
    elif predicted_scene == "flooded":
        interpretation += (
            "flooded scenes are confused with incident_zone because "
            "both are hazard scenes. The key difference is standing "
            "water vs. emergency vehicles/wreckage."
        )

    result["interpretation"] = interpretation
    return result


def _check_confusion_risk_l3(predicted_scene: str) -> dict[str, Any]:
    result = _check_confusion_risk_l2(predicted_scene)
    pairs = _get_confusion_pairs()
    info = pairs.get(predicted_scene)

    if info is None:
        result["visual_checklist"] = {
            "distinguishing_features": [],
        }
        result["decision_framework"] = (
            "No significant confusion risk. Proceed with your prediction."
        )
        return result

    confused_with = info["confused_with"]
    error_rate = info["error_rate"]

    # Get distinguishing features
    distinguishers = CONFUSION_DISTINGUISHERS.get(predicted_scene, {}).get(
        confused_with, []
    )

    if error_rate > 0.4:
        risk = "high"
    elif error_rate > 0.25:
        risk = "moderate"
    else:
        risk = "low"

    result["classification_risk"] = risk
    result["confidence_required"] = (
        f"This prediction is wrong {error_rate:.0%} of the time. "
        + (
            "Very strong visual evidence is required."
            if risk == "high"
            else (
                "Clear visual evidence is required."
                if risk == "moderate"
                else "Standard visual confirmation is sufficient."
            )
        )
    )
    result["visual_checklist"] = {
        f"features_confirming_{predicted_scene}": SCENE_MUST_PRESENT.get(
            predicted_scene, []
        ),
        f"features_confirming_{confused_with}": SCENE_MUST_PRESENT.get(
            confused_with, []
        ),
        f"distinguishing_{predicted_scene}_from_{confused_with}": (
            distinguishers
        ),
    }
    result["decision_framework"] = (
        f"For each feature in 'features_confirming_{predicted_scene}', "
        f"check if it is present. Then for each feature in "
        f"'features_confirming_{confused_with}', check if it is present. "
        f"Whichever list has MORE confirmed features is the likely "
        f"correct class. Use the 'distinguishing' list to break ties. "
        f"If equal or uncertain, prefer {confused_with} when "
        f"{predicted_scene} has a high error rate."
    )
    return result


def _check_confusion_risk_l4(predicted_scene: str) -> dict[str, Any]:
    result = _check_confusion_risk_l3(predicted_scene)
    pairs = _get_confusion_pairs()
    info = pairs.get(predicted_scene)

    if info is None:
        result["decision_rule"] = (
            f"No confusion risk for {predicted_scene}. Keep your "
            f"prediction as-is."
        )
        result["final_instruction"] = "Confirm your prediction."
        return result

    confused_with = info["confused_with"]

    # Build scene-specific explicit decision tree
    if predicted_scene == "incident_zone":
        decision_rule = (
            "You predicted incident_zone. This class is wrong "
            "41% of the time (actually nominal). Follow these steps:\n"
            "1. Do you see emergency vehicles with FLASHING lights? "
            "If NO -> go to step 2.\n"
            "2. Do you see crashed or damaged vehicles on the road? "
            "If NO -> go to step 3.\n"
            "3. Is the road blocked by wreckage, debris, or an "
            "active emergency? If NO -> go to step 4.\n"
            "4. You answered NO to all 3 questions. Change your "
            "prediction to NOMINAL. Traffic cones, barriers, and "
            "construction zones do not make a scene incident_zone.\n"
            "5. If you answered YES to any question in steps 1-3, "
            "confirm incident_zone."
        )
    elif predicted_scene == "nominal" and confused_with == "incident_zone":
        decision_rule = (
            "You predicted nominal. While nominal is common (78.3%), "
            "50.9% of nominal predictions are wrong (actually "
            "incident_zone). Check:\n"
            "1. Do you see ANY emergency vehicles, crashes, road "
            "closures, or active incident response? If YES -> "
            "change to incident_zone.\n"
            "2. Do you see standing water on the road? If YES -> "
            "change to flooded.\n"
            "3. Do you see a person flagging traffic? If YES -> "
            "change to flagger.\n"
            "4. Do you see a horse on or near the road? If YES -> "
            "change to mounted_police.\n"
            "5. If NO to all, confirm nominal."
        )
    elif predicted_scene == "mounted_police":
        decision_rule = (
            "You predicted mounted_police. This class is wrong "
            "50.7% of the time (actually nominal). Follow these steps:\n"
            "1. Do you clearly see a HORSE? If NO -> change to "
            "nominal immediately.\n"
            "2. Is there a rider on the horse? If NO -> this may "
            "be another animal, change to nominal.\n"
            "3. If YES to both, confirm mounted_police."
        )
    elif predicted_scene == "flagger":
        decision_rule = (
            "You predicted flagger. This class is wrong 34.4% of "
            "the time (actually incident_zone). Follow these steps:\n"
            "1. Do you see a PERSON actively directing traffic? "
            "If NO -> go to step 2.\n"
            "2. Do you see emergency vehicles or a crash scene "
            "instead? If YES -> change to incident_zone.\n"
            "3. Do you see neither a flagger nor an incident? "
            "Change to nominal.\n"
            "4. If you see a person with a sign/flag directing "
            "traffic, confirm flagger."
        )
    elif predicted_scene == "flooded":
        decision_rule = (
            "You predicted flooded. This class is wrong 28.0% of "
            "the time (actually incident_zone). Follow these steps:\n"
            "1. Do you see standing water on the road surface? "
            "If NO -> go to step 2.\n"
            "2. Do you see emergency vehicles or a crash instead? "
            "If YES -> change to incident_zone.\n"
            "3. Is the road just wet from rain (no pooling)? "
            "If YES -> change to nominal.\n"
            "4. If water is clearly pooling on the road, confirm "
            "flooded."
        )
    else:
        decision_rule = (
            f"No specific decision tree for {predicted_scene} vs. "
            f"{confused_with}. Use visual evidence to decide."
        )

    result["decision_rule"] = decision_rule
    result["final_instruction"] = (
        "Apply the decision rule above step by step. Answer each "
        "question YES or NO based on what you see. Follow the "
        "instructions exactly. Output ONLY the final scene "
        "classification."
    )
    return result


# ------------------------------------------------------------------
# check_scene_action_compatibility -- all 4 levels
# ------------------------------------------------------------------
def _check_scene_action_l1(
    scene: str, long_action: str, lat_action: str
) -> dict[str, Any]:
    scene_data, total, typical = _scene_action_stats(scene)
    key = f"{long_action}|{lat_action}"
    count = scene_data.get(key, 0)
    valid_actions = list(scene_data.keys())
    return {
        "scene": scene,
        "long_action": long_action,
        "lat_action": lat_action,
        "compatible": count > 0,
        "count": count,
        "total": total,
        "rate": round(count / total, 3) if total else 0.0,
        "valid_actions": valid_actions,
    }


def _check_scene_action_l2(
    scene: str, long_action: str, lat_action: str
) -> dict[str, Any]:
    result = _check_scene_action_l1(scene, long_action, lat_action)
    _, total, typical = _scene_action_stats(scene)
    result["typical_actions"] = typical

    count = result["count"]

    if count == 0:
        # Never observed
        if total == 0:
            interpretation = (
                f"No co-occurrence data available for scene={scene}. "
                f"Cannot verify action compatibility."
            )
        else:
            # What was the most common action?
            top = typical[0]["actions"] if typical else "unknown"
            interpretation = (
                f"The combination ({long_action}, {lat_action}) has "
                f"NEVER been observed with scene={scene} in the training "
                f"data ({total} samples). This is likely an error. "
                f"The most common action for {scene} is {top}."
            )
    else:
        rate_pct = round(count / total * 100, 1) if total else 0
        if rate_pct < 5:
            interpretation = (
                f"The combination ({long_action}, {lat_action}) with "
                f"{scene} is very rare: only {count}/{total} samples "
                f"({rate_pct}%). Double-check that both the scene and "
                f"action are correct."
            )
        elif rate_pct < 20:
            interpretation = (
                f"The combination ({long_action}, {lat_action}) with "
                f"{scene} is uncommon but observed: {count}/{total} "
                f"samples ({rate_pct}%). This combination can occur but "
                f"is not the typical response."
            )
        else:
            interpretation = (
                f"The combination ({long_action}, {lat_action}) with "
                f"{scene} is well-established: {count}/{total} samples "
                f"({rate_pct}%). This is a common and valid combination."
            )

    result["interpretation"] = interpretation
    return result


def _check_scene_action_l3(
    scene: str, long_action: str, lat_action: str
) -> dict[str, Any]:
    result = _check_scene_action_l2(scene, long_action, lat_action)
    count = result["count"]

    # Diagnostic questions based on scene type
    if scene in ("nominal", "mounted_police"):
        diagnostic = {
            "question_1": (
                f"Is the scene actually {scene}? If yes, the ONLY valid "
                f"action is (null, null)."
            ),
            "question_2": (
                "If you predicted a non-null action, "
                "reconsider whether the scene might actually "
                "be incident_zone, flooded, or flagger."
            ),
        }
    elif count == 0:
        diagnostic = {
            "question_1": (
                f"Is {scene} the correct scene classification? If the "
                f"scene is wrong, the action may need to change too."
            ),
            "question_2": (
                f"If {scene} is correct, is the action ({long_action}, "
                f"{lat_action}) what the image shows? Look at the typical "
                f"actions list for valid alternatives."
            ),
            "question_3": (
                "Could this be an edge case? Is there "
                "something unusual about this scene that "
                "explains a rare action?"
            ),
        }
    else:
        diagnostic = {
            "question_1": (
                f"Does the visual scene match {scene}? Verify the scene "
                f"before evaluating the action."
            ),
            "question_2": (
                f"Given {scene}, does ({long_action}, {lat_action}) match "
                f"what you see? E.g., is the road blocked (stop), "
                f"partially obstructed (slowdown), or clear (proceed)?"
            ),
        }

    # Specific guidance for lane changes
    if lat_action != "null":
        if scene != "incident_zone":
            diagnostic["lane_change_warning"] = (
                f"Lane changes are ONLY valid for incident_zone. You "
                f"predicted {lat_action} with {scene}, which has never "
                f"been observed. Change lat_action to null, or change "
                f"the scene to incident_zone if there is truly an "
                f"incident requiring a lane change."
            )
        else:
            diagnostic["lane_change_check"] = (
                f"Lane change ({lat_action}) is valid for incident_zone. "
                f"Verify: Is the current lane blocked? Is there an "
                f"adjacent lane available to the "
                f"{'left' if lat_action == 'lc_left' else 'right'}?"
            )

    result["diagnostic_questions"] = diagnostic
    result["resolution_guide"] = (
        "Work through the diagnostic questions in order. If any "
        "question suggests the scene or action is wrong, adjust "
        "your prediction accordingly."
    )
    return result


def _check_scene_action_l4(
    scene: str, long_action: str, lat_action: str
) -> dict[str, Any]:
    result = _check_scene_action_l3(scene, long_action, lat_action)

    # Explicit resolution rules
    count = result["count"]
    _, _, typical = _scene_action_stats(scene)

    if scene in ("nominal", "mounted_police"):
        if long_action != "null" or lat_action != "null":
            resolution = (
                f"RULE: Scene is {scene}. The ONLY valid action is "
                f"(null, null). Your prediction ({long_action}, "
                f"{lat_action}) is invalid.\n"
                f"ACTION: Change to long_action=null, lat_action=null.\n"
                f"ALTERNATIVELY: If you believe a non-null action is "
                f"correct, the scene classification must be wrong. "
                f"Re-evaluate the scene."
            )
        else:
            resolution = (
                f"CORRECT: ({long_action}, {lat_action}) is the only "
                f"valid action for {scene}. No changes needed."
            )
    elif count == 0:
        # Build explicit alternatives
        top_action = typical[0]["actions"] if typical else "unknown"
        resolution = (
            f"RULE: ({long_action}, {lat_action}) has NEVER been "
            f"observed with {scene}.\n"
            f"OPTION 1: Change action to the most common for {scene}: "
            f"{top_action}.\n"
            f"OPTION 2: If you are confident in the action, reconsider "
            f"the scene classification.\n"
            f"DEFAULT: Use OPTION 1 -- change to {top_action}."
        )
    elif lat_action != "null" and scene != "incident_zone":
        resolution = (
            f"RULE: Lane changes are ONLY valid for incident_zone. "
            f"Scene is {scene}.\n"
            f"ACTION: Change lat_action to null. Keep long_action as "
            f"{long_action} if it is valid for {scene}."
        )
    else:
        rate_pct = round(count / result["total"] * 100, 1) if result["total"] else 0
        if rate_pct < 5:
            top_action = typical[0]["actions"] if typical else "unknown"
            resolution = (
                f"WARNING: ({long_action}, {lat_action}) with {scene} "
                f"is very rare ({rate_pct}%). Consider changing to the "
                f"most common action: {top_action}.\n"
                f"If the visual evidence strongly supports this specific "
                f"action, you may keep it."
            )
        else:
            resolution = (
                f"VALID: ({long_action}, {lat_action}) is a recognized "
                f"combination for {scene} ({rate_pct}%). No changes "
                f"needed."
            )

    # Add the scene-level action rule
    action_rule = ACTION_DECISION_RULES.get(scene, {})
    result["decision_rule"] = resolution
    result["scene_action_rule"] = action_rule.get(
        "rule",
        f"No specific action rule for {scene}.",
    )
    result["final_instruction"] = (
        "Apply the decision rule. If it says to change the action, "
        "change it. Output the final (scene, long_action, lat_action) "
        "triple."
    )
    return result


# ------------------------------------------------------------------
# check_waypoint_feasibility -- all 4 levels
# ------------------------------------------------------------------
def _classify_waypoint_position(
    x: float,
    y: float,
    action_stats: dict[str, float] | None,
) -> str:
    """Classify where the waypoint falls relative to typical range."""
    if action_stats is None:
        return "unknown"
    x_mean = action_stats.get("x_mean", 0.0)
    x_std = action_stats.get("x_std", 1.0)
    y_mean = action_stats.get("y_mean", 0.0)
    y_std = action_stats.get("y_std", 1.0)

    x_z = (x - x_mean) / x_std if x_std > 0 else 0.0
    y_z = (y - y_mean) / y_std if y_std > 0 else 0.0

    if abs(x_z) <= 2 and abs(y_z) <= 2:
        return "normal"
    if x_z < -2:
        return "far_left"
    if x_z > 2:
        return "far_right"
    if y_z > 2:
        return "far_forward"
    if y_z < -2:
        return "far_behind"
    return "normal"


def _check_waypoint_l1(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
) -> dict[str, Any]:
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
            "has_data": False,
        }

    x_mean = action_stats.get("x_mean", 0.0)
    x_std = action_stats.get("x_std", 1.0)
    y_mean = action_stats.get("y_mean", 0.0)
    y_std = action_stats.get("y_std", 1.0)
    x_z = abs(first_waypoint_x - x_mean) / x_std if x_std > 0 else 0.0
    y_z = abs(first_waypoint_y - y_mean) / y_std if y_std > 0 else 0.0
    feasible = x_z < 3.0 and y_z < 3.0

    return {
        "scene": scene,
        "long_action": long_action,
        "first_waypoint_x": first_waypoint_x,
        "first_waypoint_y": first_waypoint_y,
        "feasible": feasible,
        "has_data": True,
        "typical_center": [round(x_mean, 3), round(y_mean, 3)],
        "typical_bounds": {
            "x_min": round(x_mean - 2 * x_std, 3),
            "x_max": round(x_mean + 2 * x_std, 3),
            "y_min": round(y_mean - 2 * y_std, 3),
            "y_max": round(y_mean + 2 * y_std, 3),
        },
        "x_z_score": round(x_z, 2),
        "y_z_score": round(y_z, 2),
    }


def _check_waypoint_l2(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
) -> dict[str, Any]:
    result = _check_waypoint_l1(
        scene, long_action, first_waypoint_x, first_waypoint_y
    )

    if not result.get("has_data", False):
        result["interpretation"] = (
            f"No waypoint statistics available for {scene}+{long_action}. "
            f"Cannot assess feasibility. The waypoint stats have not been "
            f"computed for this scene-action combination."
        )
        return result

    wp = _get_waypoint_stats()
    action_stats = wp.get(scene, {}).get(long_action)
    position = _classify_waypoint_position(
        first_waypoint_x, first_waypoint_y, action_stats
    )
    feasible = result["feasible"]
    x_z = result["x_z_score"]
    y_z = result["y_z_score"]
    center = result["typical_center"]

    if feasible:
        if position == "normal":
            interpretation = (
                f"Waypoint ({first_waypoint_x:.3f}, {first_waypoint_y:.3f}) "
                f"is within the typical range for {scene}+{long_action}. "
                f"Typical center is ({center[0]}, {center[1]}). "
                f"No spatial concerns."
            )
        else:
            _dir_map = {
                "far_left": "to the left",
                "far_right": "to the right",
                "far_forward": "forward",
                "far_behind": "back",
            }
            shift_dir = _dir_map.get(position, "off-center")
            interpretation = (
                f"Waypoint ({first_waypoint_x:.3f}, "
                f"{first_waypoint_y:.3f}) is within "
                f"acceptable range but shifted "
                f"{shift_dir}. "
                f"Typical center is "
                f"({center[0]}, {center[1]}). "
                f"This may be valid if the road curves "
                f"or a lane change is appropriate."
            )
    else:
        parts = []
        if x_z >= 3.0:
            direction = (
                "left" if first_waypoint_x < center[0] else "right"
            )
            parts.append(
                f"x-delta is {x_z:.1f} std devs to the {direction} "
                f"of typical (center={center[0]})"
            )
        if y_z >= 3.0:
            direction = (
                "further forward"
                if first_waypoint_y > center[1]
                else "further back"
            )
            parts.append(
                f"y-delta is {y_z:.1f} std devs {direction} "
                f"than typical (center={center[1]})"
            )
        interpretation = (
            f"Waypoint ({first_waypoint_x:.3f}, {first_waypoint_y:.3f}) "
            f"is OUTSIDE the typical range for {scene}+{long_action}. "
            + ". ".join(parts)
            + ". This suggests a potential error in either "
            "the waypoint, the scene classification, or "
            "the action prediction."
        )

    result["interpretation"] = interpretation
    return result


def _check_waypoint_l3(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
) -> dict[str, Any]:
    result = _check_waypoint_l2(
        scene, long_action, first_waypoint_x, first_waypoint_y
    )

    if not result.get("has_data", False):
        result["spatial_reasoning_guide"] = (
            "No data available for guided reasoning. Use general "
            "driving heuristics: waypoints should follow the road "
            "geometry visible in the image."
        )
        return result

    wp = _get_waypoint_stats()
    action_stats = wp.get(scene, {}).get(long_action)
    position = _classify_waypoint_position(
        first_waypoint_x, first_waypoint_y, action_stats
    )

    # Build spatial reasoning guide
    guidance = WAYPOINT_SPATIAL_GUIDANCE.get(position, "")

    reasoning_checklist = {
        "road_geometry": (
            "Look at the road in the image. Is it straight, curving left, "
            "or curving right? The waypoint should follow the road curvature."
        ),
        "lane_position": (
            "Is the vehicle centered in its lane? A waypoint shifted left "
            "or right could indicate a lane change or lane drift."
        ),
        "action_consistency": (
            f"You predicted {long_action}. Does the waypoint magnitude "
            f"match? stop = minimal forward motion, slowdown = moderate "
            f"deceleration, proceed = maintaining/increasing speed."
        ),
    }

    if position in ("far_left", "far_right"):
        reasoning_checklist["lateral_shift"] = (
            f"The waypoint is shifted {'left' if position == 'far_left' else 'right'}. "
            f"Valid reasons: road curves in that direction, or a lane "
            f"change is appropriate. Invalid: straight road with no "
            f"obstruction requiring a lane change."
        )

    result["spatial_reasoning_guide"] = guidance
    result["reasoning_checklist"] = reasoning_checklist
    return result


def _check_waypoint_l4(
    scene: str,
    long_action: str,
    first_waypoint_x: float,
    first_waypoint_y: float,
) -> dict[str, Any]:
    result = _check_waypoint_l3(
        scene, long_action, first_waypoint_x, first_waypoint_y
    )

    if not result.get("has_data", False):
        result["correction_rule"] = (
            f"No waypoint statistics for {scene}+{long_action}. "
            f"Use these defaults based on action type:\n"
            f"- null: keep waypoint near (0, 0)\n"
            f"- stop: small forward motion, near center laterally\n"
            f"- slowdown: moderate forward motion, near center laterally\n"
            f"- proceed: larger forward motion, near center laterally\n"
            f"For lane changes (incident_zone only): shift x by ~0.2 in "
            f"the lane change direction."
        )
        result["final_instruction"] = (
            "Apply the correction rule if the waypoint seems implausible. "
            "Output the final waypoint values."
        )
        return result

    feasible = result["feasible"]
    center = result["typical_center"]
    bounds = result["typical_bounds"]

    if feasible:
        correction_rule = (
            f"VALID: Waypoint ({first_waypoint_x:.3f}, "
            f"{first_waypoint_y:.3f}) is within acceptable range. "
            f"No correction needed."
        )
    else:
        # Build explicit correction
        corrected_x = first_waypoint_x
        corrected_y = first_waypoint_y

        if first_waypoint_x < bounds["x_min"]:
            corrected_x = bounds["x_min"]
        elif first_waypoint_x > bounds["x_max"]:
            corrected_x = bounds["x_max"]

        if first_waypoint_y < bounds["y_min"]:
            corrected_y = bounds["y_min"]
        elif first_waypoint_y > bounds["y_max"]:
            corrected_y = bounds["y_max"]

        correction_rule = (
            f"OUT OF RANGE: Waypoint ({first_waypoint_x:.3f}, "
            f"{first_waypoint_y:.3f}) is outside the typical range.\n"
            f"STEP 1: Check if the road curves to explain the lateral "
            f"shift. If yes, the waypoint may be correct.\n"
            f"STEP 2: Check if the action matches the scene severity. "
            f"If the waypoint implies more/less braking than the action "
            f"suggests, adjust the action.\n"
            f"STEP 3: If the waypoint is simply wrong, move it to the "
            f"typical region: ({corrected_x:.3f}, {corrected_y:.3f}). "
            f"Typical center is ({center[0]}, {center[1]}), typical "
            f"bounds are x=[{bounds['x_min']}, {bounds['x_max']}], "
            f"y=[{bounds['y_min']}, {bounds['y_max']}]."
        )

    result["correction_rule"] = correction_rule
    result["final_instruction"] = (
        "Follow the correction rule. If the waypoint needs adjustment, "
        "output the corrected values. Otherwise, keep the original."
    )
    return result


# ------------------------------------------------------------------
# Level dispatcher maps
# ------------------------------------------------------------------
_LEVEL_DISPATCH: dict[str, dict[int, Any]] = {
    "check_scene_prior": {
        1: _check_scene_prior_l1,
        2: _check_scene_prior_l2,
        3: _check_scene_prior_l3,
        4: _check_scene_prior_l4,
    },
    "check_confusion_risk": {
        1: _check_confusion_risk_l1,
        2: _check_confusion_risk_l2,
        3: _check_confusion_risk_l3,
        4: _check_confusion_risk_l4,
    },
    "check_scene_action_compatibility": {
        1: _check_scene_action_l1,
        2: _check_scene_action_l2,
        3: _check_scene_action_l3,
        4: _check_scene_action_l4,
    },
    "check_waypoint_feasibility": {
        1: _check_waypoint_l1,
        2: _check_waypoint_l2,
        3: _check_waypoint_l3,
        4: _check_waypoint_l4,
    },
}


# ------------------------------------------------------------------
# Main dispatcher
# ------------------------------------------------------------------
def execute_tool_v2(
    tool_name: str,
    arguments: dict[str, Any],
    level: int = 1,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a tool at the specified scaffolding level.

    Parameters
    ----------
    tool_name:
        One of the four tool function names.
    arguments:
        Keyword arguments for the tool function.
    level:
        Scaffolding level (1-4). Default is 1 (raw data).
    ground_truth:
        If provided, augment the result with oracle information.

    Returns
    -------
    dict with tool result (JSON-serializable).
    """
    if level not in (1, 2, 3, 4):
        return {
            "error": f"Invalid level: {level}. Must be 1, 2, 3, or 4."
        }

    tool_dispatch = _LEVEL_DISPATCH.get(tool_name)
    if tool_dispatch is None:
        return {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(_LEVEL_DISPATCH.keys()),
        }

    fn = tool_dispatch.get(level)
    if fn is None:
        return {
            "error": (
                f"Level {level} not implemented for {tool_name}."
            )
        }

    result = fn(**arguments)

    # Add oracle info if ground truth provided
    if ground_truth is not None:
        result = _augment_with_oracle(
            tool_name, result, arguments, ground_truth
        )

    # Tag the level
    result["_level"] = level

    return result


def _augment_with_oracle(
    tool_name: str,
    result: dict[str, Any],
    arguments: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Add oracle (ground truth) information to a tool result."""
    gt_scene = ground_truth.get("scene_type_gt", "unknown")
    gt_long = ground_truth.get("long_action_gt", "unknown")
    gt_lat = ground_truth.get("lat_action_gt", "unknown")

    result["oracle"] = True
    result["ground_truth_scene"] = gt_scene

    if tool_name in ("check_scene_prior", "check_confusion_risk"):
        predicted = arguments.get("predicted_scene", "")
        result["prediction_is_correct"] = predicted == gt_scene
        if predicted != gt_scene:
            result["oracle_warning"] = (
                f"ORACLE: The correct scene is {gt_scene}, "
                f"not {predicted}."
            )

    elif tool_name == "check_scene_action_compatibility":
        result["ground_truth_long_action"] = gt_long
        result["ground_truth_lat_action"] = gt_lat
        scene_ok = arguments.get("scene") == gt_scene
        long_ok = arguments.get("long_action") == gt_long
        lat_ok = arguments.get("lat_action") == gt_lat
        if not (scene_ok and long_ok and lat_ok):
            parts = []
            if not scene_ok:
                parts.append(f"scene should be {gt_scene}")
            if not long_ok:
                parts.append(f"long_action should be {gt_long}")
            if not lat_ok:
                parts.append(f"lat_action should be {gt_lat}")
            result["oracle_warning"] = (
                "ORACLE: " + ", ".join(parts) + "."
            )

    elif tool_name == "check_waypoint_feasibility":
        result["ground_truth_long_action"] = gt_long
        if arguments.get("scene") != gt_scene:
            result["oracle_warning"] = (
                f"ORACLE: Scene should be {gt_scene}, not "
                f"{arguments.get('scene')}. Re-evaluate waypoints."
            )

    return result


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------
def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible dispatcher. Delegates to level=1."""
    return execute_tool_v2(
        tool_name, arguments, level=1, ground_truth=ground_truth
    )


# ------------------------------------------------------------------
# Convenience: get tools for a condition (same as tools.py)
# ------------------------------------------------------------------
def get_tools_for_condition(
    condition_name: str,
) -> list[dict[str, Any]]:
    """Return tool definitions for *condition_name*."""
    mapping: dict[str, list[dict[str, Any]]] = {
        "prior_only": [TOOL_PRIOR_CHECK],
        "confusion_only": [TOOL_CONFUSION_CHECK],
        "all_tools": ALL_TOOLS,
        "oracle": ALL_TOOLS,
        "staged": ALL_TOOLS,
    }
    return mapping.get(condition_name, ALL_TOOLS)
