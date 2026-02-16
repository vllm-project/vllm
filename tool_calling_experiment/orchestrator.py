#!/usr/bin/env python3
"""ToolCallingOrchestrator -- reusable engine for tool-calling conversations.

Manages the multi-turn conversation loop between a vLLM model (via
OpenAI-compatible HTTP API) and visual/statistical tools.

Handles:
- Sending messages with images to vLLM via OpenAI-compatible API
- Parsing tool calls from model output (hermes parser, structured response)
- Executing tools and injecting results (including images) back into
  conversation
- Collecting the final prediction
- Logging every intermediate step for analysis

Usage:
    from orchestrator import ToolCallingOrchestrator

    orch = ToolCallingOrchestrator(
        server_url="http://localhost:8300",
        tools={"zoom_region": zoom_fn, "check_scene_prior": prior_fn},
        tool_definitions=[TOOL_ZOOM_DEF, TOOL_PRIOR_DEF],
        max_tool_rounds=5,
    )

    result = orch.run(
        image_path="/path/to/image.jpg",
        system_prompt="You are a driving scene analyst.",
        user_prompt="Analyze this driving scene.",
    )
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
from collections.abc import Callable
from typing import Any

import requests  # type: ignore[import-not-found]

# ====================================================================
# Helper functions
# ====================================================================

def image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    lower = image_path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    # Default to JPEG
    return "image/jpeg"


# ------------------------------------------------------------------
# Prediction parsing
# ------------------------------------------------------------------

_SCENE_PATTERN = re.compile(r"FINAL_SCENE:\s*(\S+)", re.IGNORECASE)
_LONG_ACTION_PATTERN = re.compile(
    r"FINAL_LONG_ACTION:\s*(\S+)", re.IGNORECASE
)
_LAT_ACTION_PATTERN = re.compile(
    r"FINAL_LAT_ACTION:\s*(\S+)", re.IGNORECASE
)
_WAYPOINT_PATTERN = re.compile(
    r"FINAL_WAYPOINT:\s*\[?\s*(-?[\d.]+)\s*[,\s]+\s*(-?[\d.]+)\s*\]?",
    re.IGNORECASE,
)

VALID_SCENES = frozenset(
    ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]
)
VALID_LONG_ACTIONS = frozenset(["stop", "slowdown", "proceed", "null"])
VALID_LAT_ACTIONS = frozenset(["lc_left", "lc_right", "null"])


def _clean_value(value: str) -> str:
    """Strip punctuation and whitespace from a parsed value."""
    return value.strip().strip(".,;:\"'`").lower()


def _try_extract_scene_freeform(text: str) -> str | None:
    """Attempt to find a scene type from freeform text."""
    lower = text.lower()
    found = []
    for scene in VALID_SCENES:
        if scene in lower:
            found.append(scene)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        non_nominal = [s for s in found if s != "nominal"]
        if len(non_nominal) == 1:
            return non_nominal[0]
    return None


def _try_extract_json_prediction(text: str) -> dict[str, Any] | None:
    """Try to extract a prediction from a JSON block in text."""
    # Look for JSON objects
    json_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    for m in json_pattern.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("scene" in obj or "final_scene" in obj):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_prediction(text: str) -> dict[str, Any]:
    """Parse scene/action/waypoint from model text.

    Supports multiple output formats:
    - Structured: FINAL_SCENE: X, FINAL_LONG_ACTION: Y, etc.
    - Freeform: look for scene type names in text
    - JSON: {"scene": "X", "action": "Y"}

    Returns:
        {
            "scene": str or None,
            "long_action": str or None,
            "lat_action": str or None,
            "waypoint": [row, col] or None,
        }
    """
    result: dict[str, Any] = {
        "scene": None,
        "long_action": None,
        "lat_action": None,
        "waypoint": None,
    }
    if not text:
        return result

    # --- Strategy 1: Structured FINAL_XXX: tags ---
    m = _SCENE_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_SCENES:
            result["scene"] = val

    m = _LONG_ACTION_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_LONG_ACTIONS:
            result["long_action"] = val

    m = _LAT_ACTION_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_LAT_ACTIONS:
            result["lat_action"] = val

    m = _WAYPOINT_PATTERN.search(text)
    if m:
        with contextlib.suppress(ValueError, IndexError):
            result["waypoint"] = [float(m.group(1)), float(m.group(2))]

    # --- Strategy 2: JSON ---
    if result["scene"] is None:
        json_pred = _try_extract_json_prediction(text)
        if json_pred:
            for key in ("scene", "final_scene"):
                val = json_pred.get(key)
                if val and _clean_value(str(val)) in VALID_SCENES:
                    result["scene"] = _clean_value(str(val))
                    break
            for key in ("long_action", "final_long_action", "longitudinal_action"):
                val = json_pred.get(key)
                if val and _clean_value(str(val)) in VALID_LONG_ACTIONS:
                    result["long_action"] = _clean_value(str(val))
                    break
            for key in ("lat_action", "final_lat_action", "lateral_action"):
                val = json_pred.get(key)
                if val and _clean_value(str(val)) in VALID_LAT_ACTIONS:
                    result["lat_action"] = _clean_value(str(val))
                    break

    # --- Strategy 3: Freeform scene extraction ---
    if result["scene"] is None:
        result["scene"] = _try_extract_scene_freeform(text)

    return result


# ====================================================================
# ToolCallingOrchestrator
# ====================================================================

class ToolCallingOrchestrator:
    """Manages the multi-turn conversation loop between a vLLM model
    and visual/statistical tools.

    Handles:
    - Sending messages with images to vLLM via OpenAI-compatible API
    - Parsing tool calls from model output (via hermes parser)
    - Executing tools and injecting results (including images) back
      into the conversation
    - Collecting the final prediction
    - Logging every intermediate step for analysis
    """

    # Tools whose execution may need the current image path injected
    IMAGE_AWARE_TOOLS = frozenset([
        "zoom_region",
        "visualize_waypoint",
        "analyze_road_geometry",
        "find_similar_scenes",
    ])

    def __init__(
        self,
        server_url: str,
        tools: dict[str, Callable[..., Any]],
        tool_definitions: list[dict[str, Any]],
        max_tool_rounds: int = 5,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> None:
        """
        Args:
            server_url: Base URL of vLLM server (e.g. "http://localhost:8300")
            tools: Dict mapping tool name -> callable function
            tool_definitions: List of OpenAI-format tool definitions
            max_tool_rounds: Max tool call rounds before forcing final answer
            temperature: Sampling temperature
            max_tokens: Max tokens per generation
        """
        self.server_url = server_url.rstrip("/")
        self.tools = tools
        self.tool_definitions = tool_definitions
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model_name: str | None = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(
        self,
        image_path: str | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        tool_choice: str = "auto",
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Run a complete tool-calling session.

        Args:
            image_path: Path to the driving scene image (None for text-only)
            system_prompt: System message
            user_prompt: User message (may contain {placeholders} already filled)
            tool_choice: "auto", "required", or "none"
            model_name: Model name for API (defaults to server's model)

        Returns:
            Dict with keys:
                final_text, final_prediction, num_rounds, num_tool_calls,
                tool_calls, full_conversation, reasoning_text, changed_mind,
                initial_assessment, latency_ms, generation_ms,
                tool_execution_ms, error
        """
        start_time = time.monotonic()
        generation_ms = 0.0
        tool_execution_ms = 0.0

        # Resolve model name
        effective_model = model_name or self._model_name or self._discover_model()

        # Track state
        tool_call_log: list[dict[str, Any]] = []
        all_model_texts: list[str] = []
        initial_assessment: str = ""
        current_image_path = image_path

        # ----------------------------------------------------------
        # Build initial conversation
        # ----------------------------------------------------------
        conversation: list[dict[str, Any]] = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        # User message -- optionally with image
        if image_path:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(image_path),
                    },
                },
            ]
            conversation.append({"role": "user", "content": user_content})
        else:
            conversation.append({"role": "user", "content": user_prompt})

        # ----------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------
        rounds = 0
        final_text = ""
        error_msg: str | None = None
        active_tool_choice = tool_choice

        while rounds < self.max_tool_rounds:
            rounds += 1

            # --- Call model ---
            gen_start = time.monotonic()
            try:
                response = self._call_model(
                    conversation,
                    model_name=effective_model,
                    tool_choice=active_tool_choice,
                )
            except Exception as exc:
                error_msg = f"Model call failed in round {rounds}: {exc}"
                break
            gen_end = time.monotonic()
            generation_ms += (gen_end - gen_start) * 1000

            # Extract content and tool calls
            content = response.get("content") or ""
            tool_calls = response.get("tool_calls")

            # Track model text
            if content:
                all_model_texts.append(content)
                if not initial_assessment:
                    initial_assessment = content

            if tool_calls:
                # --- Model wants to use tools ---
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                conversation.append(assistant_msg)

                for tc in tool_calls:
                    tc_name = tc["function"]["name"]

                    # Parse arguments
                    try:
                        tc_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}

                    # Inject image_path for image-aware tools
                    if tc_name in self.IMAGE_AWARE_TOOLS and current_image_path:
                        tc_args.setdefault("image_path", current_image_path)

                    # --- Execute tool ---
                    tool_start = time.monotonic()
                    tool_result, tool_error = self._execute_tool(tc_name, tc_args)
                    tool_end = time.monotonic()
                    tool_execution_ms += (tool_end - tool_start) * 1000

                    # Determine if the tool result contains an image
                    result_image_path = self._extract_image_from_result(tool_result)
                    result_has_image = result_image_path is not None

                    # Build metadata (non-image parts)
                    result_metadata = self._strip_image_from_result(tool_result)
                    if tool_error:
                        result_metadata = {"error": tool_error}

                    # Log the tool call
                    tool_call_log.append({
                        "round": rounds,
                        "tool_name": tc_name,
                        "arguments": tc_args,
                        "result_metadata": result_metadata,
                        "result_has_image": result_has_image,
                        "result_image_path": result_image_path,
                    })

                    # Build tool response message
                    tool_msg = self._build_tool_response(
                        tool_call_id=tc["id"],
                        metadata=result_metadata,
                        image_path=result_image_path,
                    )
                    conversation.append(tool_msg)

                # After first round, switch to auto to avoid forcing tools forever
                if active_tool_choice == "required":
                    active_tool_choice = "auto"

                continue

            else:
                # --- Model gave final answer (no tool calls) ---
                final_text = content
                # Also check if response is completely empty
                if not content and not tool_calls:
                    # Model generated nothing -- break to avoid infinite loop
                    pass
                break
        else:
            # Hit max_tool_rounds -- force a final answer without tools
            conversation.append({
                "role": "user",
                "content": (
                    "You have used all available tool call rounds. "
                    "Please give your final prediction now.\n\n"
                    "Output format:\n"
                    "FINAL_SCENE: <scene_type>\n"
                    "FINAL_LONG_ACTION: <action>\n"
                    "FINAL_LAT_ACTION: <action>"
                ),
            })
            gen_start = time.monotonic()
            try:
                response = self._call_model(
                    conversation,
                    model_name=effective_model,
                    tool_choice="none",
                )
                final_text = response.get("content") or ""
                if final_text:
                    all_model_texts.append(final_text)
            except Exception as exc:
                error_msg = f"Final answer call failed: {exc}"
                final_text = ""
            gen_end = time.monotonic()
            generation_ms += (gen_end - gen_start) * 1000

        # ----------------------------------------------------------
        # Parse final prediction
        # ----------------------------------------------------------
        final_prediction = parse_prediction(final_text)

        # Determine if the model changed its mind (initial vs final prediction)
        initial_prediction = (
            parse_prediction(initial_assessment)
            if initial_assessment
            else {}
        )
        changed_mind = (
            initial_prediction.get("scene") is not None
            and final_prediction.get("scene") is not None
            and initial_prediction["scene"] != final_prediction["scene"]
        )

        # Concatenate all reasoning text
        reasoning_text = "\n---\n".join(all_model_texts)

        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000

        return {
            "final_text": final_text,
            "final_prediction": final_prediction,
            "num_rounds": rounds,
            "num_tool_calls": len(tool_call_log),
            "tool_calls": tool_call_log,
            "full_conversation": conversation,
            "reasoning_text": reasoning_text,
            "changed_mind": changed_mind,
            "initial_assessment": initial_assessment,
            "latency_ms": round(latency_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "tool_execution_ms": round(tool_execution_ms, 1),
            "error": error_msg,
        }

    # ----------------------------------------------------------
    # Private: model communication
    # ----------------------------------------------------------

    def _discover_model(self) -> str:
        """Query the /v1/models endpoint to discover the served model name."""
        try:
            resp = requests.get(
                f"{self.server_url}/v1/models", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                self._model_name = models[0]["id"]
                return self._model_name
        except Exception:
            pass
        # Fallback: return a generic name (the server usually accepts anything)
        return "default"

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        model_name: str = "default",
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Make a single /v1/chat/completions request.

        Returns the ``message`` object from the first choice.
        """
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Only include tools if we have definitions and tool_choice is not "none"
        if self.tool_definitions and tool_choice != "none":
            payload["tools"] = self.tool_definitions
            payload["tool_choice"] = tool_choice

        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    # ----------------------------------------------------------
    # Private: tool execution
    # ----------------------------------------------------------

    def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """Execute a tool by name with the given arguments.

        Returns:
            (result_dict, error_string_or_none)
        """
        fn = self.tools.get(tool_name)
        if fn is None:
            available = list(self.tools.keys())
            return (
                {},
                f"Unknown tool: {tool_name}. "
                f"Available: {available}",
            )

        try:
            result = fn(**arguments)
            if not isinstance(result, dict):
                result = {"result": result}
            return result, None
        except Exception as exc:
            return {}, f"Tool execution failed for {tool_name}: {exc}"

    # ----------------------------------------------------------
    # Private: image handling
    # ----------------------------------------------------------

    @staticmethod
    def _image_data_url(path: str) -> str:
        """Build a data: URL from an image file path."""
        media = _image_media_type(path)
        b64 = image_to_base64(path)
        return f"data:{media};base64,{b64}"

    @staticmethod
    def _extract_image_from_result(result: dict[str, Any]) -> str | None:
        """Check if a tool result contains an image path.

        Tools that produce images should return a dict with one of:
          - "image_path": str
          - "zoomed_image": str  (for zoom_region)
          - "visualization_path": str
        """
        if not isinstance(result, dict):
            return None
        for key in ("image_path", "zoomed_image", "visualization_path",
                     "result_image_path", "output_image", "annotated_image"):
            val = result.get(key)
            if val and isinstance(val, str):
                return val
        return None

    @staticmethod
    def _strip_image_from_result(result: dict[str, Any]) -> dict[str, Any]:
        """Return the result dict without image data (for logging/metadata)."""
        if not isinstance(result, dict):
            return {}
        image_keys = {
            "image_path", "zoomed_image", "visualization_path",
            "result_image_path", "output_image", "annotated_image",
            "image_base64",
        }
        return {k: v for k, v in result.items() if k not in image_keys}

    # ----------------------------------------------------------
    # Private: message building
    # ----------------------------------------------------------

    def _build_tool_response(
        self,
        tool_call_id: str,
        metadata: dict[str, Any],
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """Build a tool response message, optionally with an image.

        vLLM's OpenAI-compatible API may not support multimodal content
        inside tool role messages. We use a fallback strategy:

        Strategy A (preferred): Multimodal tool response with text + image.
        Strategy B (fallback): Tool response with text only. If an image is
            present, it would need a separate user message. However, inserting
            a user message mid-tool-response breaks the expected
            assistant -> tool -> assistant flow.

        We try Strategy A first. If the server rejects it (unlikely with
        recent vLLM), the caller can handle the error.
        """
        text_content = json.dumps(metadata, default=str)

        if image_path:
            # Strategy A: multimodal tool response
            try:
                data_url = self._image_data_url(image_path)
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            except (FileNotFoundError, OSError):
                # Image file not readable -- fall back to text-only
                metadata["_image_error"] = f"Could not read image: {image_path}"
                text_content = json.dumps(metadata, default=str)

        # Text-only tool response
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text_content,
        }


# ====================================================================
# Smoke test
# ====================================================================

def _smoke_test() -> None:
    """Quick smoke test with mock tools -- no live server needed."""
    print("Running orchestrator smoke test...")

    # Mock tools
    call_log: list[str] = []

    def mock_zoom(
        image_path: str = "test.jpg",
        center_x: float = 0.5,
        center_y: float = 0.5,
        crop_size: int = 128,
    ) -> dict[str, Any]:
        call_log.append("zoom_region")
        return {
            "zoomed_image": image_path,
            "original_region": [0, 0, 128, 128],
            "grid_region": "center",
        }

    def mock_check_scene_prior(
        predicted_scene: str = "nominal",
    ) -> dict[str, Any]:
        call_log.append("check_scene_prior")
        return {
            "predicted_scene": predicted_scene,
            "base_rate": 0.78,
            "most_common_scene": "nominal",
            "most_common_rate": 0.78,
            "interpretation": f"{predicted_scene} base rate is 0.78",
        }

    def mock_check_confusion_risk(
        predicted_scene: str = "nominal",
    ) -> dict[str, Any]:
        call_log.append("check_confusion_risk")
        return {
            "predicted_scene": predicted_scene,
            "has_confusion_risk": True,
            "confused_with": "incident_zone",
            "error_rate": 0.41,
        }

    # Tool definitions (minimal)
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "zoom_region",
                "description": "Zoom into a region of the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "center_x": {"type": "number"},
                        "center_y": {"type": "number"},
                        "crop_size": {"type": "integer"},
                    },
                    "required": ["center_x", "center_y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_scene_prior",
                "description": "Check scene base rate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicted_scene": {
                            "type": "string",
                            "enum": ["nominal", "incident_zone", "flooded",
                                     "flagger", "mounted_police"],
                        },
                    },
                    "required": ["predicted_scene"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_confusion_risk",
                "description": "Check confusion risk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicted_scene": {
                            "type": "string",
                            "enum": ["nominal", "incident_zone", "flooded",
                                     "flagger", "mounted_police"],
                        },
                    },
                    "required": ["predicted_scene"],
                },
            },
        },
    ]

    orch = ToolCallingOrchestrator(
        server_url="http://localhost:9999",  # dummy, not actually called
        tools={
            "zoom_region": mock_zoom,
            "check_scene_prior": mock_check_scene_prior,
            "check_confusion_risk": mock_check_confusion_risk,
        },
        tool_definitions=tool_defs,
        max_tool_rounds=3,
        temperature=0,
        max_tokens=512,
    )

    # ---- Test 1: Construction and attributes ----
    assert orch.server_url == "http://localhost:9999"
    assert orch.max_tool_rounds == 3
    assert orch.temperature == 0
    assert orch.max_tokens == 512
    assert len(orch.tools) == 3
    assert len(orch.tool_definitions) == 3
    print("  [PASS] Construction and attributes")

    # ---- Test 2: parse_prediction structured ----
    text = (
        "Based on my analysis:\n"
        "FINAL_SCENE: nominal\n"
        "FINAL_LONG_ACTION: null\n"
        "FINAL_LAT_ACTION: null"
    )
    pred = parse_prediction(text)
    assert pred["scene"] == "nominal", f"Expected nominal, got {pred['scene']}"
    assert pred["long_action"] == "null"
    assert pred["lat_action"] == "null"
    print("  [PASS] parse_prediction (structured)")

    # ---- Test 3: parse_prediction freeform ----
    text2 = "The scene appears to be a flooded road with standing water."
    pred2 = parse_prediction(text2)
    assert pred2["scene"] == "flooded", f"Expected flooded, got {pred2['scene']}"
    print("  [PASS] parse_prediction (freeform)")

    # ---- Test 4: parse_prediction JSON ----
    text3 = 'I conclude: {"scene": "incident_zone", "long_action": "stop"}'
    pred3 = parse_prediction(text3)
    assert pred3["scene"] == "incident_zone"
    assert pred3["long_action"] == "stop"
    print("  [PASS] parse_prediction (JSON)")

    # ---- Test 5: parse_prediction waypoint ----
    text4 = (
        "FINAL_SCENE: nominal\n"
        "FINAL_LONG_ACTION: null\n"
        "FINAL_LAT_ACTION: null\n"
        "FINAL_WAYPOINT: [0.15, -0.03]"
    )
    pred4 = parse_prediction(text4)
    assert pred4["waypoint"] is not None
    assert abs(pred4["waypoint"][0] - 0.15) < 1e-6
    assert abs(pred4["waypoint"][1] - (-0.03)) < 1e-6
    print("  [PASS] parse_prediction (waypoint)")

    # ---- Test 6: parse_prediction empty ----
    pred5 = parse_prediction("")
    assert pred5["scene"] is None
    assert pred5["long_action"] is None
    print("  [PASS] parse_prediction (empty)")

    # ---- Test 7: Tool execution (mock) ----
    result, err = orch._execute_tool(
        "check_scene_prior", {"predicted_scene": "nominal"}
    )
    assert err is None
    assert result["base_rate"] == 0.78
    print("  [PASS] Tool execution (mock)")

    # ---- Test 8: Unknown tool ----
    result2, err2 = orch._execute_tool("nonexistent_tool", {})
    assert err2 is not None
    assert "Unknown tool" in err2
    print("  [PASS] Unknown tool error handling")

    # ---- Test 9: Tool execution error handling ----
    def bad_tool(**kwargs: Any) -> dict[str, Any]:
        raise ValueError("Intentional test error")

    orch.tools["bad_tool"] = bad_tool
    result3, err3 = orch._execute_tool("bad_tool", {})
    assert err3 is not None
    assert "Intentional test error" in err3
    print("  [PASS] Tool execution error handling")

    # ---- Test 10: _strip_image_from_result ----
    tool_result_with_image = {
        "grid_region": "center",
        "zoomed_image": "/tmp/zoomed.jpg",
        "original_region": [0, 0, 128, 128],
    }
    stripped = ToolCallingOrchestrator._strip_image_from_result(tool_result_with_image)
    assert "zoomed_image" not in stripped
    assert "grid_region" in stripped
    assert "original_region" in stripped
    print("  [PASS] _strip_image_from_result")

    # ---- Test 11: _extract_image_from_result ----
    img = ToolCallingOrchestrator._extract_image_from_result(tool_result_with_image)
    assert img == "/tmp/zoomed.jpg"
    no_img = ToolCallingOrchestrator._extract_image_from_result({"data": "text only"})
    assert no_img is None
    print("  [PASS] _extract_image_from_result")

    # ---- Test 12: _build_tool_response text-only ----
    msg = orch._build_tool_response(
        tool_call_id="call_123",
        metadata={"base_rate": 0.78, "scene": "nominal"},
        image_path=None,
    )
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_123"
    assert isinstance(msg["content"], str)
    assert "0.78" in msg["content"]
    print("  [PASS] _build_tool_response (text-only)")

    # ---- Test 13: image_to_base64 round-trip ----
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"\xff\xd8\xff\xe0test_image_data")
        tmp_path = tmp.name
    try:
        b64 = image_to_base64(tmp_path)
        decoded = base64.b64decode(b64)
        assert decoded == b"\xff\xd8\xff\xe0test_image_data"
        print("  [PASS] image_to_base64 round-trip")

        # ---- Test 14: _build_tool_response with image ----
        msg2 = orch._build_tool_response(
            tool_call_id="call_456",
            metadata={"grid_region": "center"},
            image_path=tmp_path,
        )
        assert msg2["role"] == "tool"
        assert isinstance(msg2["content"], list)
        assert len(msg2["content"]) == 2
        assert msg2["content"][0]["type"] == "text"
        assert msg2["content"][1]["type"] == "image_url"
        assert "base64" in msg2["content"][1]["image_url"]["url"]
        print("  [PASS] _build_tool_response (with image)")
    finally:
        os.unlink(tmp_path)

    # ---- Test 15: _image_media_type ----
    assert _image_media_type("photo.jpg") == "image/jpeg"
    assert _image_media_type("PHOTO.JPEG") == "image/jpeg"
    assert _image_media_type("screenshot.png") == "image/png"
    assert _image_media_type("anim.gif") == "image/gif"
    assert _image_media_type("photo.webp") == "image/webp"
    print("  [PASS] _image_media_type")

    # ---- Test 16: Verify run() return schema (mock, no server) ----
    # We cannot call run() without a server, but we can verify the
    # expected return keys are documented in the method
    import inspect
    sig = inspect.signature(orch.run)
    assert "image_path" in sig.parameters
    assert "system_prompt" in sig.parameters
    assert "user_prompt" in sig.parameters
    assert "tool_choice" in sig.parameters
    assert "model_name" in sig.parameters
    print("  [PASS] run() signature verification")

    print()
    print("All 16 tests passed. Orchestrator is ready for use.")
    print(f"  Tools registered: {list(orch.tools.keys())}")
    print(f"  Tool calls logged in smoke test: {call_log}")


if __name__ == "__main__":
    _smoke_test()
