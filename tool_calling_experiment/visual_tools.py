#!/usr/bin/env python3
"""Visual tools for SceneIQ tool-calling experiment.

Provides image manipulation tools that take driving scene images and
return modified images + metadata. These complement the statistical
tools in tools_v2.py.

Tools:
    1. zoom_region -- crop and enlarge a region for closer inspection
    2. visualize_waypoint -- draw a predicted waypoint marker on the image
    3. analyze_road_geometry -- classical CV for road structure extraction
    4. find_similar_scenes -- CLIP embeddings + FAISS for kNN retrieval

Utilities:
    - load_sample_image -- load an image from the MDS dataset to a temp file
    - load_sample_metadata -- load metadata for a sample from MDS
    - image_to_base64 -- encode an image file as base64 for API injection
    - build_scene_index -- pre-compute FAISS index (run once)

Usage:
    from visual_tools import (
        zoom_region, visualize_waypoint,
        analyze_road_geometry, find_similar_scenes,
        load_sample_image, load_sample_metadata, image_to_base64,
        build_scene_index,
        TOOL_ZOOM, TOOL_WAYPOINT_VIZ,
        TOOL_ROAD_GEOMETRY, TOOL_SIMILAR_SCENES,
    )

    img_path = load_sample_image(0)
    result = zoom_region(img_path, 320, 240, crop_size=128)
    result = visualize_waypoint(img_path, 32, 32, label="predicted")
    result = analyze_road_geometry(img_path)
    result = find_similar_scenes(img_path, k=3)
    b64 = image_to_base64(result["similar_images"][0]["image"])
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import pickle
import sqlite3
import tempfile
from typing import Any

import cv2  # type: ignore[import-not-found]  # noqa: I001
import numpy as np  # type: ignore[import-not-found]
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DEFAULT_DATASET_PATH = "/workspace/vllm/models/dataset/"

# Default grid size used by the SceneIQ waypoint prediction model
DEFAULT_GRID_SIZE = 63

# Zoom upscale factor
ZOOM_UPSCALE = 4

# Marker drawing parameters
MARKER_RADIUS = 12
CROSSHAIR_LENGTH = 20
MARKER_COLOR = (255, 0, 0)  # bright red
MARKER_OUTLINE_COLOR = (255, 255, 255)  # white outline for visibility
TEXT_COLOR = (255, 0, 0)
TEXT_OUTLINE_COLOR = (255, 255, 255)

# Paths for FAISS index
_DIR = os.path.dirname(os.path.abspath(__file__))
SC_DB_PATH = os.path.join(
    os.path.dirname(_DIR),
    "self_consistency_experiment",
    "self_consistency.db",
)
INDEX_DIR = _DIR
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "scene_index.faiss")
INDEX_METADATA_PATH = os.path.join(INDEX_DIR, "scene_index_metadata.pkl")

# Temp directory for annotated images
TEMP_DIR = os.path.join(tempfile.gettempdir(), "sceneiq_visual_tools")
os.makedirs(TEMP_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Utility: load sample image from MDS dataset
# ------------------------------------------------------------------
_dataset_cache = None


def _get_dataset(dataset_path: str = DEFAULT_DATASET_PATH):
    """Lazily load the streaming dataset (singleton)."""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    import streaming  # type: ignore[import-not-found]

    _dataset_cache = streaming.StreamingDataset(
        local=dataset_path, shuffle=False
    )
    return _dataset_cache


def load_sample_image(
    sample_index: int,
    dataset_path: str = DEFAULT_DATASET_PATH,
    camera: str = "image_0003",
) -> str:
    """Load an image from the MDS dataset and save to a temp file.

    Parameters
    ----------
    sample_index:
        Index of the sample in the streaming dataset.
    dataset_path:
        Path to the MDS dataset directory.
    camera:
        Which camera image to load (image_0000 through image_0003).
        Default is image_0003 (most recent frame).

    Returns
    -------
    str: Path to the saved temporary JPEG file.
    """
    ds = _get_dataset(dataset_path)
    sample = ds[sample_index]
    img_bytes = sample[camera]

    # Use a stable path so repeated calls reuse the same file
    out_path = os.path.join(TEMP_DIR, f"sample_{sample_index:05d}.jpg")
    with open(out_path, "wb") as f:
        f.write(img_bytes)

    return out_path


def load_sample_metadata(
    sample_index: int,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Load metadata for a sample from the MDS dataset.

    Returns
    -------
    Dictionary with keys like odd_label, fine_class,
    long_action, lat_action, chum_uri, etc.
    """
    ds = _get_dataset(dataset_path)
    sample = ds[sample_index]
    meta = sample["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta


# ------------------------------------------------------------------
# Utility: encode image as base64
# ------------------------------------------------------------------
def image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string.

    This is useful for injecting images into OpenAI-compatible API
    messages as base64 data URIs.

    Parameters
    ----------
    image_path:
        Absolute path to the image file (JPEG or PNG).

    Returns
    -------
    str: Base64-encoded string of the image bytes.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------
def _fine_class_to_scene_type(fine_class: str) -> str:
    """Convert fine_class to the coarse scene_type label."""
    fc = fine_class.lower().strip()
    if "nominal" in fc:
        return "nominal"
    if "flood" in fc:
        return "flooded"
    if "incident" in fc:
        return "incident_zone"
    if "police" in fc or "horse" in fc or "mounted" in fc:
        return "mounted_police"
    if "flagger" in fc:
        return "flagger"
    return fc


# ------------------------------------------------------------------
# Tool 1: zoom_region
# ------------------------------------------------------------------
def zoom_region(
    image_path: str,
    center_x: int,
    center_y: int,
    crop_size: int = 128,
) -> dict[str, Any]:
    """Crop a region around the specified pixel coordinates and return
    an enlarged view for closer visual inspection.

    Parameters
    ----------
    image_path:
        Path to the driving scene image.
    center_x:
        X pixel coordinate of the region of interest (0 = left edge).
    center_y:
        Y pixel coordinate of the region of interest (0 = top edge).
    crop_size:
        Size of the square crop in pixels (default 128).

    Returns
    -------
    dict with:
        - zoomed_image: Path to the enlarged cropped image (4x upscale)
        - original_region: The pixel bounds that were cropped [x1, y1, x2, y2]
        - grid_region: What 63x63 bins this crop approximately covers
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    half = crop_size // 2

    # Compute raw crop bounds
    x1 = center_x - half
    y1 = center_y - half
    x2 = center_x + half
    y2 = center_y + half

    # Clamp to image dimensions
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Crop the region
    cropped = img.crop((x1, y1, x2, y2))

    # Upscale 4x using LANCZOS resampling
    crop_w, crop_h = cropped.size
    upscaled = cropped.resize(
        (crop_w * ZOOM_UPSCALE, crop_h * ZOOM_UPSCALE),
        Image.LANCZOS,
    )

    # Save to temp file
    fd, out_path = tempfile.mkstemp(suffix=".jpg", prefix="sceneiq_zoom_")
    with os.fdopen(fd, "wb") as f:
        upscaled.save(f, format="JPEG", quality=95)

    # Compute approximate 63x63 grid bins
    bin_w = img_w / DEFAULT_GRID_SIZE
    bin_h = img_h / DEFAULT_GRID_SIZE

    grid_col_start = int(x1 / bin_w)
    grid_col_end = int(x2 / bin_w)
    grid_row_start = int(y1 / bin_h)
    grid_row_end = int(y2 / bin_h)

    # Clamp grid indices to valid range
    grid_col_start = max(0, min(grid_col_start, DEFAULT_GRID_SIZE - 1))
    grid_col_end = max(0, min(grid_col_end, DEFAULT_GRID_SIZE - 1))
    grid_row_start = max(0, min(grid_row_start, DEFAULT_GRID_SIZE - 1))
    grid_row_end = max(0, min(grid_row_end, DEFAULT_GRID_SIZE - 1))

    return {
        "zoomed_image": out_path,
        "original_region": [x1, y1, x2, y2],
        "grid_region": {
            "row_start": grid_row_start,
            "row_end": grid_row_end,
            "col_start": grid_col_start,
            "col_end": grid_col_end,
        },
        "upscale_factor": ZOOM_UPSCALE,
        "zoomed_size": [crop_w * ZOOM_UPSCALE, crop_h * ZOOM_UPSCALE],
    }


# ------------------------------------------------------------------
# Tool 2: visualize_waypoint
# ------------------------------------------------------------------
def _get_region_description(px: int, py: int, img_w: int, img_h: int) -> str:
    """Divide the image into a 3x3 named grid and return a human-readable
    description of where the pixel falls.

    Rows:  top (0-33%), middle (33-66%), bottom (66-100%)
    Cols:  left (0-33%), center (33-66%), right (66-100%)
    """
    # Compute fractional position
    fx = px / img_w if img_w > 0 else 0.5
    fy = py / img_h if img_h > 0 else 0.5

    # Row label
    if fy < 1.0 / 3.0:
        row_label = "top"
    elif fy < 2.0 / 3.0:
        row_label = "middle"
    else:
        row_label = "bottom"

    # Column label
    if fx < 1.0 / 3.0:
        col_label = "left"
    elif fx < 2.0 / 3.0:
        col_label = "center"
    else:
        col_label = "right"

    return f"{row_label}-{col_label}"


def visualize_waypoint(
    image_path: str,
    waypoint_row: int,
    waypoint_col: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    label: str = "predicted",
) -> dict[str, Any]:
    """Draw the predicted waypoint location directly on the driving scene
    image. Maps the grid bin to pixel coordinates and draws a visible marker.

    Parameters
    ----------
    image_path:
        Path to the driving scene image.
    waypoint_row:
        Row in the grid (0 = top).
    waypoint_col:
        Column in the grid (0 = left).
    grid_size:
        Size of the grid (default 63).
    label:
        Text label for the marker (default "predicted").

    Returns
    -------
    dict with:
        - annotated_image: Path to the image with waypoint marker drawn
        - pixel_position: [x, y] pixel coordinates where marker was placed
        - image_region_description: Human-readable region name
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Map grid bin to pixel coordinates (center of the bin)
    bin_w = img_w / grid_size
    bin_h = img_h / grid_size
    px = int((waypoint_col + 0.5) * bin_w)
    py = int((waypoint_row + 0.5) * bin_h)

    # Clamp to image bounds
    px = max(0, min(px, img_w - 1))
    py = max(0, min(py, img_h - 1))

    draw = ImageDraw.Draw(img)

    # Draw white outline circle (slightly larger for contrast)
    draw.ellipse(
        [
            px - MARKER_RADIUS - 2,
            py - MARKER_RADIUS - 2,
            px + MARKER_RADIUS + 2,
            py + MARKER_RADIUS + 2,
        ],
        outline=MARKER_OUTLINE_COLOR,
        width=2,
    )

    # Draw red circle
    draw.ellipse(
        [
            px - MARKER_RADIUS,
            py - MARKER_RADIUS,
            px + MARKER_RADIUS,
            py + MARKER_RADIUS,
        ],
        outline=MARKER_COLOR,
        width=3,
    )

    # Draw crosshairs extending 20px in each direction
    # White outline for crosshairs (slightly thicker)
    draw.line(
        [(px - CROSSHAIR_LENGTH, py), (px + CROSSHAIR_LENGTH, py)],
        fill=MARKER_OUTLINE_COLOR,
        width=3,
    )
    draw.line(
        [(px, py - CROSSHAIR_LENGTH), (px, py + CROSSHAIR_LENGTH)],
        fill=MARKER_OUTLINE_COLOR,
        width=3,
    )
    # Red crosshairs on top
    draw.line(
        [(px - CROSSHAIR_LENGTH, py), (px + CROSSHAIR_LENGTH, py)],
        fill=MARKER_COLOR,
        width=1,
    )
    draw.line(
        [(px, py - CROSSHAIR_LENGTH), (px, py + CROSSHAIR_LENGTH)],
        fill=MARKER_COLOR,
        width=1,
    )

    # Add text label near the marker
    # Try to load a default font; fall back to PIL default
    _bold = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    _regular = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        font = ImageFont.truetype(_bold, 14)
    except OSError:
        try:
            font = ImageFont.truetype(_regular, 14)
        except OSError:
            font = ImageFont.load_default()

    text_x = px + MARKER_RADIUS + 5
    text_y = py - MARKER_RADIUS

    # Keep text within image bounds
    if text_x + 80 > img_w:
        text_x = px - MARKER_RADIUS - 85
    if text_y < 0:
        text_y = py + MARKER_RADIUS + 2

    # Draw text outline (white) for readability
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text(
                (text_x + dx, text_y + dy),
                label,
                fill=TEXT_OUTLINE_COLOR,
                font=font,
            )
    # Draw text in red
    draw.text((text_x, text_y), label, fill=TEXT_COLOR, font=font)

    # Save annotated image to temp file
    fd, out_path = tempfile.mkstemp(suffix=".jpg", prefix="sceneiq_waypoint_")
    with os.fdopen(fd, "wb") as f:
        img.save(f, format="JPEG", quality=95)

    # Compute human-readable region description
    region_desc = _get_region_description(px, py, img_w, img_h)

    return {
        "annotated_image": out_path,
        "pixel_position": [px, py],
        "image_region_description": region_desc,
        "grid_position": {
            "row": waypoint_row,
            "col": waypoint_col,
            "grid_size": grid_size,
        },
    }


# ------------------------------------------------------------------
# Tool 3: analyze_road_geometry
# ------------------------------------------------------------------
def analyze_road_geometry(image_path: str) -> dict[str, Any]:
    """Analyze road structure using classical computer vision.

    Uses Canny edge detection, Hough line transform, and geometric
    analysis to extract road features from a driving scene image.

    Parameters
    ----------
    image_path:
        Path to the input image file.

    Returns
    -------
    Dictionary with:
        - annotated_image: Path to image with detected features drawn
        - num_lanes_detected: Number of lane markings found
        - road_curvature: straight/gentle_left/gentle_right/sharp_left/sharp_right
        - vanishing_point: [x, y] pixel coords if detected, null if not
        - road_boundaries: Left and right road edge approximate x-positions
        - drivable_region: [x1, y1, x2, y2] pixel bounds of drivable surface
        - drivable_bins: Which 63x63 grid bins fall within drivable region
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    h, w = img.shape[:2]
    annotated = img.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur (kernel=5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection (threshold1=50, threshold2=150)
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Mask lower 60% of image (road region)
    mask = np.zeros_like(edges)
    road_top = int(h * 0.4)
    mask[road_top:, :] = 255
    masked_edges = cv2.bitwise_and(edges, mask)

    # Step 5: Hough line transform (HoughLinesP)
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=20,
    )

    # Step 6: Classify lines by angle
    lane_lines: list[tuple[int, int, int, int, float]] = []
    road_edge_lines: list[tuple[int, int, int, int, float]] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = 90.0 if dx == 0 else math.degrees(math.atan2(abs(dy), abs(dx)))

            # Near-vertical lines (50-90 degrees) = lane markings
            if 50 <= angle <= 90:
                lane_lines.append((x1, y1, x2, y2, angle))
            # Lines between 20-50 degrees could be road edges
            elif 20 <= angle < 50:
                road_edge_lines.append((x1, y1, x2, y2, angle))

    num_lanes = len(lane_lines)

    # Step 7: Estimate vanishing point from line intersections
    vanishing_point = _estimate_vanishing_point(lane_lines, w, h)

    # Step 8: Classify curvature from lane line angle distribution
    road_curvature = _classify_curvature(lane_lines, road_edge_lines, w)

    # Step 9: Estimate road boundaries
    left_boundary, right_boundary = _estimate_road_boundaries(
        lane_lines, road_edge_lines, w
    )

    # Step 10: Compute drivable region
    drivable_region = [left_boundary, road_top, right_boundary, h]

    # Step 11: Map drivable region to 63x63 bins
    drivable_bins = _compute_drivable_bins(
        left_boundary, road_top, right_boundary, h, w, h
    )

    # Draw detected features on annotated image
    # Green lines for detected lanes
    for x1, y1, x2, y2, _ in lane_lines:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Blue lines for road edges
    for x1, y1, x2, y2, _ in road_edge_lines:
        cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Red circle for vanishing point
    if vanishing_point is not None:
        vx, vy = vanishing_point
        if 0 <= vx < w and 0 <= vy < h:
            cv2.circle(annotated, (int(vx), int(vy)), 10, (0, 0, 255), 3)

    # Draw drivable region boundary (yellow)
    cv2.rectangle(
        annotated,
        (left_boundary, road_top),
        (right_boundary, h),
        (0, 255, 255),
        2,
    )

    # Add text labels
    cv2.putText(
        annotated,
        f"Lanes: {num_lanes} | Curve: {road_curvature}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Save annotated image
    basename = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(TEMP_DIR, f"{basename}_geometry.jpg")
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "annotated_image": annotated_path,
        "num_lanes_detected": num_lanes,
        "road_curvature": road_curvature,
        "vanishing_point": (
            [int(vanishing_point[0]), int(vanishing_point[1])]
            if vanishing_point is not None
            else None
        ),
        "road_boundaries": {
            "left_x": left_boundary,
            "right_x": right_boundary,
        },
        "drivable_region": drivable_region,
        "drivable_bins": drivable_bins,
        "image_size": {"width": w, "height": h},
        "total_lines_detected": len(lines) if lines is not None else 0,
    }


def _estimate_vanishing_point(
    lane_lines: list[tuple[int, int, int, int, float]],
    img_w: int,
    img_h: int,
) -> tuple[float, float] | None:
    """Estimate the vanishing point from lane line intersections.

    Finds where near-vertical lane lines converge in the upper
    portion of the image.
    """
    if len(lane_lines) < 2:
        return None

    # Find pair-wise intersections among lane lines
    intersections: list[tuple[float, float]] = []
    for i in range(len(lane_lines)):
        for j in range(i + 1, len(lane_lines)):
            pt = _line_intersection(
                lane_lines[i][:4], lane_lines[j][:4]
            )
            if pt is not None:
                px, py = pt
                # Vanishing point should be in upper half, within
                # reasonable bounds
                if (
                    -img_w < px < 2 * img_w
                    and 0 < py < img_h * 0.6
                ):
                    intersections.append((px, py))

    if not intersections:
        return None

    # Use median of intersections as robust estimate
    xs = [p[0] for p in intersections]
    ys = [p[1] for p in intersections]
    vx = float(np.median(xs))
    vy = float(np.median(ys))
    return (vx, vy)


def _line_intersection(
    line1: tuple,
    line2: tuple,
) -> tuple[float, float] | None:
    """Find intersection point of two line segments (extended).

    Uses the parametric line-line intersection formula.
    """
    x1, y1, x2, y2 = line1[:4]
    x3, y3, x4, y4 = line2[:4]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (float(px), float(py))


def _classify_curvature(
    lane_lines: list[tuple[int, int, int, int, float]],
    road_edge_lines: list[tuple[int, int, int, int, float]],
    img_w: int,
) -> str:
    """Classify road curvature from detected line angles.

    Analyzes the distribution of line angles and their positions
    relative to the image center.
    """
    all_lines = lane_lines + road_edge_lines
    if not all_lines:
        return "straight"

    center_x = img_w / 2
    left_angles: list[float] = []
    right_angles: list[float] = []

    for x1, y1, x2, y2, _angle in all_lines:
        mid_x = (x1 + x2) / 2
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue
        # Signed angle: positive = tilts right, negative = tilts left
        signed_angle = math.degrees(math.atan2(dx, abs(dy)))

        if mid_x < center_x:
            left_angles.append(signed_angle)
        else:
            right_angles.append(signed_angle)

    avg_left = float(np.mean(left_angles)) if left_angles else 0.0
    avg_right = float(np.mean(right_angles)) if right_angles else 0.0

    # Asymmetry indicates curvature direction
    asymmetry = (avg_left + avg_right) / 2.0

    if abs(asymmetry) < 3.0:
        return "straight"
    elif asymmetry > 10.0:
        return "sharp_right"
    elif asymmetry > 3.0:
        return "gentle_right"
    elif asymmetry < -10.0:
        return "sharp_left"
    else:
        return "gentle_left"


def _estimate_road_boundaries(
    lane_lines: list[tuple[int, int, int, int, float]],
    road_edge_lines: list[tuple[int, int, int, int, float]],
    img_w: int,
) -> tuple[int, int]:
    """Estimate left and right road boundary x-positions."""
    all_x: list[int] = []
    for x1, y1, x2, y2, _ in lane_lines + road_edge_lines:
        all_x.extend([x1, x2])

    if not all_x:
        # Default: assume road fills middle 70% of image
        margin = int(img_w * 0.15)
        return margin, img_w - margin

    left = max(0, int(np.percentile(all_x, 10)))
    right = min(img_w, int(np.percentile(all_x, 90)))

    # Ensure reasonable bounds (at least 20% of image width)
    if right - left < img_w * 0.2:
        center = (left + right) // 2
        half_width = int(img_w * 0.35)
        left = max(0, center - half_width)
        right = min(img_w, center + half_width)

    return left, right


def _compute_drivable_bins(
    left_x: int,
    top_y: int,
    right_x: int,
    bottom_y: int,
    img_w: int,
    img_h: int,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> list[list[int]]:
    """Compute which bins in a grid_size x grid_size grid
    fall within the drivable region.

    Returns a list of [row, col] pairs that are drivable.
    """
    bin_w = img_w / grid_size
    bin_h = img_h / grid_size

    col_start = max(0, int(left_x / bin_w))
    col_end = min(grid_size - 1, int(right_x / bin_w))
    row_start = max(0, int(top_y / bin_h))
    row_end = min(grid_size - 1, int(bottom_y / bin_h))

    bins: list[list[int]] = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            bins.append([row, col])

    return bins


# ------------------------------------------------------------------
# Tool 4: find_similar_scenes (CLIP + FAISS)
# ------------------------------------------------------------------

# Lazy-loaded CLIP model singleton
_clip_model = None
_clip_preprocess = None


def _get_clip_model():
    """Lazily load the CLIP model (singleton, CPU mode)."""
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return _clip_model, _clip_preprocess

    import open_clip  # type: ignore[import-not-found]

    logger.info("Loading CLIP model (ViT-B-32, laion2b_s34b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    # Use CPU to avoid GPU memory contention with vLLM
    model = model.cpu()
    model.eval()

    _clip_model = model
    _clip_preprocess = preprocess
    logger.info("CLIP model loaded.")
    return model, preprocess


def _embed_image_clip(img: Image.Image) -> np.ndarray:
    """Embed a single PIL image using CLIP. Returns 1-D float32 array."""
    import torch  # type: ignore[import-not-found]

    model, preprocess = _get_clip_model()
    tensor = preprocess(img).unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().astype(np.float32)


def _embed_image_from_path(image_path: str) -> np.ndarray:
    """Embed an image file using CLIP."""
    img = Image.open(image_path).convert("RGB")
    return _embed_image_clip(img)


# ------------------------------------------------------------------
# FAISS index building
# ------------------------------------------------------------------
def build_scene_index(
    dataset_path: str = DEFAULT_DATASET_PATH,
    db_path: str = SC_DB_PATH,
    output_dir: str = INDEX_DIR,
    batch_size: int = 64,
    image_key: str = "image_0003",
) -> dict[str, Any]:
    """Pre-compute CLIP embeddings and build a FAISS index.

    This is expensive (~8,000+ images) so run it once and cache.

    Parameters
    ----------
    dataset_path:
        Path to the MDS dataset directory.
    db_path:
        Path to the self-consistency SQLite DB (for GT labels).
    output_dir:
        Directory to save the FAISS index and metadata.
    batch_size:
        Number of images to embed at once.
    image_key:
        Which camera image to use from the MDS sample.

    Returns
    -------
    Dictionary with build statistics.
    """
    import faiss  # type: ignore[import-not-found]
    from streaming import StreamingDataset  # type: ignore[import-not-found]

    faiss_path = os.path.join(output_dir, "scene_index.faiss")
    meta_path = os.path.join(output_dir, "scene_index_metadata.pkl")

    # Load CLIP model
    model, preprocess = _get_clip_model()

    # Load dataset
    logger.info("Loading MDS dataset from %s", dataset_path)
    ds = StreamingDataset(local=dataset_path, shuffle=False)
    total = len(ds)
    logger.info("Dataset has %d samples", total)

    # Load GT labels from DB if available
    gt_labels: dict[int, dict[str, str]] = {}
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT sample_id, scene_type_gt, long_action_gt, "
                "lat_action_gt FROM predictions"
            ).fetchall()
            for sid, scene, la, lat in rows:
                gt_labels[sid] = {
                    "scene_type_gt": scene,
                    "long_action_gt": la,
                    "lat_action_gt": lat,
                }
            conn.close()
        except Exception as e:
            logger.warning("Could not load GT from DB: %s", e)
    logger.info("Loaded GT labels for %d samples from DB", len(gt_labels))

    # Embed all images in batches
    all_embeddings: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []

    logger.info(
        "Embedding %d images (batch_size=%d)...", total, batch_size
    )
    batch_tensors: list = []
    batch_indices: list[int] = []

    for i in range(total):
        if i % 500 == 0:
            logger.info(
                "  Progress: %d / %d (%.1f%%)",
                i, total, 100 * i / total,
            )

        try:
            sample = ds[i]
            img_bytes = sample[image_key]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = preprocess(img)
            batch_tensors.append(tensor)
            batch_indices.append(i)
        except Exception as e:
            logger.warning("  Skipping sample %d: %s", i, e)
            continue

        # Process batch when full
        if len(batch_tensors) >= batch_size:
            _process_embedding_batch(
                model, batch_tensors, batch_indices,
                ds, gt_labels, all_embeddings, all_metadata,
            )
            batch_tensors = []
            batch_indices = []

    # Process remaining batch
    if batch_tensors:
        _process_embedding_batch(
            model, batch_tensors, batch_indices,
            ds, gt_labels, all_embeddings, all_metadata,
        )

    logger.info("Embedded %d images successfully.", len(all_embeddings))

    if not all_embeddings:
        return {"error": "No images were successfully embedded."}

    # Build FAISS index (inner product = cosine sim for normalized vecs)
    embedding_matrix = np.stack(all_embeddings)
    dim = embedding_matrix.shape[1]
    logger.info(
        "Building FAISS index: %d vectors of dim %d",
        embedding_matrix.shape[0], dim,
    )

    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    # Save index and metadata
    faiss.write_index(index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_metadata, f)

    logger.info("FAISS index saved to %s", faiss_path)
    logger.info("Metadata saved to %s", meta_path)

    # Compute scene distribution
    scene_counts: dict[str, int] = {}
    for m in all_metadata:
        s = m["scene_type_gt"]
        scene_counts[s] = scene_counts.get(s, 0) + 1

    stats = {
        "total_indexed": len(all_embeddings),
        "embedding_dim": dim,
        "faiss_index_path": faiss_path,
        "metadata_path": meta_path,
        "scene_distribution": scene_counts,
    }
    logger.info("Index build stats: %s", json.dumps(stats, indent=2))
    return stats


def _process_embedding_batch(
    model,
    batch_tensors: list,
    batch_indices: list[int],
    ds,
    gt_labels: dict[int, dict[str, str]],
    all_embeddings: list[np.ndarray],
    all_metadata: list[dict[str, Any]],
) -> None:
    """Process a batch of image tensors: embed and collect metadata."""
    import torch  # type: ignore[import-not-found]

    batch_tensor = torch.stack(batch_tensors)
    with torch.no_grad():
        features = model.encode_image(batch_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    embs = features.cpu().numpy().astype(np.float32)

    for j, idx in enumerate(batch_indices):
        all_embeddings.append(embs[j])

        # Get metadata from the MDS sample
        sample_meta = ds[idx]["metadata"]
        if isinstance(sample_meta, str):
            sample_meta = json.loads(sample_meta)

        odd_label = sample_meta.get("odd_label", "unknown")
        fine_class = sample_meta.get("fine_class", "unknown")
        scene_type = _fine_class_to_scene_type(fine_class)

        # Prefer DB GT if available, else derive from MDS metadata
        db_gt = gt_labels.get(idx, {})
        entry = {
            "dataset_index": idx,
            "scene_type_gt": db_gt.get("scene_type_gt", scene_type),
            "long_action_gt": db_gt.get(
                "long_action_gt",
                sample_meta.get("long_action", "null"),
            ),
            "lat_action_gt": db_gt.get(
                "lat_action_gt",
                sample_meta.get("lat_action", "null"),
            ),
            "odd_label": odd_label,
            "fine_class": fine_class,
        }
        all_metadata.append(entry)


# Cached FAISS index and metadata
_faiss_index = None
_faiss_metadata = None


def _load_faiss_index():
    """Lazily load the FAISS index and metadata (singleton)."""
    global _faiss_index, _faiss_metadata
    if _faiss_index is not None:
        return _faiss_index, _faiss_metadata

    import faiss  # type: ignore[import-not-found]

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            f"Run build_scene_index() first."
        )
    if not os.path.exists(INDEX_METADATA_PATH):
        raise FileNotFoundError(
            f"Index metadata not found at {INDEX_METADATA_PATH}. "
            f"Run build_scene_index() first."
        )

    logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(INDEX_METADATA_PATH, "rb") as f:
        _faiss_metadata = pickle.load(f)
    logger.info(
        "FAISS index loaded: %d vectors", _faiss_index.ntotal,
    )
    return _faiss_index, _faiss_metadata


def find_similar_scenes(
    image_path: str,
    k: int = 3,
) -> dict[str, Any]:
    """Find k most visually similar images from the eval set.

    Uses CLIP embeddings and FAISS cosine similarity search.

    Parameters
    ----------
    image_path:
        Path to the query image.
    k:
        Number of similar scenes to retrieve.

    Returns
    -------
    Dictionary with:
        - similar_images: List of k entries with image path,
          similarity score, and ground truth labels
        - consensus: Majority scene label and agreement ratio
    """
    try:
        index, metadata = _load_faiss_index()
    except FileNotFoundError as e:
        return {
            "error": str(e),
            "suggestion": (
                "Run build_scene_index() to create the FAISS index."
            ),
        }

    # Embed query image
    query_emb = _embed_image_from_path(image_path)
    query_emb = query_emb.reshape(1, -1)

    # Search: k+5 because the query itself might be in the index
    search_k = min(k + 5, index.ntotal)
    scores, indices = index.search(query_emb, search_k)

    # Collect results, skipping exact matches (score ~ 1.0)
    similar_images: list[dict[str, Any]] = []
    for i in range(search_k):
        idx = int(indices[0][i])
        score = float(scores[0][i])

        # Skip near-perfect matches (likely the query image itself)
        if score > 0.999:
            continue

        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        dataset_idx = meta["dataset_index"]

        # Save the similar image to a temp file
        try:
            img_path = load_sample_image(dataset_idx)
        except Exception as e:
            logger.warning(
                "Could not load similar image %d: %s",
                dataset_idx, e,
            )
            img_path = None

        similar_images.append({
            "image": img_path,
            "dataset_index": dataset_idx,
            "similarity_score": round(score, 4),
            "ground_truth_scene": meta["scene_type_gt"],
            "ground_truth_long_action": meta["long_action_gt"],
            "ground_truth_lat_action": meta["lat_action_gt"],
            "odd_label": meta.get("odd_label", "unknown"),
            "fine_class": meta.get("fine_class", "unknown"),
        })

        if len(similar_images) >= k:
            break

    # Compute consensus among neighbors
    consensus = _compute_consensus(similar_images)

    return {
        "similar_images": similar_images,
        "consensus": consensus,
        "query_image": image_path,
        "k_requested": k,
        "k_returned": len(similar_images),
    }


def _compute_consensus(
    similar_images: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute consensus among k nearest neighbors.

    Returns the majority scene label, agreement ratio, and
    action consensus.
    """
    if not similar_images:
        return {
            "scene_label": "unknown",
            "scene_agreement": 0.0,
            "note": "No similar images found.",
        }

    # Scene consensus
    scene_counts: dict[str, int] = {}
    for s in similar_images:
        label = s["ground_truth_scene"]
        scene_counts[label] = scene_counts.get(label, 0) + 1

    total = len(similar_images)
    majority_scene = max(scene_counts, key=lambda x: scene_counts[x])
    agreement = scene_counts[majority_scene] / total

    # Action consensus
    action_counts: dict[str, int] = {}
    for s in similar_images:
        key = (
            f"{s['ground_truth_long_action']}|"
            f"{s['ground_truth_lat_action']}"
        )
        action_counts[key] = action_counts.get(key, 0) + 1

    majority_action = max(action_counts, key=lambda x: action_counts[x])
    action_agreement = action_counts[majority_action] / total

    result: dict[str, Any] = {
        "scene_label": majority_scene,
        "scene_agreement": round(agreement, 3),
        "scene_distribution": scene_counts,
        "action_label": majority_action,
        "action_agreement": round(action_agreement, 3),
    }

    # Add interpretation
    if agreement >= 0.8:
        result["note"] = (
            f"Strong consensus: {int(agreement * 100)}% of similar "
            f"scenes are labeled {majority_scene}. This is likely "
            f"the correct scene type."
        )
    elif agreement >= 0.5:
        result["note"] = (
            f"Moderate consensus: {int(agreement * 100)}% of similar "
            f"scenes are labeled {majority_scene}. Distribution: "
            f"{scene_counts}. Some ambiguity exists."
        )
    else:
        result["note"] = (
            f"Weak consensus: only {int(agreement * 100)}% agree on "
            f"{majority_scene}. Distribution: {scene_counts}. "
            f"Similar scenes span multiple classes, suggesting this "
            f"is an ambiguous or boundary case."
        )

    return result


# ------------------------------------------------------------------
# Tool definitions in OpenAI function-calling format
# ------------------------------------------------------------------
TOOL_ZOOM: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "zoom_region",
        "description": (
            "Crops a region around the specified pixel coordinates "
            "and returns an enlarged view for closer visual inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "center_x": {
                    "type": "integer",
                    "description": (
                        "X pixel coordinate of region center "
                        "(0 = left edge)"
                    ),
                },
                "center_y": {
                    "type": "integer",
                    "description": (
                        "Y pixel coordinate of region center "
                        "(0 = top edge)"
                    ),
                },
                "crop_size": {
                    "type": "integer",
                    "description": (
                        "Size of square crop in pixels (default 128)"
                    ),
                    "default": 128,
                },
            },
            "required": ["center_x", "center_y"],
        },
    },
}

TOOL_WAYPOINT_VIZ: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "visualize_waypoint",
        "description": (
            "Draws a predicted waypoint location on the driving scene "
            "image. Maps a 63x63 grid coordinate to pixel position and "
            "draws a visible marker."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "waypoint_row": {
                    "type": "integer",
                    "description": "Row in 63x63 grid (0 = top)",
                },
                "waypoint_col": {
                    "type": "integer",
                    "description": "Column in 63x63 grid (0 = left)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Label for the marker (default: 'predicted')"
                    ),
                    "default": "predicted",
                },
            },
            "required": ["waypoint_row", "waypoint_col"],
        },
    },
}

TOOL_ROAD_GEOMETRY: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_road_geometry",
        "description": (
            "Analyzes road structure using computer vision. Returns "
            "detected lane markings, road curvature, vanishing point, "
            "and drivable region. The annotated image shows detected "
            "features."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TOOL_SIMILAR_SCENES: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_similar_scenes",
        "description": (
            "Finds the k most visually similar driving scenes from a "
            "reference database. Returns the similar images with their "
            "known ground truth labels. Use this to see what similar "
            "scenes were actually classified as."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "k": {
                    "type": "integer",
                    "description": (
                        "Number of similar scenes to retrieve "
                        "(default 3)"
                    ),
                    "default": 3,
                },
            },
            "required": [],
        },
    },
}

# All visual tools
ALL_VISUAL_TOOLS: list[dict[str, Any]] = [
    TOOL_ZOOM,
    TOOL_WAYPOINT_VIZ,
    TOOL_ROAD_GEOMETRY,
    TOOL_SIMILAR_SCENES,
]


# ------------------------------------------------------------------
# Dispatcher for executing visual tools by name
# ------------------------------------------------------------------
def execute_visual_tool(
    tool_name: str,
    arguments: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    """Execute a visual tool by name.

    Parameters
    ----------
    tool_name:
        One of the visual tool function names.
    arguments:
        Keyword arguments for the tool (excluding image_path).
    image_path:
        Path to the source image file.

    Returns
    -------
    dict with tool result (JSON-serializable, except paths are
    local file paths).
    """
    if tool_name == "zoom_region":
        return zoom_region(
            image_path=image_path,
            center_x=arguments["center_x"],
            center_y=arguments["center_y"],
            crop_size=arguments.get("crop_size", 128),
        )
    elif tool_name == "visualize_waypoint":
        return visualize_waypoint(
            image_path=image_path,
            waypoint_row=arguments["waypoint_row"],
            waypoint_col=arguments["waypoint_col"],
            grid_size=arguments.get(
                "grid_size", DEFAULT_GRID_SIZE
            ),
            label=arguments.get("label", "predicted"),
        )
    elif tool_name == "analyze_road_geometry":
        return analyze_road_geometry(image_path=image_path)
    elif tool_name == "find_similar_scenes":
        k = arguments.get("k", 3)
        return find_similar_scenes(image_path=image_path, k=k)
    else:
        return {
            "error": f"Unknown visual tool: {tool_name}",
            "available_tools": [
                "zoom_region",
                "visualize_waypoint",
                "analyze_road_geometry",
                "find_similar_scenes",
            ],
        }


# ------------------------------------------------------------------
# CLI: build index and test tools
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "build-index":
        print("=== Building FAISS scene index ===")
        stats = build_scene_index()
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    print("=== Visual Tools Self-Test ===\n")

    # Load a sample image
    print("Loading sample image from MDS dataset...")
    img_path = load_sample_image(0)
    print(f"  Image path: {img_path}")

    # Check dimensions
    with Image.open(img_path) as img:
        print(f"  Dimensions: {img.size}")
        print(f"  Mode: {img.mode}")

    # Test zoom_region
    print("\nTesting zoom_region(center_x=320, center_y=240, crop_size=128)...")
    result = zoom_region(img_path, 320, 240, crop_size=128)
    print(f"  zoomed_image: {result['zoomed_image']}")
    print(f"  original_region: {result['original_region']}")
    print(f"  grid_region: {result['grid_region']}")
    print(f"  zoomed_size: {result['zoomed_size']}")
    with Image.open(result["zoomed_image"]) as zimg:
        print(f"  Actual zoomed size: {zimg.size}")
    assert os.path.exists(result["zoomed_image"]), "Zoomed image file missing"

    # Test visualize_waypoint (center of grid)
    print("\nTesting visualize_waypoint(row=32, col=32, label='test')...")
    result = visualize_waypoint(img_path, 32, 32, label="test")
    print(f"  annotated_image: {result['annotated_image']}")
    print(f"  pixel_position: {result['pixel_position']}")
    print(f"  image_region_description: {result['image_region_description']}")
    assert os.path.exists(result["annotated_image"]), "Annotated image missing"

    # Test analyze_road_geometry
    print("\nTesting analyze_road_geometry...")
    geo_result = analyze_road_geometry(img_path)
    print(f"  Curvature: {geo_result['road_curvature']}")
    print(f"  Lanes detected: {geo_result['num_lanes_detected']}")
    print(f"  Vanishing point: {geo_result['vanishing_point']}")
    print(f"  Road boundaries: {geo_result['road_boundaries']}")
    print(f"  Drivable region: {geo_result['drivable_region']}")
    print(f"  Drivable bins count: {len(geo_result['drivable_bins'])}")
    print(f"  Annotated image: {geo_result['annotated_image']}")
    print(f"  Total lines: {geo_result['total_lines_detected']}")
    assert os.path.exists(geo_result["annotated_image"]), (
        "Geometry annotated image missing"
    )

    # Test find_similar_scenes (if index exists)
    if os.path.exists(FAISS_INDEX_PATH):
        print("\nTesting find_similar_scenes...")
        sim_result = find_similar_scenes(img_path, k=3)
        for s in sim_result["similar_images"]:
            print(
                f"  Similar: scene={s['ground_truth_scene']}, "
                f"score={s['similarity_score']:.3f}"
            )
        print(f"  Consensus: {sim_result['consensus']}")
    else:
        print(
            f"\nFAISS index not found at {FAISS_INDEX_PATH}"
        )
        print("Run: python visual_tools.py build-index")

    # Test base64 encoding
    print("\nTesting image_to_base64...")
    b64 = image_to_base64(img_path)
    print(f"  base64 length: {len(b64)}")
    assert len(b64) > 100, "Base64 string too short"

    # Test execute_visual_tool dispatcher
    print("\nTesting execute_visual_tool dispatcher...")
    result = execute_visual_tool(
        "analyze_road_geometry", {}, img_path,
    )
    print(f"  geometry result keys: {list(result.keys())}")

    result = execute_visual_tool(
        "find_similar_scenes", {"k": 2}, img_path,
    )
    if "error" in result:
        print(f"  similar scenes (no index): {result['error'][:60]}...")
    else:
        print(f"  similar scenes returned: {result['k_returned']}")

    result = execute_visual_tool("nonexistent", {}, img_path)
    assert "error" in result
    print(f"  Unknown tool error: {result['error']}")

    print("\n=== All tests passed ===")
