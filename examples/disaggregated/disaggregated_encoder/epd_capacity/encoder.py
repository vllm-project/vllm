# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from InferRSI examples/epd_provisioning/calibration.py.
"""Fit encoder occupancy curves from caller-supplied measurement summaries."""

import warnings
from math import isfinite
from statistics import fmean

SERIAL_DEGREES = {"H": 1, "M": 1, "C": 1, "X": 2, "G": 2, "R": 1}
SATURATED_DEGREES = {"H": 0, "M": 1, "C": 0, "X": 2, "G": 2, "R": 0}


def token_count(value):
    if type(value) is not int or value < 0:
        raise ValueError("visual_tokens must be a nonnegative integer")
    return value


def milliseconds(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("duration must be finite, nonnegative and in milliseconds")
    return value


def fit_encoder(data):
    """Fit separate strict-serial spans and saturated active resource times."""
    from numpy.polynomial import Polynomial

    anchors = data["anchors"]
    tokens = [token_count(row["visual_tokens"]) for row in anchors]
    if len(set(tokens)) < 3 or len(set(tokens)) != len(tokens):
        raise ValueError(
            "supply one row per anchor, with at least three distinct token counts"
        )
    if len(tokens) == 3:
        warnings.warn(
            "Three anchors give no leave-one-anchor-out validation for quadratic "
            "components. Supply additional distinct token counts.",
            stacklevel=2,
        )
    saturated_degrees = dict(SATURATED_DEGREES)
    if "media_decode" in data["parallelism"]:
        saturated_degrees["media_decode"] = 1
    parallelism = {
        lane: milliseconds(data["parallelism"][lane]) for lane in saturated_degrees
    }
    if not all(parallelism.values()):
        raise ValueError("all effective lane parallelisms must be positive")
    for row in anchors:
        for key, degrees in (
            ("serial_ms", SERIAL_DEGREES),
            ("saturated_ms", saturated_degrees),
        ):
            spans = {lane: milliseconds(row[key][lane]) for lane in degrees}
            if spans["G"] > spans["X"]:
                raise ValueError(
                    f"{key}: the GPU interval must fit inside the worker interval"
                )

    lo, hi = min(tokens), max(tokens)

    def fit_affine(sizes, times):
        model = Polynomial.fit(sizes, times, 1).convert()
        if min(model(lo), model(hi)) >= 0:
            return model
        # An infeasible unconstrained fit reaches its constrained optimum
        # on an endpoint boundary. Search both boundaries in the fitted range.
        candidates = []
        for endpoint, direction in ((lo, 1), (hi, -1)):
            basis = [direction * (size - endpoint) for size in sizes]
            slope = sum(x * y for x, y in zip(basis, times, strict=True)) / sum(
                x**2 for x in basis
            )
            candidates.append(
                Polynomial([-direction * slope * endpoint, direction * slope])
            )
        return min(
            candidates,
            key=lambda fit: sum(
                (fit(v) - t) ** 2 for v, t in zip(sizes, times, strict=True)
            ),
        )

    def fit(key, degrees):
        def fit_component(rows, lane, degree):
            sizes = [row["visual_tokens"] for row in rows]
            if lane != "X":
                times = [row[key][lane] for row in rows]
                if degree == 1:
                    return fit_affine(sizes, times)
                return Polynomial.fit(sizes, times, degree).convert()
            # The forward is nested in the worker span. Only the forward
            # contributes curvature to the worker model.
            gpu_times = [row[key]["G"] for row in rows]
            cpu_times = [row[key]["X"] - row[key]["G"] for row in rows]
            gpu = Polynomial.fit(sizes, gpu_times, degrees["G"]).convert()
            cpu = fit_affine(sizes, cpu_times)
            return gpu + cpu

        coefficients = {}
        diagnostics = {}
        for lane, degree in degrees.items():
            observed = [row[key][lane] for row in anchors]
            model = fit_component(anchors, lane, degree)
            # A polynomial's minimum is at an endpoint or a stationary point.
            points = [lo, hi]
            points.extend(point for point in model.deriv().roots() if lo < point < hi)
            for point in points:
                duration = model(point)
                if not isfinite(duration) or duration < 0:
                    raise ValueError(
                        f"{key}.{lane} must be finite and nonnegative across "
                        f"[{lo}, {hi}] visual tokens. Got {duration:.6g} ms at "
                        f"{point:.6g} visual tokens."
                    )
            coefficients[lane] = model.coef.tolist()
            residual = sum(
                (value - model(token)) ** 2
                for token, value in zip(tokens, observed, strict=True)
            )
            mean = fmean(observed)
            variance = sum((value - mean) ** 2 for value in observed)
            errors = []
            if len(tokens) > degree + 1:
                for index, token in enumerate(tokens):
                    rows = anchors[:index] + anchors[index + 1 :]
                    reduced = fit_component(rows, lane, degree)
                    errors.append(abs(float(reduced(token)) - observed[index]))
            diagnostics[lane] = {
                "r_squared": float(1 - residual / variance) if variance else None,
                "leave_one_anchor_out_mae_ms": fmean(errors) if errors else None,
            }
        return coefficients, diagnostics

    serial, serial_diagnostics = fit("serial_ms", SERIAL_DEGREES)
    saturated, saturated_diagnostics = fit("saturated_ms", saturated_degrees)
    return {
        "visual_token_range": [min(tokens), max(tokens)],
        "serial_ms": serial,
        "saturated_ms": saturated,
        "parallelism": parallelism,
        "fit_diagnostics": {
            "serial": serial_diagnostics,
            "saturated": saturated_diagnostics,
        },
    }


def encoder_demands(model, visual_tokens):
    """Predict disjoint serial elapsed time and normalized resource demands."""
    visual_tokens = token_count(visual_tokens)
    lo, hi = model["visual_token_range"]
    if not lo <= visual_tokens <= hi:
        raise ValueError(
            f"visual_tokens={visual_tokens} outside calibrated range [{lo}, {hi}]"
        )

    def predict(key, lane):
        coeffs = model[key][lane]
        value = 0.0
        for coefficient in reversed(coeffs):
            value = value * visual_tokens + coefficient
        # Least-squares fits can produce negative demands. Require recalibration
        # instead of silently treating a negative fit as a zero-time resource.
        return milliseconds(value)

    serial = {lane: predict("serial_ms", lane) for lane in SERIAL_DEGREES}
    active = {lane: predict("saturated_ms", lane) for lane in model["saturated_ms"]}
    if any(
        times["G"] > times["X"] + 1e-9 * max(1.0, times["X"])
        for times in (serial, active)
    ):
        raise ValueError("fitted GPU interval exceeds fitted worker cycle")
    return (
        sum(serial[lane] for lane in ("H", "M", "C", "X", "R")),
        {lane: active[lane] / model["parallelism"][lane] for lane in active},
    )


def workload_rates(model, requests, pd_rate=None):
    """Average each resource before taking the workload-level bottleneck."""
    if not requests:
        raise ValueError("workload must contain at least one request")
    profiles = [
        encoder_demands(model, request["visual_tokens"]) for request in requests
    ]
    means = {
        lane: fmean(demands[lane] for _, demands in profiles)
        for lane in model["saturated_ms"]
    }
    bottleneck = max(means, key=means.__getitem__)
    if means[bottleneck] <= 0:
        raise ValueError("mean encoder bottleneck demand must be positive")
    if pd_rate is None:
        pd_ms = fmean(milliseconds(request["pd_demand_ms"]) for request in requests)
        if pd_ms <= 0:
            raise ValueError("mean PD demand must be positive")
        pd_rate = 1000 / pd_ms
    return {
        "e_rate_upper_bound_rps": 1000 / means[bottleneck],
        "pd_rate_rps": pd_rate,
        "encoder_bottleneck_lane": bottleneck,
        "mean_normalized_lane_ms": means,
        "mean_serial_e_ms": fmean(serial for serial, _ in profiles),
        "request_count": len(requests),
    }
