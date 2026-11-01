"""Primary-court selection, temporal lock, subpixel decode, and release gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pipeline.court_keypoints import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    LANDMARKS,
    fit_homography_from_landmarks,
)


# Held-out eval sources must never enter training manifests.
DEFAULT_HELD_OUT_SOURCE_PREFIXES: tuple[str, ...] = ("bad_2",)

# Release thresholds for claiming usable court calibration on a labeled eval set.
RELEASE_THRESHOLDS: dict[str, float] = {
    "max_mean_mae_source_px": 80.0,
    "min_mean_visible_coverage": 0.8,
    "max_abstention_rate_calibratable": 0.4,
    "min_fit_acceptance_rate_calibratable": 0.55,
}

TRUSTED_CALIBRATION_STATUSES = frozenset({"verified"})
UNTRUSTED_CALIBRATION_STATUSES = frozenset(
    {"unverified", "model_prediction_unverified", "provisional", "abstained", "diagnostic"}
)


def decode_heatmap_peak_subpixel(heatmap: np.ndarray) -> tuple[float, float, float]:
    """Return (x, y, peak_value) with a local quadratic refinement around the max."""
    if heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError("heatmap must be a non-empty 2D array")
    flat_index = int(np.argmax(heatmap))
    y0, x0 = np.unravel_index(flat_index, heatmap.shape)
    peak = float(heatmap[y0, x0])
    x_ref, y_ref = float(x0), float(y0)
    height, width = heatmap.shape
    if 0 < x0 < width - 1:
        left, center, right = float(heatmap[y0, x0 - 1]), peak, float(heatmap[y0, x0 + 1])
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-6:
            x_ref = x0 + 0.5 * (left - right) / denom
    if 0 < y0 < height - 1:
        up, center, down = float(heatmap[y0 - 1, x0]), peak, float(heatmap[y0 + 1, x0])
        denom = up - 2.0 * center + down
        if abs(denom) > 1e-6:
            y_ref = y0 + 0.5 * (up - down) / denom
    return float(np.clip(x_ref, 0, width - 1)), float(np.clip(y_ref, 0, height - 1)), peak


def _court_center_image(image_to_court: np.ndarray) -> tuple[float, float]:
    court_to_image = np.linalg.inv(image_to_court)
    center = np.asarray([[[COURT_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(center, court_to_image).reshape(2)
    return float(projected[0]), float(projected[1])


def score_court_hypothesis(
    image_to_court: np.ndarray,
    diagnostics: dict[str, Any],
    image_shape: tuple[int, ...],
) -> float:
    """Higher is better: fit quality, completeness, and centrality."""
    height, width = image_shape[:2]
    cx, cy = _court_center_image(image_to_court)
    # Prefer courts whose center sits near mid-width and slightly below mid-height.
    target = (width * 0.5, height * 0.58)
    distance = float(np.hypot(cx - target[0], cy - target[1]))
    centrality = 1.0 - min(1.0, distance / float(np.hypot(width, height)))
    completeness = float(diagnostics["inlier_landmarks"]) / float(len(LANDMARKS))
    rmse = float(diagnostics["reprojection_rmse_px"])
    fit_quality = 1.0 / (1.0 + rmse)
    return 0.45 * fit_quality + 0.35 * centrality + 0.20 * completeness


def _candidate_point_sets(
    points: dict[str, list[float] | None],
    image_shape: tuple[int, ...],
) -> list[dict[str, list[float] | None]]:
    height, width = image_shape[:2]
    visible = {name: point for name, point in points.items() if point is not None}
    candidates = [points]
    if len(visible) < 4:
        return candidates

    def subset(predicate) -> dict[str, list[float] | None]:
        result = {name: None for name in points}
        for name, point in visible.items():
            if predicate(point):
                result[name] = point
        return result

    candidates.append(subset(lambda p: p[0] <= width * 0.62))
    candidates.append(subset(lambda p: p[0] >= width * 0.38))
    candidates.append(
        subset(lambda p: width * 0.12 <= p[0] <= width * 0.88 and height * 0.08 <= p[1] <= height * 0.95)
    )
    # Deduplicate by visible name set.
    unique: list[dict[str, list[float] | None]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(sorted(name for name, point in candidate.items() if point is not None))
        if key in seen or len(key) < 4:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique or [points]


def select_primary_court(
    points: dict[str, list[float] | None],
    image_shape: tuple[int, ...],
    ambiguity_margin: float = 0.03,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and auto-pick one primary court when neighbor courts pollute edges."""
    scored: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    errors: list[str] = []
    for candidate in _candidate_point_sets(points, image_shape):
        try:
            matrix, diagnostics = fit_homography_from_landmarks(candidate)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        score = score_court_hypothesis(matrix, diagnostics, image_shape)
        payload = {
            **diagnostics,
            "score": score,
            "court_center_image": list(_court_center_image(matrix)),
        }
        scored.append((score, matrix, payload))
    if not scored:
        reason = errors[0] if errors else "No viable primary-court hypothesis"
        raise ValueError(reason)
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_matrix, best_diag = scored[0]
    if len(scored) >= 2 and (best_score - scored[1][0]) < ambiguity_margin:
        raise ValueError("Ambiguous primary court: top hypotheses are too close in score")
    best_diag["hypotheses_considered"] = len(scored)
    best_diag["runner_up_score"] = scored[1][0] if len(scored) > 1 else None
    return best_matrix, best_diag


def outline_iou(image_to_court_a: np.ndarray, image_to_court_b: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Approximate overlap of projected full-court outlines inside the image."""
    height, width = image_shape[:2]
    corners = np.asarray(
        [[[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]]],
        dtype=np.float32,
    )

    def mask_for(matrix: np.ndarray) -> np.ndarray:
        court_to_image = np.linalg.inv(matrix)
        outline = cv2.perspectiveTransform(corners, court_to_image).reshape(-1, 2)
        canvas = np.zeros((height, width), dtype=np.uint8)
        clipped = np.round(outline).astype(np.int32)
        cv2.fillConvexPoly(canvas, clipped, 1)
        return canvas

    a = mask_for(image_to_court_a)
    b = mask_for(image_to_court_b)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union else 0.0


@dataclass
class CourtTemporalLock:
    """Keep one court solution across a static clip once confidence is established."""

    min_lock_frames: int = 3
    switch_iou_threshold: float = 0.55
    locked_matrix: np.ndarray | None = None
    lock_count: int = 0
    pending_matrix: np.ndarray | None = None
    pending_count: int = 0

    def update(
        self,
        points: dict[str, list[float] | None],
        image_shape: tuple[int, ...],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        try:
            matrix, diagnostics = select_primary_court(points, image_shape)
            frame_status = "detected"
        except ValueError as exc:
            if self.locked_matrix is not None:
                return self.locked_matrix, {
                    "status": "held_previous",
                    "reason": str(exc),
                    "lock_count": self.lock_count,
                }
            return None, {"status": "abstained", "reason": str(exc)}

        if self.locked_matrix is None:
            self.lock_count += 1
            if self.lock_count >= self.min_lock_frames:
                self.locked_matrix = matrix
                return matrix, {**diagnostics, "status": "locked_new", "lock_count": self.lock_count}
            self.locked_matrix = matrix  # provisional first accept
            return matrix, {**diagnostics, "status": "locking", "lock_count": self.lock_count}

        iou = outline_iou(self.locked_matrix, matrix, image_shape)
        if iou >= self.switch_iou_threshold:
            self.pending_matrix = None
            self.pending_count = 0
            self.lock_count += 1
            return self.locked_matrix, {
                **diagnostics,
                "status": "locked",
                "iou_with_lock": iou,
                "lock_count": self.lock_count,
            }

        # Candidate switch: require sustained disagreement before flipping courts.
        if self.pending_matrix is None:
            self.pending_matrix = matrix
            self.pending_count = 1
        else:
            pending_iou = outline_iou(self.pending_matrix, matrix, image_shape)
            if pending_iou >= self.switch_iou_threshold:
                self.pending_count += 1
                self.pending_matrix = matrix
            else:
                self.pending_matrix = matrix
                self.pending_count = 1
        if self.pending_count >= self.min_lock_frames:
            self.locked_matrix = self.pending_matrix
            self.pending_matrix = None
            self.pending_count = 0
            self.lock_count = self.min_lock_frames
            return self.locked_matrix, {
                **diagnostics,
                "status": "switched",
                "iou_with_previous": iou,
                "lock_count": self.lock_count,
            }
        return self.locked_matrix, {
            **diagnostics,
            "status": "locked_hold_pending_switch",
            "iou_with_lock": iou,
            "pending_count": self.pending_count,
            "frame_status": frame_status,
            "lock_count": self.lock_count,
        }


def assert_training_excludes_held_out(
    source_videos: list[str],
    held_out_prefixes: tuple[str, ...] = DEFAULT_HELD_OUT_SOURCE_PREFIXES,
) -> None:
    offenders = sorted(
        {
            source
            for source in source_videos
            if any(
                source == prefix
                or source.startswith(f"{prefix}-")
                or source.split("-", 1)[0] == prefix
                for prefix in held_out_prefixes
            )
        }
    )
    if offenders:
        raise ValueError(
            "Held-out evaluation sources must not enter training: "
            + ", ".join(offenders)
        )


def summarize_eval_for_release(eval_payload: dict[str, Any]) -> dict[str, Any]:
    """Compute release-oriented metrics; only frames with >=4 labeled visibles are calibratable."""
    rows = eval_payload.get("results", [])
    calibratable = [row for row in rows if int(row.get("annotated_visible") or 0) >= 4]
    if not calibratable:
        return {
            "calibratable_frames": 0,
            "passed": False,
            "reasons": ["No calibratable frames with >=4 annotated visible landmarks"],
            "thresholds": RELEASE_THRESHOLDS,
        }
    coverages = [float(row["visible_coverage"]) for row in calibratable if row.get("visible_coverage") is not None]
    maes = [float(row["mae_source_px"]) for row in calibratable if row.get("mae_source_px") is not None]
    abstained = sum(row.get("status") == "abstained" for row in calibratable)
    accepted = sum(row.get("status") != "abstained" for row in calibratable)
    summary = {
        "calibratable_frames": len(calibratable),
        "mean_visible_coverage": float(np.mean(coverages)) if coverages else None,
        "mean_mae_source_px": float(np.mean(maes)) if maes else None,
        "abstention_rate_calibratable": abstained / len(calibratable),
        "fit_acceptance_rate_calibratable": accepted / len(calibratable),
        "thresholds": RELEASE_THRESHOLDS,
    }
    reasons: list[str] = []
    if summary["mean_visible_coverage"] is None or summary["mean_visible_coverage"] < RELEASE_THRESHOLDS["min_mean_visible_coverage"]:
        reasons.append("visible coverage below release threshold")
    if summary["mean_mae_source_px"] is None or summary["mean_mae_source_px"] > RELEASE_THRESHOLDS["max_mean_mae_source_px"]:
        reasons.append("landmark MAE above release threshold")
    if summary["abstention_rate_calibratable"] > RELEASE_THRESHOLDS["max_abstention_rate_calibratable"]:
        reasons.append("abstention rate too high on calibratable frames")
    if summary["fit_acceptance_rate_calibratable"] < RELEASE_THRESHOLDS["min_fit_acceptance_rate_calibratable"]:
        reasons.append("fit acceptance too low on calibratable frames")
    summary["passed"] = not reasons
    summary["reasons"] = reasons
    summary["metrics_trustworthy"] = bool(summary["passed"])
    return summary


def movement_metrics_allowed(calibration_status: str) -> bool:
    """Movement/speed numbers may only be presented as trustworthy when verified."""
    return calibration_status in TRUSTED_CALIBRATION_STATUSES
