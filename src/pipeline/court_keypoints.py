"""Canonical badminton-court landmarks and annotation validation.

Coordinates use metres on a doubles court. ``y=0`` is the far baseline and
``y=13.4`` is the near baseline from the camera's point of view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
NET_Y_M = COURT_LENGTH_M / 2
SHORT_SERVICE_OFFSET_M = 1.98
DOUBLES_LONG_SERVICE_OFFSET_M = 0.76


@dataclass(frozen=True)
class CourtLandmark:
    name: str
    label: str
    court_x_m: float
    court_y_m: float


LANDMARKS: tuple[CourtLandmark, ...] = (
    CourtLandmark("far_baseline_left", "Far baseline · left outer", 0.0, 0.0),
    CourtLandmark("far_baseline_right", "Far baseline · right outer", COURT_WIDTH_M, 0.0),
    CourtLandmark("far_long_service_left", "Far doubles service · left outer", 0.0, DOUBLES_LONG_SERVICE_OFFSET_M),
    CourtLandmark("far_long_service_right", "Far doubles service · right outer", COURT_WIDTH_M, DOUBLES_LONG_SERVICE_OFFSET_M),
    CourtLandmark("far_short_service_left", "Far short service · left outer", 0.0, NET_Y_M - SHORT_SERVICE_OFFSET_M),
    CourtLandmark("far_short_service_center", "Far short service · center", COURT_WIDTH_M / 2, NET_Y_M - SHORT_SERVICE_OFFSET_M),
    CourtLandmark("far_short_service_right", "Far short service · right outer", COURT_WIDTH_M, NET_Y_M - SHORT_SERVICE_OFFSET_M),
    CourtLandmark("net_left", "Net floor line · left outer", 0.0, NET_Y_M),
    CourtLandmark("net_right", "Net floor line · right outer", COURT_WIDTH_M, NET_Y_M),
    CourtLandmark("near_short_service_left", "Near short service · left outer", 0.0, NET_Y_M + SHORT_SERVICE_OFFSET_M),
    CourtLandmark("near_short_service_center", "Near short service · center", COURT_WIDTH_M / 2, NET_Y_M + SHORT_SERVICE_OFFSET_M),
    CourtLandmark("near_short_service_right", "Near short service · right outer", COURT_WIDTH_M, NET_Y_M + SHORT_SERVICE_OFFSET_M),
    CourtLandmark("near_long_service_left", "Near doubles service · left outer", 0.0, COURT_LENGTH_M - DOUBLES_LONG_SERVICE_OFFSET_M),
    CourtLandmark("near_long_service_right", "Near doubles service · right outer", COURT_WIDTH_M, COURT_LENGTH_M - DOUBLES_LONG_SERVICE_OFFSET_M),
    CourtLandmark("near_baseline_left", "Near baseline · left outer", 0.0, COURT_LENGTH_M),
    CourtLandmark("near_baseline_right", "Near baseline · right outer", COURT_WIDTH_M, COURT_LENGTH_M),
)

LANDMARK_BY_NAME = {landmark.name: landmark for landmark in LANDMARKS}


def schema_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": landmark.name,
            "label": landmark.label,
            "court_x_m": landmark.court_x_m,
            "court_y_m": landmark.court_y_m,
        }
        for landmark in LANDMARKS
    ]


def validate_annotation(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload.get("image"), str) or not payload["image"]:
        errors.append("image must be a non-empty string")
    width = payload.get("image_width")
    height = payload.get("image_height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        errors.append("image_width and image_height must be positive integers")
        width = height = 0
    points = payload.get("points")
    if not isinstance(points, dict):
        return errors + ["points must be an object keyed by landmark name"]
    unknown = set(points) - set(LANDMARK_BY_NAME)
    if unknown:
        errors.append(f"unknown landmark names: {sorted(unknown)}")
    for landmark in LANDMARKS:
        if landmark.name not in points:
            errors.append(f"missing landmark: {landmark.name}")
            continue
        point = points[landmark.name]
        if point is None:
            continue
        if not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, (int, float)) for value in point):
            errors.append(f"{landmark.name} must be [x, y] or null")
            continue
        if not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height):
            errors.append(f"{landmark.name} is outside the image")
    if payload.get("complete") is not True:
        errors.append("complete must be true before an annotation can train a model")
    return errors


def fit_homography_from_landmarks(
    points: dict[str, list[float] | None],
    ransac_threshold_px: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit image→court geometry from any sufficient visible landmark subset.

    A full court is not required. Four or more visible, non-collinear named
    intersections are enough to estimate a projective transform, after which
    missing court boundaries may lie outside the image.
    """
    image_points: list[list[float]] = []
    court_points: list[list[float]] = []
    names: list[str] = []
    for landmark in LANDMARKS:
        point = points.get(landmark.name)
        if point is None:
            continue
        image_points.append([float(point[0]), float(point[1])])
        court_points.append([landmark.court_x_m, landmark.court_y_m])
        names.append(landmark.name)
    if len(image_points) < 4:
        raise ValueError(f"Need at least four visible named landmarks; found {len(image_points)}")
    image_array = np.asarray(image_points, dtype=np.float32)
    court_array = np.asarray(court_points, dtype=np.float32)
    if cv2.contourArea(cv2.convexHull(court_array)) < 0.25:
        raise ValueError("Visible landmarks are too close to one line to fit a stable homography")
    court_to_image, inlier_mask = cv2.findHomography(
        court_array,
        image_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
    )
    if court_to_image is None:
        raise ValueError("Homography fitting failed")
    image_to_court = np.linalg.inv(court_to_image)
    reprojected = cv2.perspectiveTransform(court_array.reshape(-1, 1, 2), court_to_image).reshape(-1, 2)
    errors = np.linalg.norm(reprojected - image_array, axis=1)
    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.ones(len(errors), dtype=bool)
    diagnostics = {
        "visible_landmarks": len(image_points),
        "inlier_landmarks": int(inliers.sum()),
        "inlier_names": [name for name, is_inlier in zip(names, inliers) if is_inlier],
        "reprojection_rmse_px": float(np.sqrt(np.mean(errors[inliers] ** 2))) if np.any(inliers) else float("inf"),
        "max_reprojection_error_px": float(errors[inliers].max()) if np.any(inliers) else float("inf"),
    }
    return image_to_court, diagnostics
