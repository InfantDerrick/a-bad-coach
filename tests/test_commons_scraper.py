from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scrape_commons_courts.py"
SPEC = importlib.util.spec_from_file_location("scrape_commons_courts", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonsScraperTests(unittest.TestCase):
    def test_strip_html(self) -> None:
        self.assertEqual(MODULE.strip_html("<b>Jane</b> &amp; John"), "Jane & John")

    def test_difference_hash_is_stable(self) -> None:
        image = np.zeros((80, 90, 3), dtype=np.uint8)
        image[:, 45:] = 255
        self.assertEqual(MODULE.difference_hash(image), MODULE.difference_hash(image.copy()))

    def test_line_score_finds_court_like_lines(self) -> None:
        image = np.full((600, 800, 3), (60, 120, 80), dtype=np.uint8)
        for y in (120, 260, 430, 540):
            cv2.line(image, (50, y), (750, y), (255, 255, 255), 5)
        cv2.line(image, (180, 50), (80, 590), (255, 255, 255), 5)
        cv2.line(image, (620, 50), (760, 590), (255, 255, 255), 5)
        self.assertGreaterEqual(MODULE.line_score(image), 3)


if __name__ == "__main__":
    unittest.main()
