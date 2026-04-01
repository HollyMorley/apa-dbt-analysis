"""
Tests for the joint angle calculation used in gait feature extraction.

The angle calculation takes three 2D points and computes the angle at the middle
point (point2) between the vectors to point1 and point3. This is used to measure
joint angles (e.g. knee, ankle) from tracked body part positions during each stride.
"""
import numpy as np
from gait.GaitFeatureExtraction import FeatureExtractor


def make_extractor():
    """Create a minimal FeatureExtractor with dummy data (only need the method)."""
    # FeatureExtractor.__init__ expects (data, fps) but we only need the method,
    # so we bypass __init__ by creating an empty instance
    ext = object.__new__(FeatureExtractor)
    return ext


def test_right_angle():
    """Three points forming a 90-degree angle should return 90."""
    ext = make_extractor()
    # L-shape: point1 straight up, point2 at origin, point3 to the right
    # N.B. arrays must be float (as DLC coordinates always are) - integer arrays
    # trigger a numpy casting error in np.full_like due to int->float mismatch
    p1 = np.array([[0.0, 1.0]])
    p2 = np.array([[0.0, 0.0]])
    p3 = np.array([[1.0, 0.0]])
    angles = ext.calculate_angle_vectorized(p1, p2, p3)
    assert np.isclose(angles[0], 90.0, atol=1e-10)


def test_straight_line():
    """Three collinear points should give 180 degrees (straight limb)."""
    ext = make_extractor()
    p1 = np.array([[0.0, 0.0]])
    p2 = np.array([[1.0, 0.0]])
    p3 = np.array([[2.0, 0.0]])
    angles = ext.calculate_angle_vectorized(p1, p2, p3)
    assert np.isclose(angles[0], 180.0, atol=1e-10)


def test_acute_angle():
    """A 60-degree equilateral triangle setup."""
    ext = make_extractor()
    p1 = np.array([[1.0, 0.0]])
    p2 = np.array([[0.0, 0.0]])
    p3 = np.array([[0.5, np.sqrt(3) / 2]])
    angles = ext.calculate_angle_vectorized(p1, p2, p3)
    assert np.isclose(angles[0], 60.0, atol=1e-10)


def test_vectorized_multiple_points():
    """The function should handle multiple angle calculations at once."""
    ext = make_extractor()
    p1 = np.array([[0.0, 1.0], [0.0, 0.0]])
    p2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    p3 = np.array([[1.0, 0.0], [2.0, 0.0]])
    angles = ext.calculate_angle_vectorized(p1, p2, p3)
    assert np.isclose(angles[0], 90.0, atol=1e-10)
    assert np.isclose(angles[1], 180.0, atol=1e-10)


def test_coincident_points_return_nan():
    """If two points are the same, the angle is undefined and should be NaN."""
    ext = make_extractor()
    p1 = np.array([[0.0, 0.0]])
    p2 = np.array([[0.0, 0.0]])  # same as p1
    p3 = np.array([[1.0, 0.0]])
    angles = ext.calculate_angle_vectorized(p1, p2, p3)
    assert np.isnan(angles[0])
