"""
Tests for the camera intrinsic matrix computation.

The intrinsic matrix converts 3D points in a camera's coordinate system to 2D pixel
coordinates. It encodes the camera's focal length (in pixels) and principal point offset.
These tests verify that the intrinsic matrices are correctly constructed from the camera
hardware specs (focal length in mm, pixel size, crop offsets).
"""
import numpy as np
from helpers.utils_3d_reconstruction import CameraData


def test_intrinsic_matrix_shape():
    """Each camera should produce a 3x3 intrinsic matrix."""
    cams = CameraData(basic=True)
    for cam_name, K in cams.intrinsic_matrices.items():
        assert K.shape == (3, 3), f"{cam_name} intrinsic matrix should be 3x3"


def test_intrinsic_matrix_structure():
    """
    The intrinsic matrix should have the standard pinhole camera form:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    where fx/fy are focal lengths in pixels and cx/cy are principal point offsets.
    """
    cams = CameraData(basic=True)
    for cam_name, K in cams.intrinsic_matrices.items():
        # Off-diagonal elements (skew) should be zero
        assert K[0, 1] == 0.0, f"{cam_name}: skew should be zero"
        assert K[1, 0] == 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        # Bottom-right element should be 1
        assert K[2, 2] == 1.0, f"{cam_name}: K[2,2] should be 1.0"
        # Focal lengths should be positive
        assert K[0, 0] > 0, f"{cam_name}: fx should be positive"
        assert K[1, 1] > 0, f"{cam_name}: fy should be positive"


def test_side_camera_focal_length():
    """
    The side camera has a 16mm lens and 4.8um pixel size.
    Focal length in pixels = 16mm / 0.0048mm = 3333.33...
    """
    cams = CameraData(basic=True)
    K = cams.intrinsic_matrices["side"]
    expected_f = 16.0 / 4.8e-3
    assert np.isclose(K[0, 0], expected_f), f"Side fx should be {expected_f}"
    assert np.isclose(K[1, 1], expected_f), f"Side fy should be {expected_f}"


def test_principal_point_adjusted_for_crop():
    """
    The principal point (cx, cy) should be the full-frame principal point
    minus the crop offset. This accounts for the fact that DLC coordinates
    are relative to the cropped image, not the full sensor.
    """
    cams = CameraData(basic=True)
    specs = cams.specs["front"]
    K = cams.intrinsic_matrices["front"]

    expected_cx = specs["principal_point_x_px"] - specs["crop_offset_x"]
    expected_cy = specs["principal_point_y_px"] - specs["crop_offset_y"]

    assert np.isclose(K[0, 2], expected_cx), f"Front cx should be {expected_cx}"
    assert np.isclose(K[1, 2], expected_cy), f"Front cy should be {expected_cy}"


def test_all_three_cameras_present():
    """The CameraData class should define specs for side, front, and overhead."""
    cams = CameraData(basic=True)
    assert set(cams.intrinsic_matrices.keys()) == {"side", "front", "overhead"}
