import numpy as np

from face_mc_gs.calibration.load import Camera
from face_mc_gs.geometry.camera import pixel_to_ray_world
from face_mc_gs.geometry.reprojection import project_points_world


def test_ray_round_trip():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    T = np.eye(4)
    cam = Camera("0", K, np.zeros(5), T)
    X = np.array([[0.1, -0.2, 3.0]])
    uv = project_points_world(X, cam)
    C, d = pixel_to_ray_world(uv, cam)
    # point should lie on ray from camera center
    v = X[0] - C
    v = v / np.linalg.norm(v)
    assert abs(np.dot(v, d[0])) > 0.99
