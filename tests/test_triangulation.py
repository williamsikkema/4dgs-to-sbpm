import numpy as np

from face_mc_gs.calibration.load import Camera
from face_mc_gs.geometry.reprojection import project_points_world
from face_mc_gs.geometry.triangulation import triangulate_two_view


def test_triangulate_reproject():
    K = np.eye(3)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.1
    c0 = Camera("0", K, np.zeros(5), T0)
    c1 = Camera("1", K, np.zeros(5), T1)
    X = np.array([[0.0, 0.0, 5.0]])
    uv0 = project_points_world(X, c0)
    uv1 = project_points_world(X, c1)
    Xh = triangulate_two_view(uv0, uv1, c0, c1)
    assert np.linalg.norm(Xh - X) < 0.05
