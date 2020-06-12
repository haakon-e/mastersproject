import numpy as np
import porepy as pp

from GTS.deprecated_files.geological_model_visualization import GeologicalModel
from GTS.fit_plane import convex_hull
from GTS.deprecated_files.old_fit_plane import FitPlane


class GeologicalModelMethodTests:
    """ Test the individual methods in:
        geological_model_visualization.py
    """

    def __init__(self):
        """ Initialize the Geological Model. """
        self.gm = GeologicalModel()

    def test_drill_boreholes(self):
        """ Test accuracy of borehole data"""
        bh_coords = self.gm.BH_coordinates

        # INJ
        real_data = np.array(
            [
                [
                    66.2310000000289,
                    88.3990000000049,
                    32.817,
                    44.66,
                    146,
                    309.57,
                    -33.52,
                ],
                [
                    66.7870000000112,
                    90.2170000000042,
                    32.8810000000001,
                    44.8,
                    146,
                    332.280,
                    -43.65,
                ],
            ]
        )
        inj_data = bh_coords["INJ"]
        assert np.allclose(inj_data, real_data)


class TestFitPlaneMethods:
    """ Test point rotations"""

    def test_fit_plane(self):
        """ Test method that fits a plane to points"""

        # 1. Case: Simple plane defined exactly by [e1, e2, e3] orthogonal vectors.
        # Excpected solution: - X - Y + 1 = Z
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        fp = FitPlane(p)
        # print(fp.plane)
        assert np.allclose(fp.plane, np.array([-1, -1, 1]))

        # 2. Case: Fit horizontal plane
        p = np.array(
            [
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [-1, -1, -1, -1, 1, 1, 1, 1],
            ]
        )
        fp = FitPlane(p)
        assert np.allclose(fp.plane, np.array([0, 0, 0]))

        # 3. Case: Numerical example from StackExchange
        # See: https://math.stackexchange.com/a/1652383
        p = np.array(
            [
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [9, 14, 20, 11, 17, 23, 15, 20, 26],
            ]
        )
        fp = FitPlane(p)
        print(fp.plane)
        print("normal vector:", fp.n)

        p = np.array(
            [
                [1.72, -0.69, -2.43, -2.19, -0.15, 2.16, 2.99, 1.72, -0.69, -2.43],
                [2.3, 2.5, 1.35, -0.29, -1.18, -0.66, 0.89, 2.29, 2.5, 1.35],
                [-2.94, -3.94, -3.05, -0.95, 0.88, 1.03, -0.73, -2.97, -4.02, -3.11],
            ]
        )
        fp = FitPlane(p)
        print("Last case")
        print("ax+by+c=z: ", fp.plane)
        print("normal vector:", fp.n)
        # We computed a*X + b*Y + c = Z
        # Here we validate against a0*X + b0*Y + c0*Z = 1
        # So we compute
        pl = fp.plane
        C0 = np.array([-pl[0], -pl[1], 1] / pl[2])
        print("ax+by+cz = 1: ", C0)

    def test_projection_matrix(self):
        """ Test externally loaded projection matrix."""

        # Test identity projection matrix.
        a = np.array([[1, 2, 3], [10, 30, -10], [0, 0, 0]])
        r = pp.map_geometry.project_plane_matrix(a, check_planar=False)
        true_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.allclose(r, true_r)

    def test_convex_hull(self):
        """ Test the method for constructing convex hulls.

        Including rotations, do-nothing hulls, and complex hulls"""

        # test do-nothing hull
        pts = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
        vertices = convex_hull(pts)
        assert np.allclose(pts, vertices)

        # Test remove one point hull
        pts2 = np.vstack((pts.T, np.array((0.5, 0.5, 0.5)))).T
        vertices = convex_hull(pts2)
        assert np.allclose(pts, vertices)


def __main__():
    # Geological model tests
    g = GeologicalModelMethodTests()
    g.test_drill_boreholes()

    # Fit plane tests
    p = TestFitPlaneMethods()
    p.test_fit_plane()
