import numpy as np
import scipy.linalg

from GTS.deprecated_files.geological_model_visualization import GeologicalModel


def get_shearzone_planes():
    """ Get the intersection points for all shearzones in
        the geological model.
    """
    gm = GeologicalModel()
    pts = gm.export_intersections()
    planes = list(pts.keys())

    shearzones = {k: {} for k in planes}

    for plane in planes:
        fp = FitPlane(pts[plane])
        shearzones[plane]["proj"] = fp.proj

        shearzones[plane]["n"] = fp.n

    return shearzones


class FitPlane:
    """ Fit a 2D plane to a cloud of 3D points.

    Computed quantities:
        plane (np.array 1x3): Equation of plane: a*X + b*Y + c = Z
        n (np.array 1x3): Unit normal vector of plane
        proj (np.array 3xn): Point cloud projected to the plane.
    """

    def __init__(self, p: np.ndarray):
        """
        p (np.ndarray): Array of points (3 x n).
        """
        assert p.shape[0] == 3
        self.p = p
        self.Np = p.shape[1]

        self.plane = self.fit_plane()
        n = np.hstack((self.plane[0:2], -1))
        self.n = n / np.linalg.norm(n)  # Normalized normal vector

        # Projections
        self.proj = self.project_p()

    def fit_plane(self):
        """ Fit point cloud to plane

        Plane given by a*X + b*Y + c = Z
        """
        p = self.p  # - np.mean(self.p, axis=1)  # Subtract mean for numerical accuracy
        A = np.c_[(p[0], p[1], np.ones(self.Np))]
        C, *_ = scipy.linalg.lstsq(A, p[2])  # coefficients
        # C = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(p[2]) # <-- Same as above

        return C

    def project_p(self):
        """ Project points to plane

        x_p = x_0 - (n . x_0) n
        """
        p = self.p.T
        l = np.atleast_2d(p.dot(self.n)).T
        return (p - l * self.n).T
