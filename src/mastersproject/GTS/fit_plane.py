"""
Methods for fitting a plane to a point cloud.
Public methods:
plane_from_points(points: np.ndarray) -> np.ndarray:
    - For an arbitrary point cloud of n points in 3D space (3,n)
        fit a plane to these points by computing the least squares fit
        with respect to the best conditioned spatial direction.
        Uses fit_normal_to_points to compute the plane of best fit.
convex_hull(p: np.ndarray) -> np.ndarray:
    - For a point cloud of n points in 3D space (3,n) constrained to
        an arbitrarily oriented 2D plane, compute the 2D convex hull
        of the point cloud. The computed points naturally forms a
        subset of the original point set, ordered counter-clockwise.

Private methods:
fit_normal_to_points(points: np.ndarray) -> np.array:
    - Compute the plane of best fit to an arbitrary point cloud in
        3D space. Returns the normalized normal to the computed plane.
        See the public method documentation for additional details.
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull

import porepy as pp


def convex_hull(p: np.ndarray) -> np.ndarray:
    """Construct the convex hull of a planar set of 2D points

    Given a point cloud in a plane, compute its convex hull, and
    return the vertices in ccw order of this hull.

    Parameters:
        p (np.ndarray, 3 x n): Planar point cloud
    Returns:
        np.ndarray, 3 x m: Vertices of convex hull, sorted ccw.

    The procedure is as follows:
    * For a given point set, assert that they lie in a plane (or map them to one).
    (1) Map the points to the plane z = 0
    (2) Construct the convex hull (algorithm guarantees ccw ordering.
    (3) Rotate points back to original plane.

    The following methods from PorePy will be utilized directly:
    project_plane_matrix(...)
    """

    assert pp.geometry_property_checks.points_are_planar(p)

    # Move to mean
    mean = np.mean(p, axis=1)
    pm = np.subtract(p.T, mean).T  # Use these for calculations, then transform back

    # Rotation matrix
    R = pp.map_geometry.project_plane_matrix(pm, check_planar=False)

    pm_2d = np.dot(R, pm)[:2, :]

    # Compute convex hull
    hull = ConvexHull(pm_2d.T)
    vertices_pl = pm_2d[:, hull.vertices]  # Guaranteed to be ccw
    vertices_pl = np.vstack((vertices_pl, np.zeros(vertices_pl.shape[1])))

    # Rotate back
    vertices = np.dot(R.T, vertices_pl)

    # Translate
    vertices = np.add(vertices.T, mean).T

    return vertices


def fit_normal_to_points(points: np.ndarray) -> np.array:
    """Compute a normal from a collection of points.

    Source: http://www.ilikebigbits.com/2015_03_04_plane_from_points.html

    Parameters:
    points : np.ndarray (3,n)
        Array of points

    Returns
    normal : np.array (3,)
        Normalized normal vector to plane
    """
    assert (points.shape[0] == 3) & (points.shape[1] >= 3)
    pts = points.copy()
    N = pts.shape[1]

    # Calculate centroid and shift points
    centroid = np.atleast_2d(np.sum(pts, axis=1) / N).T
    pts = pts - centroid
    x, y, z = pts[0], pts[1], pts[2]

    # Compute the dot products, x*x, x*y, x*z, y*y, y*z, z*z
    xx = np.dot(x, x)
    xy = np.dot(x, y)
    xz = np.dot(x, z)
    yy = np.dot(y, y)
    yz = np.dot(y, z)
    zz = np.dot(z, z)

    # Compute the determinants
    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    det_max = max(det_x, det_y, det_z)
    if det_max <= 0.0:
        return None  # The points do not span a plane

    # Pick path with best conditioning
    if det_x == det_max:
        normal = np.array([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    elif det_y == det_max:
        normal = np.array([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    else:  # det_z == det_max
        normal = np.array([xy * yz - xz * yy, xy * xz - yz * xx, det_z])

    return normal / np.sqrt((normal ** 2).sum())  # Normalize


def plane_from_points(points: np.ndarray) -> np.ndarray:
    """Compute a plane for a given point cloud.

    Compute the fitted plane to a point cloud, returning
    the points projected to the plane

    Parameters:
    points : nd.ndarray (3, n)
        Array of n 3D points

    Returns:
    proj : nd.ndarray (3, n)
        Array of n 3D points, projected to the best fit plane

    """
    assert (points.shape[0] == 3) & (points.shape[1] >= 3), "Wrong input shape."

    N = points.shape[1]
    centroid = np.sum(points, axis=1) / N
    pointsR = (points.T - centroid).T
    normal = fit_normal_to_points(pointsR)
    assert np.isclose(np.sqrt((normal ** 2).sum()), 1)  # Assert normalized

    proj = centroid + pointsR.T - np.outer(np.dot(pointsR.T, normal), normal)

    # Relative error of projected points
    error = np.sqrt(((proj - points.T) ** 2).sum(axis=1))
    length = np.sqrt(((points.T) ** 2).sum(axis=1))
    rel_error = error / length
    logging.info(f"Sum of pointwise relative errors: {rel_error.sum():.4f}")

    return proj.T
