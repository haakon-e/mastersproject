import numpy as np
import porepy as pp


def structured_grid_2_frac_vertical(length_scale: float):
    nx = np.array([20, 20, 20])
    physdims = np.array([300, 300, 300])

    # fmt: off
    frac_pts1 = np.array(
        [[0, 300, 300, 0],
         [150, 150, 150, 150],
         [0, 0, 150, 150]])

    frac_pts2 = np.array(
        [[0, 300, 300, 0],
         [150, 150, 150, 150],
         [0, 0, 150, 150]])
    # fmt: on
    gb = pp.meshing.cart_grid(
        [frac_pts1, frac_pts2],
        nx=nx,
        physdims=physdims / length_scale,
    )
    return gb
