from pathlib import Path

import numpy as np

import porepy as pp
from GTS import BiotParameters, ISCBiotContactMechanics


def structured_grid_2_intx_fracs():
    """ Two orthogonal intersecting fractures"""
    nx = np.array([6, 6, 6])

    # fmt: off
    frac_pts = np.array(
        [[0, 4, 4, 0],
         [0, 0, 4, 4],
         [1, 1, 1, 1]]
    )
    frac_pts2 = np.array(
        [[2, 2, 2, 2],
         [0, 0, 4, 4],
         [0, 2, 2, 0]]
    )
    # fmt: on
    gb = pp.meshing.cart_grid([frac_pts, frac_pts2], nx=nx)
    return gb


def test_aperture_of_fracture_intersection():
    here = Path(__file__).parent / "simulations"

    options = {
        "folder_name": here,
        "shearzone_names": ["S1_1", "S3_1"],
        "bounding_box": {
            "xmin": 0,
            "ymin": 0,
            "zmin": 0,
            "xmax": 6,
            "ymax": 6,
            "zmax": 6,
        },
        "source_scalar_borehole_shearzone": None,
    }
    params = BiotParameters(**options)
    setup = ISCBiotContactMechanics(params)
    setup.gb = structured_grid_2_intx_fracs()
    setup._prepare_grid()
    setup.set_biot_parameters()
    setup.initial_biot_condition()

    # Set an aperture on the mortar grids
    gb = setup.gb
    s11 = setup.grids_by_name("S1_1")[0]
    s31 = setup.grids_by_name("S3_1")[0]
    de11 = gb.edge_props((s11, setup._nd_grid()))
    de31 = gb.edge_props((s31, setup._nd_grid()))
    mg11, mg31 = de11["mortar_grid"], de31["mortar_grid"]
    sgn11, sgn31 = mg11.sign_of_mortar_sides(3), mg31.sign_of_mortar_sides(3)
    u11 = sgn11 * np.ones_like(de11[pp.STATE][pp.ITERATE]["mortar_u"])
    u31 = sgn31 * np.ones_like(de31[pp.STATE][pp.ITERATE]["mortar_u"])
    de11[pp.STATE][pp.ITERATE]["mortar_u"] = u11 * 5
    de31[pp.STATE][pp.ITERATE]["mortar_u"] = u31 * 2

    # Get mechanical aperture in the intersection
    ix = gb.grids_of_dimension(1)[0]
    ap = setup.mechanical_aperture(ix, scaled=False, from_iterate=True)
    assert np.all(ap == 5 * 2), (
        "Disregarding shear, we expected the max jump to be 2 * 5, "
        "where 5 is each side of the u11 displacement"
    )
