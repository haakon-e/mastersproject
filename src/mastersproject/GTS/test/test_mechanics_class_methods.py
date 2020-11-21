import logging

import numpy as np

import GTS as gts
import GTS.test.util as test_util
import pendulum
import porepy as pp
from mastersproject.util.logging_util import trace

logger = logging.getLogger(__name__)


def test_mechanical_boundary_conditions():
    """Test that mechanical boundary conditions are correctly set"""

    this_method = test_mechanical_boundary_conditions.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")

    # Create setup with no fractures
    params = {"shearzone_names": None}
    setup = test_util.prepare_setup(
        model=gts.ContactMechanicsISC,
        path_head=f"{this_method}/{now_as_YYMMDD}/test_1",
        params=params,
    )

    gb: pp.GridBucket = setup.gb
    g: pp.Grid = gb.grids_of_dimension(3)[0]
    data = gb.node_props(g)

    mech_params: dict = data[pp.PARAMETERS][setup.mechanics_parameter_key]
    # Get the mechanical bc values
    mech_bc = mech_params["bc_values"].reshape((3, -1), order="F")
    mech_bc_type: pp.BoundaryConditionVectorial = mech_params["bc"]

    all_bf, east, west, north, south, top, bottom = setup.domain_boundary_sides(g)
    # --- TESTS ---

    # 1. Test that mechanical BCs have the right type
    #   We want neumann on all faces except setup.faces_to_fix(),
    #   which should be on the bottom of the grid.
    dir_faces = setup.faces_to_fix(g)
    neu_faces = np.setdiff1d(all_bf, dir_faces)
    assert np.all(mech_bc_type.is_dir[:, dir_faces])
    assert np.all(mech_bc_type.is_neu[:, neu_faces])

    # 2.a. Test that Neumann BCs all point into the domain.
    #   The physical conditions at Grimsel Test Site implies
    #   a compressive boundary traction on all faces.

    # We expect the dirichlet faces to be on bottom. Remove them
    # for the sake of testing neumann conditions
    btm = np.setdiff1d(np.where(bottom)[0], dir_faces)

    # West
    assert np.all(mech_bc[0, west] > 0)
    # East
    assert np.all(mech_bc[0, east] < 0)
    # North
    assert np.all(mech_bc[1, north] < 0)
    # South
    assert np.all(mech_bc[1, south] > 0)
    # Top
    assert np.all(mech_bc[2, top] < 0)
    # Bottom
    assert np.all(mech_bc[2, btm] > 0)

    # 2.b. Test that Dirichlet BCs are zero
    np.allclose(mech_bc[:, dir_faces], 0)

    # 3. Test linear increase of mechanical boundary conditions (lithostatic conditions)


@trace(logger)
def test_decomposition_of_stress(setup="normal_shear"):
    """Test the solutions acquired when decomposing stress to
    purely compressive and purely rotational components.

    --- Setup ---
    A: 1. Get the stress tensor and split in diagonal and off-diagonal components.
    B: 1. Get the stress tensor and split in hydrostatic and deviatoric components.
    --
    2. Acquire solutions for each split tensor and the full tensor separately (3 cases).
    3. Compare solutions quantitatively (linearity of solution)
        and qualitatively (Paraview)

    Parameters
    ----------
    setup : str : {'normal_shear', 'hydrostatic'}
        Type of setup.
        'normal_shear' simply separates the normal and shear components
        'hydrostatic' separates hydrostatic and deviatoric stresses
    """

    # Import stress tensor
    stress = gts.stress_tensor()

    if setup == "normal_shear":
        # Get normal and shear stresses
        normal_stress = np.diag(np.diag(stress).copy())
        shear_stress = stress - normal_stress
        fname_n = "normal_stress"
        fname_s = "shear_stress"
    elif setup == "hydrostatic":
        # Get hydrostatic and deviatoric stresses
        hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
        normal_stress = np.diag(hydrostatic)
        shear_stress = stress - normal_stress
        fname_n = "hydrostatic_stress"
        fname_s = "deviatoric_stress"
    else:
        raise ValueError(f"Did not recognise input setup={setup}")

    # 1. Full stress tensor
    this_method_name = test_decomposition_of_stress.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/no_gravity/{setup}"
    gravity = False
    no_shearzones = None
    params = {
        "shearzone_names": no_shearzones,
        "stress": stress,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
    }
    setup = test_util.prepare_setup(
        model=gts.ContactMechanicsISC,
        path_head=f"{_folder_root}",
        params=params,
        prepare_simulation=False,
        setup_loggers=True,
    )
    setup.create_grid()

    # -- Safely copy the grid generated above --
    # Ensures the implicit in-place variable changes
    #   across grid_buckets (gb.copy() is insufficient)
    # Get the grid_bucket .msh path
    path_to_gb_msh = f"{setup.viz_folder_name}/gmsh_frac_file.msh"
    # Re-create the grid buckets
    gb_n = pp.fracture_importer.dfm_from_gmsh(
        path_to_gb_msh, dim=3, network=setup._network
    )
    gb_s = pp.fracture_importer.dfm_from_gmsh(
        path_to_gb_msh, dim=3, network=setup._network
    )

    # 1. Pure normal stress / hydrostatic stress
    params_n = params.update({"stress": normal_stress})
    setup_n = test_util.prepare_setup(
        model=gts.ContactMechanicsISC,
        path_head=f"{_folder_root}/{fname_n}",
        params=params_n,
        prepare_simulation=False,
        setup_loggers=False,
    )
    setup_n.set_grid(gb_n)

    # 2. Pure shear stress / deviatoric stress
    params_s = params.update({"stress": shear_stress})
    setup_s = test_util.prepare_setup(
        model=gts.ContactMechanicsISC,
        path_head=f"{_folder_root}/{fname_s}",
        params=params_s,
        prepare_simulation=False,
        setup_loggers=False,
    )
    setup_s.set_grid(gb_s)

    # --- Run simulations ---
    params_nl = {}  # Use default Newton solver parameters

    # 1. Full stress tensor
    pp.run_stationary_model(setup, params_nl)

    # 2. Pure normal stress
    pp.run_stationary_model(setup_n, params_nl)

    # 3. Pure shear stress
    pp.run_stationary_model(setup_s, params_nl)

    # --- Compare results ---
    def get_u(_setup):
        gb = _setup.gb
        g = gb.grids_of_dimension(3)[0]
        d = gb.node_props(g)
        u = d["state"]["u"].reshape((3, -1), order="F")
        return u

    return get_u(setup), get_u(setup_n), get_u(setup_s), [setup, setup_n, setup_s]
