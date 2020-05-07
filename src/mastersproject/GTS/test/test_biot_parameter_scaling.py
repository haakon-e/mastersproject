import logging
from typing import (  # noqa
    Any,
    Coroutine,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import os
from pathlib import Path

import porepy as pp
import numpy as np
from porepy.models.contact_mechanics_model import ContactMechanics
import pendulum

import GTS as gts
from .test_create_grid import test_create_grid

from util.logging_util import __setup_logging
import src.mastersproject.GTS.test.util as test_util

logger = logging.getLogger(__name__)


def test_biot_parameter_scaling(**kw):
    """ Test scaling of parameters in Biot

    No methods are solved. The parameters are simply checked
    for consistency
    """
    # --- THIS METHOD NO LONGER WORKS OVER A GRID BECAUSE LENGTH SCALING GIVES SLIGHTLY DIFFERENT MESHES ---
    _this_file = Path(os.path.abspath(__file__)).parent
    _results_path = _this_file / "results/test_biot_parameter_scaling/default"
    _results_path.mkdir(parents=True, exist_ok=True)  # Create path if not exists
    __setup_logging(_results_path)
    logger.info(f"Path to results: {_results_path}")

    # --- DOMAIN ARGUMENTS ---
    params = {
        'mesh_args':
            {'mesh_size_frac': 10, 'mesh_size_min': .1*10, 'mesh_size_bound': 6*10},
        'bounding_box':
            {'xmin': -20, 'xmax': 80, 'ymin': 50, 'ymax': 150, 'zmin': -25, 'zmax': 75},
        'shearzone_names':
            ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"],
        'folder_name':
            _results_path,
        'solver':
            'direct',
        'stress':
            gts.isc_modelling.stress_tensor(),
        'source_scalar_borehole_shearzone':
            {'borehole': 'INJ1', 'shearzone': 'S1_1'},
        # Initially: We calculate unscaled variables.
        'length_scale':
            1,
        'scalar_scale':
            1,
    }
    setup = gts.ContactMechanicsBiotISC(params)
    setup.create_grid()
    setup.well_cells()
    setup.set_parameters()

    # Save copies of the original data on the 3D grid
    gb = setup.gb
    g = gb.grids_of_dimension(3)[0]
    data = gb.node_props(g)
    mech_params = data['parameters']['mechanics']
    flow_params = data['parameters']['flow']
    mech = {
        'bc_values':
            mech_params['bc_values'].copy(),
        'source':
            mech_params['source'].copy(),
        'fourth_order_tensor':
            mech_params['fourth_order_tensor'].copy(),
        'biot_alpha':
            mech_params['biot_alpha'],  # Float
    }

    flow = {
        'bc_values':
            flow_params['bc_values'].copy(),
        'mass_weight':
            flow_params['mass_weight'],  # Float
        'source':
            flow_params['source'].copy(),
        'second_order_tensor':
            flow_params['second_order_tensor'].copy(),
        'biot_alpha':
            flow_params['biot_alpha'],  # Float
    }

    # Scale grid:
    setup.scalar_scale = kw.get('ss', 1 * pp.GIGA)
    setup.length_scale = kw.get('ls', 100)
    setup.create_grid(overwrite_grid=True)
    ss = setup.scalar_scale
    ls = setup.length_scale

    # Recompute parameters
    setup.prepare_simulation()

    # Mimic NewtonSolver:
    setup.before_newton_iteration()

    # Check size of entries in matrix A.
    A, b = setup.assembler.assemble_matrix_rhs()
    logger.info("------------------------------------------")
    logger.info(f"Max element in A {np.max(np.abs(A)):.2e}")
    logger.info(f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum.")

    # Find the new parameter dictionaries
    dim = 3
    gb = setup.gb
    g = gb.grids_of_dimension(dim)[0]
    data = gb.node_props(g)
    scaled_mech = data['parameters']['mechanics']
    scaled_flow = data['parameters']['flow']
    all_bf, east, west, north, south, top, bottom = setup.domain_boundary_sides(g)

    # --- Do the comparisons: ---
    test_cell = 0
    logger.info(f"scalar_scale={setup.scalar_scale:.2e}. length_scale={setup.length_scale:.2e}")
    # - FLOW -
    # Permeability [m2]
    #   k_scl = k * scalar_scale / length_scale ** 2
    k = flow['second_order_tensor'].values[0, 0, :]
    k_scl = scaled_flow['second_order_tensor'].values[0, 0, :]
    logger.info(f"unscaled k/mu={k[test_cell]:.2e}")
    logger.info(f"Scaled k/mu={k_scl[test_cell]:.2e}")
    assert np.allclose(k * ss / ls ** 2, k_scl), "k_scl = k * scalar_scale / length_scale ** 2"

    # Mass weight / Effective Storage term (possibly aperture scaled) [1/Pa]
    #   mw_scl = mw * scalar_scale
    mw = flow['mass_weight']
    mw_scl = scaled_flow['mass_weight']
    logger.info(f'mass_weight={mw:.2e}')
    logger.info(f'mass_weight scaled={mw_scl:.2e}')
    assert np.allclose(mw * ss, mw_scl), "mw_scl = mw * scalar_scale"

    # Source [m ** dim / s]
    #   fs_scl = fs * length_scale ** dim
    fs = flow['source'][test_cell]
    fs_scl = scaled_flow['source'][test_cell]
    logger.info(f"Unscaled flow source={fs:.2e}")
    logger.info(f"Scaled flow source={fs_scl:.2e}")
    assert np.allclose(fs / ls ** dim, fs_scl), "fs_scl = fs / length_scale ** dim"

    # Boundary conditions (FLOW)
    #   Dirchlet [Pa]
    #   fd_scl = fd / scalar_scale
    fd = flow['bc_values']
    fd_scl = scaled_flow['bc_values']
    logger.info(f"bc flow={fd[all_bf][0]:.2e}")
    logger.info(f"bc flow scaled={fd_scl[all_bf][0]:.2e}")
    assert np.allclose(fd / ss, fd_scl), "fd_scl = fd / scalar_scale"
    #   Neumann [m2 / s] (integrated across 2D surfaces)


    # Biot alpha should remain unchanged
    assert flow['biot_alpha'] == scaled_flow['biot_alpha']

    # - MECHANICS -
    # Mu and Lambda [Pa]
    #   m_scl = m / scalar_scale AND lm_scl = lm / scalar_scale
    mu = mech['fourth_order_tensor'].mu
    mu_scl = scaled_mech['fourth_order_tensor'].mu
    lmbda = mech['fourth_order_tensor'].lmbda
    lmbda_scl = scaled_mech['fourth_order_tensor'].lmbda

    logger.info(f"mu={mu[test_cell]:.2e}. mu scaled = {mu_scl[test_cell]:.2e}")
    logger.info(f"lambda={lmbda[test_cell]:.2e}. lambda scaled={lmbda_scl[test_cell]:.2e}")
    assert np.allclose(mu / ss, mu_scl)
    assert np.allclose(lmbda / ss, lmbda_scl)

    # Mechanics source [Pa m2] (integrated over 3D volume)
    #   In the code: ms_scl = ms * length_scale / scalar_scale (assuming integration is scaled)
    #   ms_scl = ms / (scalar_scale * length_scale ** 2)
    ms = mech['source'].reshape((3, -1), order='F')
    ms_scl = scaled_mech['source'].reshape((3, -1), order='F')
    logger.info(f"Mechanics source={ms[:, test_cell]}")
    logger.info(f"Mechanics source scaled={ms_scl[:, test_cell]}")
    assert np.allclose(ms / ls, ms_scl)


    # Boundary conditions (MECHANICS)
    #   Neumann [Pa m2] (integrated across 2D surfaces)
    #   Note: In the code, we divide by scalar_scale. length_scale**2 is incorporated by pre-scaled grid.
    #   mn_scl = mn / ( scalar_scale * length_scale**(dim-1) )
    mn = mech['bc_values'].reshape((3, -1), order='F')
    mn_scl = scaled_mech['bc_values'].reshape((3, -1), order='F')
    logger.info(f"mech neumann (3 faces on east) =\n{mn[:, east][:, :3]}")
    logger.info(f"mech neumann scaled (3 faces on east) =\n{mn_scl[:, east][:, :3]}")
    assert np.allclose(mn[:, all_bf] / (ss * ls ** (dim - 1)), mn_scl[:, all_bf])

    return setup, mech, flow, scaled_mech, scaled_flow


def test_biot_condition_number(
        test_name: str,
        ls: float,
        ss: float,
        shearzones: List[str] = None,
):
    """ Test scaling of parameters in Biot

    Note: No problem is solved. Condition number is simply computed.

    Parameters
    ----------
    test_name : str
        name of this test
    ls, ss : float
        length scale and scalar scale, respectively
    shearzones : List[str] (Default: None)
        list of shearzones
    """

    # Custom parameters
    params = {
        "shearzone_names": shearzones,
        "length_scale": ls,
        "scalar_scale": ss,
    }

    # Storage folder
    this_method_name = test_biot_condition_number.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    shearzone_str = str(shearzones) if not shearzones else "-".join(shearzones)
    scale_str = f"ss{ss}_ls{ls}"
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/{test_name}/{shearzone_str}/{scale_str}"

    params = test_util.prepare_params(
        path_head=_folder_root,
        params=params,
        setup_loggers=True,
    )

    # Set up model class
    setup = gts.ContactMechanicsBiotISC(params)
    setup.prepare_simulation()

    # Mimic NewtonSolver: --> This can probably be removed
    setup.before_newton_iteration()

    # Check size of entries in matrix A.
    A, b = setup.assembler.assemble_matrix_rhs()
    logger.info("------------------------------------------")
    logger.info(f"Max element in A {np.max(np.abs(A)):.2e}")
    logger.info(f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum.")
    logger.info(f"Length scale: {setup.length_scale:.1e}. Scalar scale: {setup.scalar_scale:.1e}.")
    return setup


def test_biot_solution_scaling(
        test_name: str,
        ls: float,
        ss: float
):
    """ Test consistency of solution subject to
    scalar scaling and length scaling of biot.

    Simplifications:
    * hydrostatic mechanical stress
    * no mechanical gravity term
    * No shear zones

    Parameters
    ----------
    test_name : str
        name of this test
    ls, ss : float
        length scale and scalar scale, respectively
    """

    # 1. Prepare parameters
    stress = gts.isc_modelling.stress_tensor()
    # We set up hydrostatic stress
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    no_shearzones = None
    gravity = False  # No gravity effects

    base_params = {
        "stress": stress,
        "shearzone_names": no_shearzones,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
        "length_scale": 1,
        "scalar_scale": 1,
    }

    # Storage folder
    this_method_name = test_biot_solution_scaling.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _master_root = f"{this_method_name}/{now_as_YYMMDD}/{test_name}"
    _folder_root = f"{_master_root}/unscaled"

    #
    # 2. Setup and run test
    params = test_util.prepare_params(
        path_head=_folder_root,
        params=base_params.copy(),
        setup_loggers=True)
    setup = gts.ContactMechanicsBiotISC(params=params)

    nl_params = {}  # Default Newton Iteration parameters
    pp.run_time_dependent_model(setup, params=nl_params)

    #
    # 3. Setup scaled problem and solve
    scale_str = f"ss{ss}_ls{ls}"
    _folder_root = f"{_master_root}/{scale_str}"

    params = base_params
    params["length_scale"] = ls
    params["scalar_scale"] = ss
    params = test_util.prepare_params(path_head=_folder_root, params=params.copy(), setup_loggers=False)
    scaled_setup = gts.ContactMechanicsBiotISC(params=params)
    pp.run_time_dependent_model(scaled_setup, params=nl_params)

    #
    # Extract useful solutions

    # 1. Approximate the mapping of variables from the unscaled to the scaled grid

    return setup, scaled_setup



def test_param_scaling_on_regular_grid():
    """ This test intends to verify scaling of variables and solutions
    by constructing a simple cartesian grid, scale the parameters, and check output.

    If the discretization method is consistent (not necessarily accurate), we should get
    comparable results on different length and scalar scales.
    """

    this_method_name = test_param_scaling_on_regular_grid.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/test_1"

    params1 = {
        "mesh_args":
            {'mesh_size_frac': 10, 'mesh_size_min': 10, 'mesh_size_bound': 10},
        "bounding_box":
            {'xmin': 0, 'xmax': 10,
             'ymin': 0, 'ymax': 10,
             'zmin': 0, 'zmax': 10},
        "shearzone_names":
            ["SZ_1"],
        "length_scale":
            1,
        "scalar_scale":
            1,
    }

    setup = test_util.prepare_setup(
        model=gts.ContactMechanicsBiotISC,
        path_head=_folder_root,
        params=params1,
        prepare_simulation=False,
        setup_loggers=True,
    )

#    gb =

