import os
import logging
from typing import (
    Callable,
    List,
    Mapping,
    Tuple,
    Type,
    Union,
    Dict,
)
from pathlib import Path
from pprint import pformat

from pydantic import BaseModel
import pendulum
import porepy as pp
import numpy as np
from porepy.models.contact_mechanics_model import ContactMechanics

# GTS methods
import GTS as gts

# Refinement
from refinement import gb_coarse_fine_cell_mapping
from refinement.convergence import grid_error
from refinement import refine_mesh

# --- LOGGING UTIL ---
try:
    from src.mastersproject.util.logging_util import timer, trace, __setup_logging
except ImportError:
    from util.logging_util import timer, trace, __setup_logging

logger = logging.getLogger(__name__)


@trace(logger)
def run_biot_model(
        *,
        viz_folder_name: str = None,
        mesh_args: Mapping[str, int] = None,
        bounding_box: Mapping[str, int] = None,
        shearzone_names: List[str] = None,
        source_scalar_borehole_shearzone: Mapping[str, str] = None,
        length_scale: float = None,
        scalar_scale: float = None,
):
    """ Send all initialization parameters to contact mechanics biot class

        Parameters
        ----------
        viz_folder_name : str
            Absolute path to folder where grid and results will be stored
            Default: /home/haakon/mastersproject/src/mastersproject/GTS/isc_modelling/results/default
        mesh_args : Mapping[str, int]
            Arguments for meshing of domain.
            Required keys: 'mesh_size_frac', 'mesh_size_min, 'mesh_size_bound'
        bounding_box : Mapping[str, int]
            Bounding box of domain
            Required keys: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
        shearzone_names : List[str]
            Which shear-zones to include in simulation
        source_scalar_borehole_shearzone : Mapping[str, str]
            Which borehole and shear-zone intersection to do injection in.
            Required keys: 'shearzone', 'borehole'
        length_scale, scalar_scale : float : Optional
            Length scale and scalar variable scale.
        """
    params = {
        "folder_name": viz_folder_name,
        "mesh_args": mesh_args,
        "bounding_box": bounding_box,
        "shearzone_names": shearzone_names,
        "source_scalar_borehole_shearzone": source_scalar_borehole_shearzone,
        "length_scale": length_scale,
        "scalar_scale": scalar_scale,
    }

    setup = run_abstract_model(
        model=gts.ContactMechanicsBiotISC,
        run_model_method=pp.run_time_dependent_model,
        params=params,
    )

    return setup


@trace(logger)
def run_biot_gts_model(params):
    """ Set up and run biot model with
    an initialization run and a main run.
    """

    setup = run_abstract_model(
        model=gts.ContactMechanicsBiotISC,
        run_model_method=gts_biot_model,
        params=params,
    )

    return setup


@trace(logger)
def run_mechanics_model(
        *,
        viz_folder_name: str = None,
        mesh_args: Mapping[str, int] = None,
        bounding_box: Mapping[str, int] = None,
        shearzone_names: List[str] = None,
        length_scale: float = None,
        scalar_scale: float = None,
):
    """ Send all initialization parameters to contact mechanics class

    Parameters
    ----------
    viz_folder_name : str
        Absolute path to folder where grid and results will be stored
        Default: /home/haakon/mastersproject/src/mastersproject/GTS/isc_modelling/results/default
    mesh_args : Mapping[str, int]
        Arguments for meshing of domain.
        Required keys: 'mesh_size_frac', 'mesh_size_min, 'mesh_size_bound'
    bounding_box : Mapping[str, int]
        Bounding box of domain
        Required keys: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
    shearzone_names : List[str]
        Which shear-zones to include in simulation
    length_scale, scalar_scale : float : Optional
            Length scale and scalar variable scale.
    """
    params = {
        "folder_name": viz_folder_name,
        "mesh_args": mesh_args,
        "bounding_box": bounding_box,
        "shearzone_names": shearzone_names,
        "length_scale": length_scale,
        "scalar_scale": scalar_scale,
    }

    setup = run_abstract_model(
        model=gts.ContactMechanicsISC,
        run_model_method=pp.run_stationary_model,
        params=params,
    )

    return setup


def gts_biot_model(setup, params):
    """ Setup for time-dependent model run at Grimsel Test Site

    Usually called by run_abstract_model if this method is supplied
    as the argument 'run_model'.
    """

    # Initialization phase
    pp.run_time_dependent_model(setup=setup, params=params)
    logger.info(
        f"Initial simulation complete. Exporting solution. Time: {pendulum.now().to_atom_string()}"
    )
    # Stimulation phase
    logger.info(f"Starting stimulation phase at time: {pendulum.now().to_atom_string()}")
    setup.prepare_main_run()
    logger.info("Setup complete. Starting time-dependent simulation")
    pp.run_time_dependent_model(setup=setup, params=params)


def run_abstract_model(
        model: Type[ContactMechanics],
        run_model_method: Callable,
        params: dict = None,
        newton_params: dict = None,
):
    """ Set up and run an abstract model

    Parameters
    ----------
    model : Type[ContactMechanics]
        Which model to run
        Only tested for subclasses of ContactMechanicsISC
    run_model_method : Callable
        Which method to run model with
        Typically pp.run_stationary_model or pp.run_time_dependent_model
    params : dict (Default: None)
        Any non-default parameters to use
    newton_params : dict (Default: None)
        Any non-default newton solver parameters to use
    """

    # -------------------
    # --- SETUP MODEL ---
    # -------------------
    params = SetupParams(
        **params,
    ).dict()

    setup = model(params=params)

    # -------------------------
    # --- SOLVE THE PROBLEM ---
    # -------------------------
    default_options = {  # Parameters for Newton solver.
        "max_iterations": 40,
        "nl_convergence_tol": 1e-6,
        "nl_divergence_tol": 1e5,
    }
    if not newton_params:
        newton_params = {}
    default_options.update(newton_params)
    logger.info(f"Options for Newton solver: \n {default_options}")
    logger.info("Setup complete. Starting simulation")

    run_model_method(setup=setup, params=default_options)

    logger.info(f"Simulation complete. Exporting solution. Time: {pendulum.now().to_atom_string()}")

    return setup


def prepare_directories(head, date=True, root=None, **kwargs):
    # --------------------------------------------------
    # --- DEFAULT FOLDER AND FILE RELATED PARAMETERS ---
    # --------------------------------------------------

    # root of modelling results: i.e. ~/mastersproject/src/mastersproject/GTS/isc_modelling/results
    _root = Path(os.path.abspath(__file__)).parent / "results"
    root = root if root else Path(_root)
    # today's date
    date = pendulum.now().format("YYMMDD") if date else ""
    # build full path
    path = root / date / head
    # create the directory
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


class SetupParams(BaseModel):
    """ Construct the default model parameters"""

    # Grid discretization
    _sz = 20
    mesh_args: Dict[str, float] = {
        "mesh_size_frac": _sz,
        "mesh_size_min": 0.2 * _sz,
        "mesh_size_bound": 3 * _sz,
    }
    bounding_box: Dict[str, float] = {
        'xmin': -20,
        'xmax': 80,
        'ymin': 50,
        'ymax': 150,
        'zmin': -25,
        'zmax': 75,
    }

    # Injection location
    source_scalar_borehole_shearzone: Dict[str, str] = {
        "shearzone": "S1_2",
        "borehole": "INJ1",
    }

    # Stress tensor
    stress: np.ndarray = gts.stress_tensor()

    # Storage folder for grid files and visualization output
    folder_name: Path = prepare_directories(head="default/default_1")

    # shearzones
    shearzone_names: List[str] = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]

    # scaling coefficients
    length_scale: float = 0.05
    scalar_scale: float = 1e12

    # solver
    solver: str = "direct"

    # needed to allow numpy arrays
    class Config:
        arbitrary_types_allowed = True


def create_isc_domain(
        viz_folder_name: Union[str, Path],
        shearzone_names: List[str],
        bounding_box: dict,
        mesh_args: dict,
        n_refinements: int = 0
) -> List[pp.GridBucket]:
    """ Create a domain (.geo file) for the ISC test site.

    Parameters
    ----------
    viz_folder_name : str or pathlib.Path
        Absolute path to folder to store results in
    shearzone_names : List of str
        Names of shearzones to include
    bounding_box : dict
        Bounding box of domain ('xmin', 'xmax', etc.)
    mesh_args : dict
        Arguments for meshing (of coarsest grid)
    n_refinements : int, Default = 0
        Number of refined grids to produce.
        The grid is refined by splitting.
    """

    # ----------------------------------------
    # --- CREATE FRACTURE NETWORK AND MESH ---
    # ----------------------------------------
    network = gts.fracture_network(
        shearzone_names=shearzone_names,
        export_vtk=True,
        domain=bounding_box,
        network_path=f"{viz_folder_name}/fracture_network.vtu"
    )

    gmsh_file_name = str(viz_folder_name / "gmsh_frac_file")
    gb = network.mesh(mesh_args=mesh_args, file_name=gmsh_file_name)

    gb_list = refine_mesh(
        in_file=f'{gmsh_file_name}.geo',
        out_file=f"{gmsh_file_name}.msh",
        dim=3,
        network=network,
        num_refinements=n_refinements,
    )

    # TODO: Make this procedure "safe".
    #   E.g. assign names by comparing normal vector and centroid.
    #   Currently, we assume that fracture order is preserved in creation process.
    #   This may be untrue if fractures are (completely) split in the process.
    # Assign node prop 'name' to each grid in the grid bucket.
    for _gb in gb_list:
        pp.contact_conditions.set_projections(_gb)
        _gb.add_node_props(keys="name")
        fracture_grids = _gb.get_grids(lambda g: g.dim == _gb.dim_max() - 1)
        if shearzone_names is not None:
            for i, sz_name in enumerate(shearzone_names):
                _gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)
                # Note: Use self.gb.node_props(g, 'name') to get value.

    return gb_list


@timer(logger)
def run_models_for_convergence_study(
        model: Type[ContactMechanics],
        run_model_method: Callable,
        params: dict,
        n_refinements: int = 1,
        newton_params: dict = None,
        variable: List[str] = None,  # This is really required for the moment
        variable_dof: List[int] = None,
) -> Tuple[List[pp.GridBucket], List[dict]]:
    """ Run a model on a grid, refined n times.

    For a given model and method to run the model,
    and a set of parameters, n refinements of the
    initially generated grid will be performed.


    Parameters
    ----------
    model : Type[ContactMechanics]
        Which model to run
        Only tested for subclasses of ContactMechanicsISC
    run_model_method : Callable
        Which method to run model with
        Typically pp.run_stationary_model or pp.run_time_dependent_model
    params : dict (Default: None)
        Custom parameters to pass to model
    n_refinements : int (Default: 1)
        Number of grid refinements
    newton_params : dict (Default: None)
        Any non-default newton solver parameters to use
    variable : List[str]
        List of variables to consider for convergence study.
        If 'None', available variables on each sub-domain will be used
    variable_dof : List[str]
        Degrees of freedom for each variable
    setup_loggers : bool (Default: True)
        whether to set up loggers backend

    Returns
    -------
    gb_list : List[pp.GridBucket]
        list of (n+1) grid buckets in increasing order
        of refinement (reference grid last)
    errors : List[dict]
        List of (n) dictionaries, each containing
        the error for each variable on each grid.
        Each list entry corresponds index-wise to an
        entry in gb_list.
    """

    # 1. Step: Create n grids by uniform refinement.
    # 2. Step: for grid i in list of n grids:
    # 2. a. Step: Set up the mechanics model.
    # 2. b. Step: Solve the mechanics problem.
    # 2. c. Step: Keep the grid (with solution data)
    # 3. Step: Let the finest grid be the reference solution.
    # 4. Step: For every other grid:
    # 4. a. Step: Map the solution to the fine grid, and compute error.
    # 5. Step: Compute order of convergence, etc.

    params = SetupParams(
        **params,
    ).dict()
    logger.info(f"Preparing setup for convergence study on {pendulum.now().to_atom_string()}")

    # 1. Step: Create n grids by uniform refinement.
    gb_list = create_isc_domain(
        viz_folder_name=params['folder_name'],
        shearzone_names=params['shearzone_names'],
        bounding_box=params['bounding_box'],
        mesh_args=params['mesh_args'],
        n_refinements=n_refinements,
    )

    # -----------------------
    # --- SETUP AND SOLVE ---
    # -----------------------

    newton_options = {  # Parameters for Newton solver.
        "max_iterations": 10,
        "convergence_tol": 1e-10,
        "divergence_tol": 1e5,
    }
    if not newton_params:
        newton_params = {}
    newton_options.update(newton_params)
    logger.info(f"Options for Newton solver: \n {newton_options}")

    for gb in gb_list:
        setup = model(params=params)
        setup.set_grid(gb)

        logger.info("Setup complete. Starting simulation")
        run_model_method(setup, params=newton_options)
        logger.info("Simulation complete. Exporting solution.")

    # ----------------------
    # --- COMPUTE ERRORS ---
    # ----------------------

    gb_ref = gb_list[-1]

    errors = []
    for i in range(0, n_refinements):
        gb_i = gb_list[i]
        gb_coarse_fine_cell_mapping(gb=gb_i, gb_ref=gb_ref)

        _error = grid_error(
            gb=gb_i,
            gb_ref=gb_ref,
            variable=variable,
            variable_dof=variable_dof,
        )
        errors.append(_error)

    return gb_list, errors


