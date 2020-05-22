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

# # Refinement
# from refinement import gb_coarse_fine_cell_mapping
# from refinement.grid_convergence import grid_error
# from refinement import refine_mesh_by_splitting

# --- LOGGING UTIL ---
try:
    from src.mastersproject.util.logging_util import timer, trace, __setup_logging
except ImportError:
    from util.logging_util import timer, trace, __setup_logging

logger = logging.getLogger(__name__)

# --- RUN MODEL METHODS ---


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
    logger.info(
        f"Starting stimulation phase at time: {pendulum.now().to_atom_string()}"
    )
    setup.prepare_main_run()
    logger.info("Setup complete. Starting time-dependent simulation")
    pp.run_time_dependent_model(setup=setup, params=params)


# * RUN ABSTRACT METHOD *


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
    params = SetupParams(**params,).dict()

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

    logger.info(
        f"Simulation complete. Exporting solution. Time: {pendulum.now().to_atom_string()}"
    )

    return setup


# --- PREPARE SIMULATIONS: DIRECTORIES AND PARAMETERS ---


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
    # bounding_box: Dict[str, float] = {
    #     'xmin': -20,
    #     'xmax': 80,
    #     'ymin': 50,
    #     'ymax': 150,
    #     'zmin': -25,
    #     'zmax': 75,
    # }
    bounding_box: Dict[str, float] = {
        "xmin": -100,
        "xmax": 200,
        "ymin": 0,
        "ymax": 300,
        "zmin": -100,
        "zmax": 200,
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
    shearzone_names: Union[List[str], None] = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]

    # scaling coefficients
    length_scale: float = 0.05
    scalar_scale: float = 1e10

    # solver
    solver: str = "direct"

    # needed to allow numpy arrays
    class Config:
        arbitrary_types_allowed = True
