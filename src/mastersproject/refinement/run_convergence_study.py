import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, List, Tuple, Type, Union

import porepy as pp
from GTS.isc_modelling.flow import Flow
from GTS.isc_modelling.parameter import GeometryParameters
from porepy.models.contact_mechanics_model import ContactMechanics
from refinement.grid_convergence import grid_error
from refinement.grid_refinement import (
    gb_coarse_fine_cell_mapping,
    refine_mesh_by_splitting,
)

logger = logging.getLogger(__name__)


def gb_refinements(
    network: Union[pp.FractureNetwork3d, pp.FractureNetwork2d],
    gmsh_folder_path: Union[str, Path],
    mesh_args: dict,
) -> Generator[pp.GridBucket, None, None]:
    """ Create n refinements of a fracture network.

    This method is used if you have a pp.FractureNetwork,
    but not a .geo file describing the mesh.
    If you have the .geo file as well, use refine_mesh_by_splitting directly.

    Parameters
    ----------
    network : Union[pp.FractureNetwork3d, pp.FractureNetwork2d]
        Fracture network
    gmsh_folder_path : Union[str, Path]
        Absolute path to folder to store results in
    mesh_args : dict
        Arguments for meshing (of coarsest grid)

    Returns
    -------
    gb : Generator[pp.GridBucket, None, None]
        A generator for grid buckets
    """

    # ----------------------------------------
    # --- CREATE FRACTURE NETWORK AND MESH ---
    # ----------------------------------------
    if isinstance(network, pp.FractureNetwork2d):
        dim = 2
    elif isinstance(network, pp.FractureNetwork3d):
        dim = 3
    else:
        # This might be too strict: consider passing dim as parameter instead.
        raise ValueError("Unknown input network")

    gmsh_file_name = str(gmsh_folder_path / "gmsh_frac_file")
    in_file = f"{gmsh_file_name}.geo"
    out_file = f"{gmsh_file_name}.msh"

    # Impose various parameters on the network.
    # See method definition for details / issues.
    network = _impose_network_parameters(network, mesh_args)

    # Dump the network description to gmsh .geo format
    network.to_gmsh(in_file, in_3d=True)

    yield from refine_mesh_by_splitting(
        in_file=in_file, out_file=out_file, dim=dim,
    )


def run_model_for_convergence_study(
    model: Union[Type[Flow], Type[ContactMechanics]],
    run_model_method: Callable,
    network: Union[pp.FractureNetwork3d, pp.FractureNetwork2d],
    params: GeometryParameters,
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
    run_model_method : Callable
        Which method to run model with
        Typically pp.run_stationary_model or pp.run_time_dependent_model
    network : Union[pp.FractureNetwork3d, pp.FractureNetwork2d]
        Fracture network
    params : GeometryParameters
        Parameters to pass to the model
    n_refinements : int (Default: 1)
        Number of grid refinements
    newton_params : dict (Default: None)
        Any non-default newton solver parameters to use
    variable : List[str]
        List of variables to consider for convergence study.
        If 'None', available variables on each sub-domain will be used
    variable_dof : List[str]
        Degrees of freedom for each variable

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

    logger.info(
        f"Preparing setup for convergence study " f"on {datetime.now().isoformat()}"
    )

    # 1. Step: Create n grids by uniform refinement.
    gb_generator = gb_refinements(
        network=network,
        gmsh_folder_path=params.folder_name,
        mesh_args=params.mesh_args,
    )
    gb_list = [next(gb_generator) for _ in range(0, n_refinements + 1)]
    gb_generator.close()

    # -----------------------
    # --- SETUP AND SOLVE ---
    # -----------------------
    newton_params = newton_params if newton_params else {}

    for gb in gb_list:
        setup = model(params=params)
        # Critical to this step is that setup.prepare_simulation()
        # (specifically setup.create_grid())
        # doesn't overwrite that we manually set a grid bucket to the model.
        setup.gb = gb
        pp.contact_conditions.set_projections(setup.gb)

        run_model_method(setup, params=newton_params)

    # ----------------------
    # --- COMPUTE ERRORS ---
    # ----------------------

    gb_ref = gb_list[-1]

    errors = []
    for i in range(0, n_refinements):
        gb_i = gb_list[i]
        gb_coarse_fine_cell_mapping(gb=gb_i, gb_ref=gb_ref)

        _error = grid_error(
            gb=gb_i, gb_ref=gb_ref, variable=variable, variable_dof=variable_dof,
        )
        errors.append(_error)

    return gb_list, errors


def _impose_network_parameters(network: pp.FractureNetwork3d, mesh_args: dict):
    """ TODO: Consider separating the start FractureNetwork3d.mesh,
            to avoid us copying its contents here
    We need to generate a .geo file for refine_mesh_by_splitting.
    refine_mesh_by_splitting will mesh the coarsest grid for us
    (in addition to all refinements),
    so we don't need to call network.mesh(...). Instead, we emulate
    its behaviour except for meshing itself.
    -- Start of FractureNetwork3d.mesh(...) below:
    """
    if not network.bounding_box_imposed:
        network.impose_external_boundary(network.domain)

    # Find intersections between fractures
    if not network.has_checked_intersections:
        network.find_intersections()
    else:
        logger.info("Use existing intersections")

    if "mesh_size_frac" not in mesh_args.keys():
        raise ValueError("Meshing algorithm needs argument mesh_size_frac")
    if "mesh_size_min" not in mesh_args.keys():
        raise ValueError("Meshing algorithm needs argument mesh_size_min")

    mesh_size_frac = mesh_args.get("mesh_size_frac", None)
    mesh_size_min = mesh_args.get("mesh_size_min", None)
    mesh_size_bound = mesh_args.get("mesh_size_bound", None)
    network._insert_auxiliary_points(mesh_size_frac, mesh_size_min, mesh_size_bound)

    # Process intersections to get a description of the
    # geometry in non-intersecting lines and polygons
    network.split_intersections()

    return network
