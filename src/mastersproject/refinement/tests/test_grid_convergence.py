import logging
from pathlib import Path
from typing import Tuple

import numpy as np

import porepy as pp
from GTS.isc_modelling.flow import FlowISC
from GTS.isc_modelling.parameter import FlowParameters, nd_injection_cell_center
from GTS.isc_modelling.setup import SetupParams
from refinement import gb_coarse_fine_cell_mapping
from refinement.grid_convergence import grid_error

logger = logging.getLogger(__name__)


class TestGridError:
    # IMPORTANT: NOTE:
    #   THIS DOESN'T WORK BECAUSE COARSE_FINE MAPPINGS
    #   AREN'T IMPLEMENTED FOR STRUCTURED GRIDS YET.
    def test_grid_error_structured_grid(self):
        """ Set up and solve a simple 3d-problem with 2 fractures.

        Verify decreasing error.

        # TODO: Modify FlowISC so that it modifies permeability on setup.

        """
        time_step = pp.HOUR
        params = FlowParameters(
            head="TestGridError/test_grid_error_structured_grid",
            time_step=time_step,
            end_time=time_step * 4,
            fluid_type=pp.UnitFluid,
            shearzone_names=["f1"],
            mesh_args={},
            source_scalar_borehole_shearzone=None,
            well_cells=nd_injection_cell_center,
            injection_rate=1,
            frac_permeability=1,
            intact_permeability=1,
        )
        n_ref = 2
        gb_list = [
            structured_grid_1_frac_horizontal_with_refinements(ref)
            for ref in range(n_ref + 1)
        ]

        for gb in gb_list:
            setup = FlowISC(params)
            setup.gb = gb

            pp.run_time_dependent_model(setup, {})

        # --- Compute errors ---
        gb_ref = gb_list[-1]

        errors = []
        for i in range(0, n_ref):
            gb_i = gb_list[i]
            gb_coarse_fine_cell_mapping(gb=gb_i, gb_ref=gb_ref)

            _error = grid_error(
                gb=gb_i, gb_ref=gb_ref, variable=["p_exp"], variable_dof=[1],
            )
            errors.append(_error)

        logger.info(errors)


def create_grid_with_two_fractures(
    path_head: str,
) -> Tuple[pp.GridBucket, SetupParams, pp.FractureNetwork3d]:
    """ Setup method

    Set up a simplified isc domain with simplified parameters

    Parameters
    ----------
    path_head : str
        head of path (the part following default root)

    Returns
    -------
    gb : pp.GridBucket
    params : SetupParams
        model parameters
    frac_network : pp.FractureNetwork3d

    """
    # Create folders
    path = Path(__file__).resolve().parent / "results"
    root = path / path_head
    root.mkdir(parents=True, exist_ok=True)
    file_name = root / "gmsh_frac_file"

    # Define domain
    domain = {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 1, "ymax": 1, "zmax": 1}

    # Define fractures
    frac_pts1 = np.array([[0.15, 0.15, 0.8, 0.8], [0, 0.9, 0.9, 0], [0, 0, 0.9, 0.9]])
    frac1 = pp.Fracture(frac_pts1)
    frac_pts2 = np.array([[1, 1, 0.15, 0.15], [0, 1, 1, 0], [0.15, 0.15, 1, 1]])
    frac2 = pp.Fracture(frac_pts2)

    # Create fracture network and mesh it
    frac_network = pp.FractureNetwork3d([frac1, frac2], domain)
    c = 5
    mesh_args = {
        "mesh_size_frac": c * 0.1,
        "mesh_size_min": c * 0.1,
        "mesh_size_bound": c * 0.4,
    }
    gb = frac_network.mesh(mesh_args, file_name=str(file_name))

    # Set up parameters based on this mesh.
    params = SetupParams(
        length_scale=1,
        scalar_scale=1,
        mesh_args=mesh_args,
        bounding_box=domain,
        folder_name=file_name.parent,
        shearzone_names=["S3_1", "S3_2"],
    )

    return gb, params, frac_network


def structured_grid_1_frac_horizontal_with_refinements(n_ref: int):
    # Domain has physical dimension (size)^3
    size = 10
    physdims = np.ones(3, dtype=np.int) * size

    # nx is initially (n_ref=0) (5)^3 cells
    # Every iteration splits each cell in 4
    nx = (np.ones(3, dtype=np.int) * 5) * (2 ** n_ref)

    # fmt: off
    frac_pts = np.array(
        [[0.2, 0.8, 0.8, 0.2],
         [0.2, 0.2, 0.8, 0.8],
         [0.5, 0.5, 0.5, 0.5]]) * size
    # fmt: on
    gb = pp.meshing.cart_grid([frac_pts], nx=nx, physdims=physdims)
    return gb
