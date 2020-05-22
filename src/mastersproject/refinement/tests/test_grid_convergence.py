from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np

import GTS as gts
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.setup import SetupParams
from GTS.isc_modelling.flow import FlowISC
from refinement.run_convergence_study import run_model_for_convergence_study


class TestGridError:
    def test_grid_error(self):
        """ Set up and solve a simple 3d-problem with 2 fractures.

        Verify decreasing error.

        # TODO: Modify FlowISC so that it modifies permeability on setup.

        """
        path_head = "TestGridError/test_grid_error"
        gb, params, network = create_grid_with_two_fractures(path_head)

        # Parameters for run_model_for_convergence_study
        model_setup = {
            "model": FlowISC,
            "run_model_method": pp.run_time_dependent_model,
            "network": network,
            "params": params.dict(),
            "n_refinements": 2,
            "variable": ['p_exp'],
            "variable_dof": [1],
        }
        gb_list, errors = run_model_for_convergence_study(**model_setup)


def create_grid_with_two_fractures(path_head: str) -> Tuple[pp.GridBucket, SetupParams, pp.FractureNetwork3d]:
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
    path = Path(__file__).resolve().parent / 'results'
    root = path / path_head
    root.mkdir(parents=True, exist_ok=True)
    file_name = root / 'gmsh_frac_file'

    # Define domain
    domain = {
        'xmin': 0, 'ymin': 0, 'zmin': 0,
        'xmax': 1, 'ymax': 1, 'zmax': 1
    }

    # Define fractures
    frac_pts1 = np.array(
        [[0.15, 0.15, 0.8, 0.8],
         [0, 0.9, 0.9, 0],
         [0, 0, 0.9, 0.9]])
    frac1 = pp.Fracture(frac_pts1)
    frac_pts2 = np.array(
        [[1, 1, 0.15, 0.15],
         [0, 1, 1, 0],
         [0.15, 0.15, 1, 1]])
    frac2 = pp.Fracture(frac_pts2)

    # Create fracture network and mesh it
    frac_network = pp.FractureNetwork3d([frac1, frac2], domain)
    c = 5
    mesh_args = {"mesh_size_frac": c * 0.1, "mesh_size_min": c * 0.1, "mesh_size_bound": c * 0.4}
    gb = frac_network.mesh(mesh_args, file_name=str(file_name))

    # Set up parameters based on this mesh.
    params = SetupParams(
        length_scale=1, scalar_scale=1, mesh_args=mesh_args, bounding_box=domain, folder_name=file_name.parent,
        shearzone_names=["S3_1", "S3_2"]
    )

    return gb, params, frac_network
