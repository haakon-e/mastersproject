from pathlib import Path
from typing import List, Optional

import numpy as np
import porepy as pp
from GTS import FlowISC
from GTS.isc_modelling.parameter import FlowParameters, nd_injection_cell_center
from refinement.run_convergence_study import run_model_for_convergence_study


class TestRunModelForConvergenceStudy:
    """ Test run_model_for_convergence_study on various domains and parameters
    """

    def test_0_frac_unit_domain(self):
        """ No fractures, incompressible
        """
        _run_convergence_study_helper(
            head="test_0_frac_unit_domain",
            shearzone_names=None,
            sz=0.3,
            n_refinements=3,
            intact_permeability=0.1,
        )

    # IMPORTANT: We are not able to run proper tests on fractured grids as we cannot
    #           run more than 1 refinement due to constraints in number of cells.

    def test_2_fracs_unit_domain(self):
        """ 2 fractures, incompressible

        # Note: The current mesh_args gives no negative cells, but we are not able
        # to solve a second refinement. So this test cannot be run to its full intent.
        # See test_grid_convergence.py instead.
        """
        _run_convergence_study_helper(
            head="test_2_fracs_unit_domain",
            shearzone_names=["f1", "f2"],
            sz=0.3,
            n_refinements=2,
            intact_permeability=0.1,
        )

    def test_1_frac_unit_domain(self):
        """ 1 fracture, incompressible

        # Note: The current mesh_args gives no negative cells, but we are not able
        # to solve a second refinement. So this test cannot be run to its full intent.
        # See test_grid_convergence.py instead.
        """
        _run_convergence_study_helper(
            head="test_1_frac_unit_domain",
            shearzone_names=["f1"],
            sz=0.3,
            n_refinements=2
        )


def _run_convergence_study_helper(
        head: str,
        shearzone_names: Optional[List[str]],
        sz: float,
        n_refinements: int,
        frac_permeability: float = 1,
        intact_permeability: float = 1,
) -> None:
    """ Helper method for run_convergence_study tests

    head is the path head. Usually, method name
    shearzone_names is a list of names given to fractures
    sz is related to mesh_args
    n_refinements is the number of refinements to run
    frac_permeability, intact_permeability: Optionally set custom permeability
    """

    fluid = pp.UnitFluid(11)
    fluid.COMPRESSIBILITY = 0  # Consider an incompressible problem

    params = FlowParameters(
        head=f"TestRunModelForConvergenceStudy/{head}",
        fluid=fluid,
        shearzone_names=shearzone_names,
        mesh_args={
            "mesh_size_frac": sz,
            "mesh_size_min": sz,
            "mesh_size_bound": sz * 4,
        },
        source_scalar_borehole_shearzone=None,
        well_cells=nd_injection_cell_center,
        injection_rate=1,
        frac_permeability=frac_permeability,
        intact_permeability=intact_permeability,
    )

    network = network_n_fractures(params.n_frac)

    gb_list, errors = run_model_for_convergence_study(
        model=FlowISC,
        run_model_method=pp.run_time_dependent_model,
        network=network,
        params=params,
        n_refinements=n_refinements,
        newton_params=None,
        variable=['p_exp'],
        variable_dof=[1],
    )

    print(errors)


def network_n_fractures(n_frac: int) -> pp.FractureNetwork3d:
    """ Create a unit domain in 3d with n (pre-defined) fractures

    """
    assert 0 <= n_frac <= 2, "Only implemented between 0 and 2 fractures"

    bounding_box = {
        'xmin': 0, 'ymin': 0, 'zmin': 0,
        'xmax': 1, 'ymax': 1, 'zmax': 1
    }
    network = pp.FractureNetwork3d(None, bounding_box)

    if n_frac >= 1:
        frac_pts1 = np.array(
            [[0.15, 0.15, 0.8, 0.8],
             [0, 0.9, 0.9, 0],
             [0, 0, 0.9, 0.9]])
        network.add(pp.Fracture(frac_pts1))

    if n_frac >= 2:
        frac_pts2 = np.array(
            [[1, 1, 0.15, 0.15],
             [0, 1, 1, 0],
             [0.15, 0.15, 1, 1]])
        network.add(pp.Fracture(frac_pts2))

    return network
