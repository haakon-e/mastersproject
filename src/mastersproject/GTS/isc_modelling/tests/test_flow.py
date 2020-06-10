from typing import Optional, List
import logging

import porepy as pp
import numpy as np
from GTS import FlowISC
from GTS.isc_modelling.ISCGrid import create_grid, optimize_mesh
from GTS.isc_modelling.parameter import FlowParameters, nd_injection_cell_center, shearzone_injection_cell

logger = logging.getLogger(__name__)


class TestFlow:

    def test_1_frac_unit_domain(self):
        """ Test that we a basic 1-fracture setup runs as
        expected with easy parameters.

        Goal: No negative pressure cells
        """
        _run_flow_models_helper(
            sz=0.1,
            incompressible=False,
            head="test_1_frac_unit_domain",
            shearzone_names=["f1"],
            time_step=pp.HOUR,
        )

    def test_2_fracs_unit_domain(self):
        """ Test that we a basic 2-fracture setup runs as
        expected with easy parameters.

        Goal: No negative pressure cells
        """
        _run_flow_models_helper(
            sz=0.1,
            incompressible=False,
            head="test_2_fracs_unit_domain",
            shearzone_names=["f1", "f2"],
            time_step=pp.HOUR,
        )

    def test_2_fracs_unit_domain_incompressible_flow(self):
        """ Test that we a basic 2-fracture setup runs as
        expected with easy parameters and incompressible flow.

        Goal: No negative pressure cells
        """
        # here; _sz=0.3 doesn't work. (=660 3d cells)
        #       _sz=0.2 works (=1262 3d cells)
        _run_flow_models_helper(
            sz=0.3,
            incompressible=True,
            head="test_2_fracs_unit_domain_incompressible_flow",
            shearzone_names=["f1", "f2"],
        )

    def test_1_frac_unit_domain_incompressible_flow(self):
        """ Test that we a basic 1-fracture setup runs as
        expected with easy parameters and incompressible flow.

        Goal: No negative pressure cells
        """
        # here; _sz=0.4 doesn't work. (=~200 3d cells)
        #       _sz=0.3 works (=344 3d cells)
        _run_flow_models_helper(
            sz=0.3,
            incompressible=True,
            head="test_1_frac_unit_domain_incompressible_flow",
            shearzone_names=["f1"],
        )

    def test_0_frac_unit_domain_incompressible_flow(self):
        """ Test that we a basic 0-fracture setup runs as
        expected with easy parameters and incompressible flow.

        Goal: No negative pressure cells
        """
        # here; _sz=0.6 works (=24 3d cells)
        # It seems like 24 3d cells is close to minimum reachable.

        _run_flow_models_helper(
            sz=0.6,
            incompressible=True,
            head="test_0_frac_unit_domain_incompressible_flow",
            shearzone_names=None,
        )


def _run_flow_models_helper(
        sz: float,
        incompressible: bool,
        head: str,
        shearzone_names: Optional[List[str]],
        time_step: float = None,
) -> None:
    """ Helper method for the test_flow setups

    sz is related to mesh_args
    incompressible turns on or off compressibility (and time stepping)
        - If compressible, you should also set time_step
    head is the path head. Usually, method name
    shearzone_names is a list of names given to fractures
    n_frac is the number of fractures to be constructed
    """

    fluid = pp.UnitFluid(11)
    if incompressible:
        # Consider an incompressible problem
        fluid.COMPRESSIBILITY = 0
        time_step = 1
        end_time = 1
    else:
        time_step = time_step
        end_time = time_step * 4

    params = FlowParameters(
        head=f"TestFlow/{head}",
        fluid=fluid,
        time_step=time_step,
        end_time=end_time,
        shearzone_names=shearzone_names,
        mesh_args={
            "mesh_size_frac": sz,
            "mesh_size_min": sz,
            "mesh_size_bound": sz * 4,
        },
        source_scalar_borehole_shearzone=None,
        well_cells=nd_injection_cell_center,
        injection_rate=1,
        frac_permeability=1,
        intact_permeability=1,
        bounding_box={
            'xmin': 0, 'ymin': 0, 'zmin': 0,
            'xmax': 1, 'ymax': 1, 'zmax': 1
        },
    )
    setup = FlowISC(params)
    network = network_n_fractures(params.n_frac)
    gb = network.mesh(
        mesh_args=params.mesh_args,
        file_name=str(params.folder_name / "gmsh_frac_file"),
    )
    setup.gb = gb
    pp.run_time_dependent_model(setup, {})

    assert setup.neg_ind.size == 0


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


class TestFlowISC:

    def test_low_k_1_frac(self):
        """ Run FlowISC on a mesh with 1 fracture"""
        _sz = 4  # _sz = 2
        intact_k = 1e-13
        frac_k = 1e-9

        time_step = pp.MINUTE
        params = FlowParameters(
            head=f"TestFlowISC/delete-me",  #test_low_k_1_frac/sz_{_sz}/kf_{intact_k}_ki_{frac_k}/1min",
            time_step=time_step,
            end_time=time_step * 4,
            shearzone_names=["S1_2"],
            bounding_box=None,
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": _sz,#0.2 * _sz,
                "mesh_size_bound": _sz,#3 * _sz,
            },
            well_cells=shearzone_injection_cell,
            injection_rate=1 / 6,
            frac_permeability=frac_k,
            intact_permeability=intact_k,
        )
        setup = FlowISC(params)
        _gb, network = create_grid(
            **params.dict(
                include={
                    "mesh_args",
                    "length_scale",
                    "bounding_box",
                    "shearzone_names",
                    "folder_name",
                }
            )
        )
        pp.run_time_dependent_model(setup, {})

        assert setup.neg_ind.size == 0

    def test_low_k_optimized_mesh_1_frac(self):
        """ Run FlowISC on optimized mesh with 1 fracture"""
        _sz = 4#2
        intact_k = 1e-20
        frac_k = 1e-16

        time_step = pp.MINUTE
        params = FlowParameters(
            head=f"TestFlowISC/test_low_k_optimized_mesh_1_frac/sz_{_sz}/kf_{intact_k}_ki_{frac_k}/1min",
            time_step=time_step,
            end_time=time_step * 4,
            shearzone_names=["S1_2"],
            bounding_box=None,
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
            well_cells=shearzone_injection_cell,
            injection_rate=1 / 6,
            frac_permeability=frac_k,
            intact_permeability=intact_k,
        )
        _helper_run_flowisc_optimized_grid(params)

    def test_low_k_optimized_mesh_5_frac(self):
        """ Run FlowISC on optimized mesh with 5 fractures"""
        _sz = 20  # 10
        intact_k = 1e-20
        frac_k = 1e-16

        time_step = pp.MINUTE
        params = FlowParameters(
            head=f"TestFlowISC/test_low_k_optimized_mesh_5_frac/sz_{_sz}/kf_{intact_k}_ki_{frac_k}/1min",
            time_step=time_step,
            end_time=time_step * 4,
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
            well_cells=shearzone_injection_cell,
            injection_rate=1 / 6,
            frac_permeability=frac_k,
            intact_permeability=intact_k,
        )
        _helper_run_flowisc_optimized_grid(params)

    def test_low_k_optimized_mesh(self):
        """ Run FlowISC on optimized meshes"""
        _sz = 10
        intact_k = 1e-16

        time_step = pp.MINUTE * 40
        params = FlowParameters(
            head=f"TestFlowISC/test_low_k_optimized_mesh/sz_{_sz}/k_{intact_k}/40min",
            time_step=time_step,
            end_time=time_step * 40,
            shearzone_names=None,
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
            source_scalar_borehole_shearzone=None,
            well_cells=nd_injection_cell_center,
            injection_rate=1 / 6,
            frac_permeability=0,
            intact_permeability=intact_k,
        )
        _helper_run_flowisc_optimized_grid(params)


def _helper_run_flowisc_optimized_grid(params: FlowParameters, optimize_method="Netgen"):
    """ Run FlowISC on optimized meshes"""

    _gb, network = create_grid(
        **params.dict(
            include={
                "mesh_args",
                "length_scale",
                "bounding_box",
                "shearzone_names",
                "folder_name",
            }
        )
    )

    logger.info(f"gb cells non-optimized: {_gb.num_cells()}")

    # Optimize the mesh
    in_file = params.folder_name / "gmsh_frac_file.geo"
    out_file = params.folder_name / "gmsh_frac_file-optimized.msh"
    optimize_mesh(
        in_file=in_file, out_file=out_file, method=optimize_method, force=True, dim_tags=[3]
    )
    gb: pp.GridBucket = pp.fracture_importer.dfm_from_gmsh(str(out_file), dim=3)
    logger.info(f"gb cells optimized: {gb.num_cells()}")
    assert _gb.num_cells() < gb.num_cells()

    # Run FlowISC
    setup = FlowISC(params)
    setup.gb = gb
    pp.run_time_dependent_model(setup, {})

    assert setup.neg_ind.size == 0
