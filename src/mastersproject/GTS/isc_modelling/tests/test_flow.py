
import porepy as pp
import numpy as np
from GTS import FlowISC
from GTS.isc_modelling.parameter import FlowParameters, nd_injection_cell_center


class TestFlow:

    def test_1_frac_unit_domain(self):
        """ Test that we a basic 1-fracture setup runs as
        expected with easy parameters.

        Goal: No negative pressure cells
        """

        _sz = 0.1
        time_step = pp.HOUR
        params = FlowParameters(
            head="TestRunModelForConvergenceStudy/test_2_fracs_unit_domain",
            time_step=time_step,
            end_time=time_step * 4,
            fluid=pp.UnitFluid(),
            shearzone_names=["f1"],
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": _sz,
                "mesh_size_bound": _sz * 4,
            },
            source_scalar_borehole_shearzone=None,
            well_cells=nd_injection_cell_center,
            injection_rate=1,
            frac_permeability=1,
            intact_permeability=1,
        )
        setup = FlowISC(params)
        network = fracture_network_1_frac_unit_domain()
        gb = network.mesh(
            mesh_args=params.mesh_args,
            file_name=str(params.folder_name / "gmsh_frac_file"),
        )
        setup.gb = gb
        pp.run_time_dependent_model(setup, {})

        assert setup.neg_ind.size == 0

    def test_2_fracs_unit_domain(self):
        """ Test that we a basic 2-fracture setup runs as
        expected with easy parameters. Goal: No negative pressure cells
        """

        _sz = 0.1
        time_step = pp.HOUR
        params = FlowParameters(
            head="TestRunModelForConvergenceStudy/test_2_fracs_unit_domain",
            time_step=time_step,
            end_time=time_step * 4,
            fluid=pp.UnitFluid(),
            shearzone_names=["f1", "f2"],
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": _sz,
                "mesh_size_bound": _sz * 4,
            },
            source_scalar_borehole_shearzone=None,
            well_cells=nd_injection_cell_center,
            injection_rate=1,
            frac_permeability=1,
            intact_permeability=1,
        )
        setup = FlowISC(params)
        network = fracture_network_2_fracs_unit_domain()
        gb = network.mesh(
            mesh_args=params.mesh_args,
            file_name=str(params.folder_name / "gmsh_frac_file"),
        )
        setup.gb = gb
        pp.run_time_dependent_model(setup, {})

        assert setup.neg_ind.size == 0

def fracture_network_1_frac_unit_domain():
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

    # Create fracture network and mesh it
    frac_network = pp.FractureNetwork3d([frac1], domain)
    return frac_network

def fracture_network_2_fracs_unit_domain():
    """ Add another fracture to the previous fracture network"""
    frac_pts2 = np.array(
        [[1, 1, 0.15, 0.15],
         [0, 1, 1, 0],
         [0.15, 0.15, 1, 1]])
    frac2 = pp.Fracture(frac_pts2)

    frac_network = fracture_network_1_frac_unit_domain()
    frac_network.add(frac2)

    return frac_network
