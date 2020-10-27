import logging

import porepy as pp
from GTS import FlowISC
from GTS.isc_modelling.ISCGrid import create_grid, optimize_mesh
from GTS.isc_modelling.parameter import FlowParameters, nd_injection_cell_center

logger = logging.getLogger(__name__)


class TestOptimizeMesh:
    def test_flow_model_regular_vs_optimized_mesh(self):
        """ Investigate if optimizing the mesh can give different
        results on a problem with known issues in the solution

        We expect regular mesh generation to produce a mesh of
        poor enough quality for the solution to be unphysical
        (here: we get 6 cells with negative pressure values).
        However, if we use the Netgen mesh optimizer, the find
        that the negative cells are eradicated from the solution.
        """

        # Create a regular, non-optimized mesh
        _sz = 10
        time_step = pp.MINUTE
        params = FlowParameters(
            head="TestOptimizeMesh/test_optimize_mesh",
            time_step=time_step,
            end_time=time_step * 4,
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
            intact_permeability=1e-13,
        )
        setup = FlowISC(params)

        pp.run_time_dependent_model(setup, {})

        assert setup.neg_ind.size == 6

        # --- Re-run test with optimized mesh ---
        # Optimize the mesh
        in_file = params.folder_name / "gmsh_frac_file.geo"
        out_file = params.folder_name / "optimized/gmsh_frac_file.msh"
        optimize_mesh(
            in_file=in_file, out_file=out_file, method="Netgen",
        )
        gb: pp.GridBucket = pp.fracture_importer.dfm_from_gmsh(str(out_file), dim=3)

        # Check that the grid was indeed optimized
        assert setup.gb.num_cells() < gb.num_cells()

        # Set up a new model
        params.folder_name = params.folder_name / "optimized"
        setup_opt = FlowISC(params)

        # Set the optimized grid bucket to the flow model
        setup_opt.gb = gb

        # Run the model
        pp.run_time_dependent_model(setup_opt, {})
        assert setup_opt.neg_ind.size == 0

    def test_optimize_mesh(self):
        # Create a regular, non-optimized mesh
        _sz = 10
        time_step = pp.MINUTE
        params = FlowParameters(
            head="TestOptimizeMesh/test_optimize_mesh_only",
            time_step=time_step,
            end_time=time_step * 4,
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
            intact_permeability=1e-13,
        )

        gb, network = create_grid(
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

        logger.info(f"gb cells: {gb.num_cells()}")

        # Optimize the mesh
        in_file = params.folder_name / "gmsh_frac_file.geo"
        out_file = params.folder_name / "gmsh_frac_file-optimized.msh"
        optimize_mesh(
            in_file=in_file, out_file=out_file, method="Netgen",
        )
        gb2: pp.GridBucket = pp.fracture_importer.dfm_from_gmsh(str(out_file), dim=3)

        logger.info(f"gb cells: {gb2.num_cells()}")

        assert gb.num_cells() < gb2.num_cells()
