import logging

import numpy as np

import GTS as gts
import GTS.test.util as test_util
import pendulum
import porepy as pp
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotISC
from mastersproject.util.logging_util import trace

logger = logging.getLogger(__name__)


def test_compare_run_mech_and_run_mech_by_filter_term():
    """ This test runs pure mechanics by running the test
    'test_mechanics_class_methods.test_decomposition_of_stress()'
    for the hydrostatic case. Then, it runs the test
    'test_run_mechanics_term_by_filter()'.

    The goal is to compare the output of these two methods and ensure they
    are the same.
    """
    # 1. Prepare parameters
    stress = gts.stress_tensor()
    # We set up hydrostatic stress
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)
    no_shearzones = None
    gravity = False  # No gravity effects
    params = {
        "stress": stress,
        "shearzone_names": no_shearzones,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
    }

    # Storage folder
    this_method_name = test_compare_run_mech_and_run_mech_by_filter_term.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}"

    # 1. --- Setup ContactMechanicsISC ---
    setup_mech = test_util.prepare_setup(
        model=gts.ContactMechanicsISC,
        path_head=f"{_folder_root}/test_mech",
        params=params,
        prepare_simulation=False,
        setup_loggers=True,
    )
    setup_mech.create_grid()

    # 2. --- Setup BiotReduceToMechanics ---
    params2 = test_util.prepare_params(
        path_head=f"{_folder_root}/test_biot_reduce_to_mech",
        params=params,
        setup_loggers=False,
    )
    setup_biot = BiotReduceToMechanics(params=params2)

    # Recreate the same mesh as for the above setup
    path_to_gb_msh = f"{setup_mech.viz_folder_name}/gmsh_frac_file.msh"
    gb2 = pp.fracture_importer.dfm_from_gmsh(
        path_to_gb_msh, dim=3, network=setup_mech._network
    )
    setup_biot.set_grid(gb2)

    # 3. --- Run simulations ---
    nl_params = {}

    # Run ContactMechanicsISC
    pp.run_stationary_model(setup_mech, nl_params)

    # Run BiotReduceToMechanics
    pp.run_time_dependent_model(setup_biot, nl_params)

    # --- Compare results ---
    def get_u(_setup):
        gb = _setup.gb
        g = gb.grids_of_dimension(3)[0]
        d = gb.node_props(g)
        u = d["state"]["u"].reshape((3, -1), order="F")
        return u

    u_mech = get_u(setup_mech)
    u_biot = get_u(setup_biot)
    assert np.isclose(np.sum(np.abs(u_mech - u_biot)), 0.0), (
        "Running mechanics or biot (only discretize mechanics "
        "term should return same result."
    )
    return setup_mech, setup_biot


def test_run_mechanics_term_by_filter():
    """ This test intends to replicate the part of the
     results of 'test_decomposition_of_stress' by only
     discretizing the mechanics term.
    """

    # 1. Prepare parameters
    stress = gts.stress_tensor()
    # We set up hydrostatic stress
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    no_shearzones = None
    gravity = False  # No gravity effects

    params = {
        "stress": stress,
        "shearzone_names": no_shearzones,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
    }

    # Storage folder
    this_method_name = test_run_mechanics_term_by_filter.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/test_1"

    #
    # 2. Setup and run test
    params = test_util.prepare_params(
        path_head=_folder_root, params=params, setup_loggers=True
    )
    setup = BiotReduceToMechanics(params=params)

    nl_params = {}  # Default Newton Iteration parameters
    pp.run_time_dependent_model(setup, params=nl_params)

    return setup


def test_run_flow_term_by_filter():
    """ This test intends to test results of running only
    the flow terms of the biot equation (on all subdomains)

    -- Key features:
    * 2 intersecting fractures
    * Full stress tensor and mechanical gravity terms included
    * hydrostatic scalar BC's
    * Initialization phase AND Injection phase (shortened)
    """

    # 1. Prepare parameters
    stress = gts.stress_tensor()
    # # We set up hydrostatic stress
    # hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    # stress = np.diag(hydrostatic)

    shearzones = ["S1_2", "S3_1"]
    gravity = True  # False  # No gravity effects

    params = {
        "stress": stress,
        "shearzone_names": shearzones,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
    }

    # Storage folder
    this_method_name = test_run_flow_term_by_filter.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/test_1"

    #
    # 2. Setup and run test
    params = test_util.prepare_params(
        path_head=_folder_root, params=params, setup_loggers=True
    )
    setup = BiotReduceToFlow(params=params)

    nl_params = {}  # Default Newton Iteration parameters
    pp.run_time_dependent_model(setup, params=nl_params)

    # Stimulation phase
    logger.info(
        f"Starting stimulation phase at time: {pendulum.now().to_atom_string()}"
    )
    setup.prepare_main_run()
    logger.info("Setup complete. Starting time-dependent simulation")
    pp.run_time_dependent_model(setup=setup, params=params)

    return setup


def test_run_biot_term_by_term(test_name: str):
    # TODO: THIS METHOD IS NOT FINISHED SET UP (maybe remove it)
    """ This test intends to investigate various
    properties of the biot equation by discretizing
    only certain terms.

    Additional simplifications:
    * hydrostatic mechanical stress
    * no mechanical gravity term
    """

    # 1. Prepare parameters
    stress = gts.stress_tensor()
    # We set up hydrostatic stress
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    no_shearzones = None
    gravity = False  # No gravity effects

    params = {
        "stress": stress,
        "shearzone_names": no_shearzones,
        "_gravity_bc": gravity,
        "_gravity_src": gravity,
    }

    # Storage folder
    this_method_name = test_run_biot_term_by_term.__name__
    now_as_YYMMDD = pendulum.now().format("YYMMDD")
    _folder_root = f"{this_method_name}/{now_as_YYMMDD}/{test_name}"

    #
    # 2. Setup and run test
    params = test_util.prepare_params(
        path_head=_folder_root, params=params, setup_loggers=True
    )
    setup = BiotReduceToMechanics(params=params)

    nl_params = {}  # Default Newton Iteration parameters
    pp.run_time_dependent_model(setup, params=nl_params)

    return setup


# --- Overwrite discretize method in ContactMechanics ---
# We overwrite the discretize() method to discretize only
# the desired terms.
class BiotReduceToMechanics(ContactMechanicsBiotISC):
    @trace(logger)
    def discretize(self):
        """ Discretize the mechanics stress term
        """
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(
                self.gb, active_variables=[self.displacement_variable]
            )

        g_max = self.gb.grids_of_dimension(self.Nd)[0]
        self.assembler.discretize(
            grid=g_max, variable_filter=self.displacement_variable
        )


class BiotReduceToFlow(ContactMechanicsBiotISC):
    @trace(logger)
    def discretize(self):
        """ Discretize the flow terms only"""
        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(
                self.gb,
                active_variables=[self.scalar_variable, self.mortar_scalar_variable],
            )

        self.assembler.discretize()

        # self.assembler.discretize(
        #     variable_filter=[self.scalar_variable, self.mortar_scalar_variable]
        # )
        # for g, _ in self.gb:
        #     self.assembler.discretize(grid=g, variable_filter=[self.scalar_variable])

    def check_convergence(self, solution, prev_solution, init_solution, nl_params=None):
        g_max = self._nd_grid()

        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = np.any(np.isnan(solution))
            converged = not diverged
            error = np.nan if diverged else 0
            return error, converged, diverged

        scalar_dof = self.assembler.dof_ind(g_max, self.scalar_variable)

        # Pressure solution
        p_scalar_now = solution[scalar_dof] * self.scalar_scale
        p_scalar_prev = prev_solution[scalar_dof] * self.scalar_scale
        p_scalar_init = init_solution[scalar_dof] * self.scalar_scale

        # Calculate errors

        # Pressure scalar error
        # scalar_norm = np.sum(p_scalar_now ** 2)
        difference_in_iterates_scalar = np.sum((p_scalar_now - p_scalar_prev) ** 2)
        difference_from_init_scalar = np.sum((p_scalar_now - p_scalar_init) ** 2)
        logger.info(f"diff iter scalar = {difference_in_iterates_scalar:.6e}")
        logger.info(f"diff init scalar = {difference_from_init_scalar:.6e}")

        tol_convergence = nl_params["nl_convergence_tol"]
        # Not sure how to use the divergence criterion
        # tol_divergence = nl_params["nl_divergence_tol"]

        converged = False
        diverged = False

        # Converge in pressure on 3D grid
        converged_p = False

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        if difference_in_iterates_scalar < tol_convergence:
            converged_p = True
            error_scalar = difference_in_iterates_scalar
            logger.info("pressure converged absolutely")
        else:
            # Relative convergence criterion:
            if (
                difference_in_iterates_scalar
                < tol_convergence * difference_from_init_scalar
            ):
                # converged = True
                converged_p = True
                logger.info("pressure converged relatively")

            error_scalar = difference_in_iterates_scalar / difference_from_init_scalar

        logger.info(f"Error in pressure is {error_scalar:.6e}.")

        converged = converged_p

        return error_scalar, converged, diverged

    def assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        # from porepy.utils.derived_discretizations import (
        #     implicit_euler as IE_discretizations,
        # )

        # Shorthand
        key_s = self.scalar_parameter_key
        var_s = self.scalar_variable

        # Scalar discretizations (all dimensions)
        diff_disc_s = pp.Mpfa(key_s)
        # mass_disc_s = pp.MassMatrix(key_s)
        # diff_disc_s = IE_discretizations.ImplicitMpfa(key_s)
        # mass_disc_s = IE_discretizations.ImplicitMassMatrix(key_s, var_s)
        source_disc_s = pp.ScalarSource(key_s)

        # Assign node discretizations
        for g, d in self.gb:
            d[pp.DISCRETIZATION] = {
                var_s: {
                    "diffusion": diff_disc_s,
                    # "mass": mass_disc_s,
                    "source": source_disc_s,
                },
            }

        for e, d in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            d[pp.COUPLING_DISCRETIZATION] = {
                self.scalar_coupling_term: {
                    g_h: (var_s, "diffusion"),
                    g_l: (var_s, "diffusion"),
                    e: (
                        self.mortar_scalar_variable,
                        pp.RobinCoupling(key_s, diff_disc_s),
                    ),
                },
            }

    # -- Sanity checks --
    def bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        pass

    def bc_values_mechanics(self, g) -> np.array:
        pass

    def bc_values_scalar(self, g) -> np.array:
        # DIRICHLET
        all_bf, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)

        # TEMPORARY: Add a BC/source for testing
        if g.dim == 3:
            all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(
                g
            )
            # point = np.array(
            #     [
            #         [self.box["xmin"]],
            #         [(self.box["ymin"] + self.box["ymax"]) / 2],
            #         [(self.box["zmin"] + self.box["zmax"]) / 2],
            #     ]
            # )
            # distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
            # indexes = np.argpartition(distances, 1)[:1]
            value = 100 * pp.MEGA * (pp.PASCAL / self.scalar_scale)
            bc_values[west] = value  # BC Pressure
        return bc_values

    def bc_type_scalar(self, g) -> pp.BoundaryCondition:
        all_bf, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, ["dir"] * all_bf.size)

    def source_scalar(self, g: pp.Grid) -> np.array:
        return np.zeros(g.num_cells)

    def source_mechanics(self, g) -> np.array:
        pass

    def set_mechanics_parameters(self) -> None:
        pass

    def set_viz(self):
        super().set_viz()
        self.export_fields = [self.p_exp]

    def export_step(self):
        gb = self.gb
        Nd = self.Nd
        ss = self.scalar_scale
        for g, d in gb:
            if self.scalar_variable in d[pp.STATE]:
                d[pp.STATE][self.p_exp] = d[pp.STATE][self.scalar_variable].copy() * ss
            else:
                d[pp.STATE][self.p_exp] = np.zeros((Nd, g.num_cells))
        self.viz.write_vtk(
            data=self.export_fields, time_step=self.time
        )  # Write visualization
        self.export_times.append(self.time)

    def after_newton_convergence(self, solution, errors, iteration_counter):
        self.assembler.distribute_variable(solution)
        self.export_step()

    def initial_condition(self) -> None:
        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            initial_value = np.zeros(mg.num_cells)
            state = {self.mortar_scalar_variable: initial_value}
            pp.set_state(d, state)
            # d[pp.STATE][self.mortar_scalar_variable] = initial_value


class BiotReduceDiscretization(ContactMechanicsBiotISC):
    def __init__(self, params):
        super().__init__(params)

        #
        # Discretize only a subset of terms or variables
        self._term_filter = params.get("_term_filter", None)
        self._variable_filter = params.get("_variable_filter", None)

    @trace(logger)
    def discretize(self):
        """ Discretize only a select term"""

        super().discretize()
