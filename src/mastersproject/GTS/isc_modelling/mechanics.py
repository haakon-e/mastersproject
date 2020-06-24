import logging
from typing import Dict, Tuple

import numpy as np
import pendulum

import GTS as gts
import porepy as pp
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.general_model import CommonAbstractModel
from GTS.isc_modelling.parameter import BaseParameters, GrimselGranodiorite
from mastersproject.util.logging_util import timer, trace
from porepy.models.contact_mechanics_model import ContactMechanics
from porepy.params.data import add_nonpresent_dictionary

logger = logging.getLogger(__name__)


class Mechanics(CommonAbstractModel):
    def __init__(self, params: BaseParameters):
        """ General mechanics model for static contact mechanics

        Parameters
        ----------
        params : BaseParameters
        """
        super().__init__(params)

        # Variables
        self.displacement_variable = "u"
        self.mortar_displacement_variable = "mortar_u"
        self.contact_traction_variable = "contact_traction"

        # Keyword
        self.mechanics_parameter_key = "mechanics"

        # Terms of the equations
        self.friction_coupling_term = "fracture_force_balance"

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            bounding_box (dict): The bounding box of the domain, defined
                through minimum and maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """
        pass

    # --- Boundary condition and source terms ---

    def bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:  # noqa
        # Define boundary regions
        all_bf = g.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(g, all_bf, "dir")  # noqa

        # Internal faces are Neumann by default. We change them to
        # Dirichlet for the contact problem. That is: The mortar
        # variable represents the displacement on the fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """ Set homogeneous conditions on all boundary faces.
        """
        # Values for all Nd components, facewise
        values = np.zeros((self.Nd, g.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def source_mechanics(self, g: pp.Grid) -> np.ndarray:
        """ Vectorial source term (Nd * g.num_cells)"""
        return np.zeros(self.Nd * g.num_cells)

    # --- Set parameters ---

    def rock_friction_coefficient(self, g: pp.Grid) -> np.ndarray:  # noqa
        """ The friction coefficient is uniform, and equal to 1.

        Assumes self.set_rock() is called
        """
        return np.ones(g.num_cells)

    def set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = (
                    self.params.rock.LAMBDA
                    * (pp.PASCAL / self.params.scalar_scale)
                    * np.ones(g.num_cells)
                )
                mu = (
                    self.params.rock.MU
                    * (pp.PASCAL / self.params.scalar_scale)
                    * np.ones(g.num_cells)
                )
                constit = pp.FourthOrderTensor(mu, lam)

                # BC and source values
                bc: pp.BoundaryConditionVectorial = self.bc_type_mechanics(g)
                bc_val: np.ndarray = self.bc_values_mechanics(g)
                source_val: np.ndarray = self.source_mechanics(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": constit,
                        # "max_memory": 7e7,
                        # "inverter": python,
                    },
                )

            elif g.dim == self.Nd - 1:
                friction = self.rock_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )
        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    # --- Primary variables and discretizations ---

    def assign_mechanics_variables(self) -> None:
        """ Assign variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        primary_vars = pp.PRIMARY_VARIABLES
        var_m = self.displacement_variable
        var_contact = self.contact_traction_variable
        var_mortar = self.mortar_displacement_variable

        for g, d in gb:
            add_nonpresent_dictionary(d, primary_vars)
            if g.dim == self.Nd:
                d[primary_vars].update({
                    var_m: {"cells": self.Nd},
                })

            elif g.dim == self.Nd - 1:
                d[primary_vars].update({
                    var_contact: {"cells": self.Nd},
                })

        for e, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            g_l, g_h = gb.nodes_of_edge(e)
            if g_h.dim == self.Nd:
                d[primary_vars].update({
                    var_mortar: {"cells": self.Nd},
                })

    def assign_mechanics_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        nd = self.Nd
        # Shorthand
        key_m, var_m = self.mechanics_parameter_key, self.displacement_variable
        var_contact = self.contact_traction_variable
        var_mortar = self.mortar_displacement_variable
        discr_key, coupling_discr_key = pp.DISCRETIZATION, pp.COUPLING_DISCRETIZATION

        # For the Nd domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(key_m)

        # We need a void discretization for the contact traction variable defined on
        # the fractures.
        empty_discr = pp.VoidDiscretization(key_m, ndof_cell=nd)

        # Assign node discretizations
        for g, d in gb:
            add_nonpresent_dictionary(d, discr_key)

            if g.dim == nd:
                d[discr_key].update({
                    var_m: {"mpsa": mpsa},
                })

            elif g.dim == nd - 1:
                d[discr_key].update({
                    var_contact: {"empty": empty_discr},
                })

        # Define the contact condition on the mortar grid
        coloumb = pp.ColoumbContact(key_m, nd, mpsa)
        contact = pp.PrimalContactCoupling(key_m, mpsa, coloumb)

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            add_nonpresent_dictionary(d, coupling_discr_key)
            if g_h.dim == nd:
                d[coupling_discr_key].update({
                    self.friction_coupling_term: {
                        g_h: (var_m, "mpsa"),
                        g_l: (var_contact, "empty"),
                        e: (var_mortar, contact),
                    },
                })

    @timer(logger, level="INFO")
    def discretize(self) -> None:
        """ Discretize all terms
        """
        if not self.assembler:
            self.assembler = pp.Assembler(self.gb)

        self.assembler.discretize()

    # --- Initial condition ---

    def initial_mechanics_condition(self) -> None:
        """ Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction,
        and -1 (that is, in contact) in the normal direction.

        We initialize "previous_iterate" for the contact traction and
        mortar displacement since they are to be updated every Newton
        iteration.

        """
        gb = self.gb
        var_m = self.displacement_variable
        var_contact = self.contact_traction_variable
        var_mortar = self.mortar_displacement_variable
        state = pp.STATE

        for g, d in gb:
            add_nonpresent_dictionary(d, state)
            if g.dim == self.Nd:
                # Initialize displacement variable
                initial_displacement_value = np.zeros(g.num_cells * self.Nd)
                d[state].update({
                    var_m: initial_displacement_value,
                })

            elif g.dim == self.Nd - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (np.zeros((g.dim, g.num_cells)), -1 * np.ones(g.num_cells))
                ).ravel(order="F")
                d[state].update({
                    "previous_iterate": {var_contact: traction},
                    var_contact: traction,
                })

        for e, d in self.gb.edges():
            add_nonpresent_dictionary(d, state)

            mg: pp.MortarGrid = d["mortar_grid"]
            if mg.dim == self.Nd - 1:
                size = mg.num_cells * self.Nd
                d[state].update({
                    var_mortar: np.zeros(size),
                    "previous_iterate": {
                        var_mortar: np.zeros(size)
                    },
                })

    # --- Simulation and solvers ---

    @timer(logger, level="INFO")
    def prepare_simulation(self) -> None:
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self._prepare_grid()

        self.set_mechanics_parameters()
        self.assign_mechanics_variables()
        self.assign_mechanics_discretizations()
        self.initial_mechanics_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

    def _prepare_grid(self) -> None:
        """ Wrapper to create grid"""
        self.create_grid()

    @timer(logger, level="INFO")
    def initialize_linear_solver(self) -> None:
        """ Initialize linear solver

        Currently, we only consider the direct solver.
        See also self.assemble_and_solve_linear_system()
        """

        # # Compute exact condition number:
        # A, _ = self.assembler.assemble_matrix_rhs()
        # cond = spla.norm(A, 1) * spla.norm(spla.inv(A), 1)
        # logger.info(f"Exact condition number: {cond:.2e}")

        if self.params.linear_solver == "direct":
            """ In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.

            """
            pass

        else:
            raise ValueError(f"Unknown linear solver {self.params.linear_solver}")

    def _check_convergence_mechanics(self, solution, prev_solution, init_solution, nl_params):
        """ Check convergence and compute error of matrix displacement variable"""
        var_m = self.displacement_variable
        g_max = self._nd_grid()
        # NOTE: In previous simulations, this was erronuously scalar scale.
        ls = self.params.length_scale

        # Get the solution from current and previous iterates,
        # as well as the initial guess.
        mech_dof = self.assembler.dof_ind(g_max, var_m)
        u_mech_now = solution[mech_dof] * ls
        u_mech_prev = prev_solution[mech_dof] * ls
        u_mech_init = init_solution[mech_dof] * ls

        # Calculate errors
        difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
        difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)

        tol_convergence = nl_params.get("nl_convergence_tol")

        converged = False
        diverged = False

        # Check absolute convergence criterion
        if difference_in_iterates_mech < tol_convergence:
            converged = True
            error_mech = difference_in_iterates_mech

        else:
            # Check relative convergence criterion
            if (
                    difference_in_iterates_mech
                    < tol_convergence * difference_from_init_mech
            ):
                converged = True
            error_mech = difference_in_iterates_mech / difference_from_init_mech

        logger.info(f"Error in matrix displacement is {error_mech:.6e}")
        logger.info(f"Matrix displacement {'converged' if converged else 'did not converge'}. ")

        return error_mech, converged, diverged

    def _check_convergence_contact(self, solution, prev_solution, init_solution, nl_params):
        """ Check convergence and compute error of contact traction variable"""

        contact_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)
            if g_h.dim == self.Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.assembler.dof_ind(e[1], self.contact_traction_variable),
                    )
                )

        # IS POREPY TRACTION WEIGHED ???
        ls = self.params.length_scale
        ss = self.params.scalar_scale
        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        contact_now = solution[contact_dof]
        contact_prev = prev_solution[contact_dof]
        contact_init = init_solution[contact_dof]

        # Calculate errors
        contact_norm = np.sum(contact_now ** 2) * ss * ls ** 2
        difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2) * ss * ls ** 2
        difference_from_init_contact = np.sum((contact_now - contact_init) ** 2) * ss * ls ** 2

        tol_convergence = nl_params["nl_convergence_tol"]

        converged = False
        diverged = False

        # The if is intended to avoid division through zero
        if (
                contact_norm < tol_convergence
                and difference_in_iterates_contact < tol_convergence
        ):
            converged = True
            error_contact = difference_in_iterates_contact
        else:
            error_contact = (
                    difference_in_iterates_contact / difference_from_init_contact
            )

        logger.info(f"Error in contact force is {error_contact:.6e}.\n"
                    f"Contact force {'converged' if converged else 'did not converge'}.")

        return error_contact, converged, diverged

    def check_convergence(
            self,
            solution: np.ndarray,
            prev_solution: np.ndarray,
            init_solution: np.ndarray,
            nl_params: Dict,
    ) -> Tuple[np.ndarray, bool, bool]:

        # Convergence check for linear problems
        if not self._is_nonlinear_problem():
            return super().check_convergence(
                solution, prev_solution, init_solution, nl_params
            )

        # -- Calculate mechanics error for non-linear simulations --

        error_mech, converged_mech, diverged_mech = self._check_convergence_mechanics(
            solution, prev_solution, init_solution, nl_params,
        )
        _, converged_contact, diverged_contact = self._check_convergence_contact(
            solution, prev_solution, init_solution, nl_params,
        )

        converged = converged_mech and converged_contact
        diverged = diverged_mech or diverged_contact

        # Only return matrix displacement error for now
        return error_mech, converged, diverged

    # --- Newton iterations ---

    def before_newton_loop(self) -> None:
        """ Will be run before entering a Newton loop.

        Discretize time-dependent quantities etc.
        """
        self.set_mechanics_parameters()

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        self.assembler.discretize(term_filter=[self.friction_coupling_term])

    def update_state(self, solution_vector: np.ndarray) -> None:
        """ Update variables for the current Newton iteration.

        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE]["previous_iterate"]
        are updated for
            - mortar displacements
            - contact traction
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector : np.ndarray
                solution vector for the current iterate.

        """
        var_mortar = self.mortar_displacement_variable
        var_contact = self.contact_traction_variable

        assembler = self.assembler
        variable_names = []
        for pair in assembler.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in assembler.block_dof.items():
                g, name = pair
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    if name == var_mortar:
                        mortar_u = (solution_vector[dof[bi]: dof[bi + 1]]).copy()
                        data = self.gb.edge_props(g)
                        data[pp.STATE]["previous_iterate"][var_mortar] = mortar_u
                else:
                    # g is a node/grid (not edge)
                    # For the fractures, update the contact force
                    if (g.dim < self.Nd) and (name == var_contact):
                        contact = (solution_vector[dof[bi]: dof[bi + 1]]).copy()
                        data = self.gb.node_props(g)
                        data[pp.STATE]["previous_iterate"][var_contact] = contact

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.gb.dim_min() < self.Nd

    # --- Helper methods ---

    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """ Compute the stress in the highest-dimensional grid based on the displacement
        states in that grid, adjacent interfaces and global boundary conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]["stress"].

        Parameters:
            previous_iterate : bool
                If True, use values from previous iteration to compute the stress.
                Default: False.

        """
        # TODO: Currently 'reconstruct_stress' does not work if 'previous_iterate = True'
        #  since the displacement variable on Nd-grid is not stored in "previous_iterate".
        if previous_iterate is True:
            raise ValueError("Not yet implemented.")
        g = self._nd_grid()
        d = self.gb.node_props(g)
        key_m = self.mechanics_parameter_key
        var_m = self.displacement_variable
        var_mortar = self.mortar_displacement_variable

        mpsa = pp.Mpsa(self.mechanics_parameter_key)

        if previous_iterate:
            u = d[pp.STATE]["previous_iterate"][var_m]
        else:
            u = d[pp.STATE][var_m]

        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key_m]

        # Stress contribution from internal cell center displacements
        stress = matrix_dictionary[mpsa.stress_matrix_key] * u

        # Contributions from global boundary conditions
        bound_stress_discr = matrix_dictionary[mpsa.bound_stress_matrix_key]
        global_bc_val = d[pp.PARAMETERS][key_m]["bc_values"]
        stress += bound_stress_discr * global_bc_val

        # Contributions from the mortar displacement variables
        for e, d_e in self.gb.edges():
            # Only contributions from interfaces to the highest dimensional grid
            mg: pp.MortarGrid = d_e["mortar_grid"]
            if mg.dim == self.Nd - 1:
                u_e = d_e[pp.STATE][var_mortar]

                stress += bound_stress_discr * mg.mortar_to_master_avg(nd=self.Nd) * u_e

        d[pp.STATE]["stress"] = stress

    def reconstruct_local_displacement_jump(self, data_edge: Dict, from_iterate: bool = True):
        """ Reconstruct the displacement jump in local coordinates.

        Parameters:
            data_edge : Dict
                The dictionary on the gb edge. Should contain
                    - a mortar grid
                    - a projection, obtained by calling
                    pp.contact_conditions.set_projections(self.gb)
            from_iterate : bool
                Whether to fetch displacement from state or previous
                iterate.
        Returns:
            u_mortar_local : np.ndarray (ambient_dim x g_l.num_cells)
                First 1-2 dimensions are in the tangential direction
                of the fracture, last dimension is normal.

        """
        var_mortar = self.mortar_displacement_variable
        nd = self.Nd

        mg: pp.MortarGrid = data_edge["mortar_grid"]
        if from_iterate:
            mortar_u = data_edge[pp.STATE]["previous_iterate"][var_mortar]
        else:
            mortar_u = data_edge[pp.STATE][var_mortar]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd)
            * mg.sign_of_mortar_sides(nd=nd)
            * mortar_u
        )
        projection: pp.TangentialNormalProjection = data_edge["tangential_normal_projection"]

        # Rotated displacement jumps. these are in the local coordinates, on the fracture.
        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        return u_mortar_local.reshape((nd, -1), order="F")

    # --- Exporting and visualization ---

    def set_viz(self) -> None:
        """ Set exporter for visualization """
        super().set_viz()

        self.u_exp = "u_exp"  # noqa
        self.traction_exp = "traction_exp"  # noqa
        self.normal_frac_u = "normal_frac_u"  # noqa
        self.tangential_frac_u = "tangential_frac_u"  # noqa
        self.stress_exp = "stress_exp"  # noqa

        self.export_fields.extend([
            self.u_exp,
            self.traction_exp,
            self.normal_frac_u,
            self.tangential_frac_u,
            # Cannot save variables that are defined on faces:
            # self.stress_exp,
        ])

    def save_matrix_stress(self, from_iterate: bool = False) -> None:
        """ Save upscaled matrix stress state to a class attribute """
        self.reconstruct_stress(from_iterate)
        gb = self.gb
        nd = self.Nd
        ss = self.params.scalar_scale

        for g, d in gb:
            state = d[pp.STATE]  # type: Dict[str, np.ndarray]
            if g.dim == nd:
                stress_exp = state["stress"].reshape((nd, -1), order="F").copy() * ss
            else:
                stress_exp = np.zeros((nd, g.num_faces))
            state[self.stress_exp] = stress_exp

    def save_frac_jump_data(self, from_iterate: bool = False) -> None:
        """ Save upscaled normal and tangential jumps to a class attribute
        Inspired by Keilegavlen 2019 (code)
        """
        gb = self.gb
        nd_grid = self._nd_grid()
        ls = self.params.length_scale

        for g, d in gb:
            if g.dim == self.Nd - 1:
                # Get edge of node pair
                edge = (g, nd_grid)
                data_edge = gb.edge_props(edge)

                u_mortar_local = (
                    self.reconstruct_local_displacement_jump(
                        data_edge, from_iterate
                    ).copy() * ls
                )

                # Jump distances in each cell
                tangential_jump = np.linalg.norm(u_mortar_local[:-1, :], axis=0)
                normal_jump = np.abs(u_mortar_local[-1, :])
            else:
                tangential_jump = np.zeros(g.num_cells)
                normal_jump = np.zeros(g.num_cells)

            d[pp.STATE][self.normal_frac_u] = normal_jump
            d[pp.STATE][self.tangential_frac_u] = tangential_jump

    def save_matrix_displacements(self) -> None:
        """ Save upscaled matrix displacements"""
        gb = self.gb
        nd = self.Nd
        var_m = self.displacement_variable
        ls = self.params.length_scale

        for g, d in gb:
            state = d[pp.STATE]  # type: Dict[str, np.ndarray]
            if g.dim == nd:
                u_exp = state[var_m].reshape((nd, -1), order="F").copy() * ls
            else:
                u_exp = np.zeros((nd, g.num_cells))
            state[self.u_exp] = u_exp

    def save_contact_traction(self) -> None:
        """ Save upscaled fracture contact traction"""
        gb = self.gb
        var_contact = self.contact_traction_variable
        ls = self.params.length_scale
        ss = self.params.scalar_scale

        for g, d in gb:
            state = d[pp.STATE]  # type: Dict[str, np.ndarray]
            if g.dim == self.Nd - 1:
                traction = state[var_contact].reshape((self.Nd, -1), order="F")
                traction_exp = traction * ss * (ls ** 2)
            else:
                traction_exp = np.zeros((self.Nd, g.num_cells))
            state[self.traction_exp] = traction_exp

    def export_step(self, write_vtk: bool = True) -> None:
        """ Export a visualization step"""
        super().export_step(write_vtk=False)
        self.save_frac_jump_data()
        self.save_matrix_displacements()
        self.save_contact_traction()
        self.save_matrix_stress()

        if write_vtk:
            self.viz.write_vtk(data=self.export_fields, time_dependent=False)

    def after_simulation(self):
        """ Called after a completed simulation """
        logger.info(f"Solution exported to folder \n {self.params.folder_name}")


class ContactMechanicsISC(ContactMechanics):
    """ Implementation of ContactMechanics for ISC

    Run a Contact Mechanics model from porepy on the geometry
    defined by the In-Situ Stimulation and Circulation (ISC)
    project at the Grimsel Test Site (GTS).
    """

    def __init__(self, params: dict):
        """ Initialize a Contact Mechanics model for GTS-ISC geometry.

        Parameters
        ----------
        params : dict
            Should contain the following key-value pairs:
                viz_folder_name : str
                    Absolute path to folder where grid and results will be stored
                --- SIMULATION RELATED PARAMETERS ---
                mesh_args : dict[str, int]
                    Arguments for meshing of domain.
                    Required keys: 'mesh_size_frac', 'mesh_size_min, 'mesh_size_bound'
                bounding_box : dict[str, int]
                    Bounding box of domain. ** Unscaled **
                    Required keys: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
                shearzone_names : List[str]
                    Which shear-zones to include in simulation
                length_scale, scalar_scale : float
                    Length scale and scalar variable scale.
                solver : str : {'direct', 'pyamg'}
                    Which solver to use
                --- PHYSICAL PARAMETERS ---
                stress : np.ndarray
                    Stress tensor for boundary conditions

                --- FOR TESTING ---
                _gravity_bc : bool [Default: True}
                    turn gravity effects on Neumann bc on/off
                _gravity_src : bool [Default: True]
                    turn gravity effects in mechanical source on/off
        """

        logger.info(
            f"Initializing contact mechanics on ISC dataset "
            f"at {pendulum.now().to_atom_string()}"
        )
        # Root name of solution files
        self.file_name = "main_run"

        # --- FRACTURES ---
        self.shearzone_names = params.get("shearzone_names")
        self.n_frac = len(self.shearzone_names) if self.shearzone_names else 0
        # Initialize data storage for normal and tangential jumps
        self.u_jumps_tangential = np.empty((1, self.n_frac))
        self.u_jumps_normal = np.empty((1, self.n_frac))

        # --- PHYSICAL PARAMETERS ---
        self.stress = params.get("stress")
        self.set_rock()

        # --- COMPUTATIONAL MESH ---
        self.mesh_args = params.get("mesh_args")
        self.box = params.get("bounding_box")
        self.gb = None
        self.Nd = None
        self._network = None

        # --- GTS-ISC DATA ---
        self.isc = gts.ISCData()

        # params should have 'folder_name' and 'linear_solver' as keys
        super().__init__(params=params)

        # Scaling coefficients (set after __init__ call because
        # ContactMechanicsISC overwrites the values.
        self.scalar_scale = params.get("scalar_scale")
        self.length_scale = params.get("length_scale")

        #
        # --- ADJUST CERTAIN PARAMETERS FOR TESTING ---

        # Turn on/off mechanical gravity term
        self._gravity_src = params.get("_gravity_src", False)

        # Turn on/off gravitational effects on (Neumann) mechanical boundary conditions
        self._gravity_bc = params.get("_gravity_bc", False)

    @trace(logger)
    def create_grid(self):
        """ Create a GridBucket of a 3D domain with fractures
        defined by the ISC dataset.

        The method requires the following attributes:
            mesh_args : dict
                Containing the mesh sizes.
            length_scale : float
                length scaling coefficient
            box : dict
                Bounding box of domain. Unscaled
            viz_folder_name : str
                path for where to store mesh files.

        Returns
        -------
        None

        Attributes
        ----------
        The method assigns the following attributes to self:
            gb : pp.GridBucket
                The produced grid bucket.
            box : dict
                The bounding box of the domain, defined through
                minimum and maximum values in each dimension.
            Nd : int
                The dimension of the matrix, i.e., the highest
                dimension in the grid bucket.
            network : pp.FractureNetwork3d
                fracture network of the domain

        """
        # Create grid
        gb, scaled_box, network = create_grid(
            self.mesh_args,
            self.length_scale,
            self.box,
            self.shearzone_names,
            self.viz_folder_name,
        )
        self.gb = gb
        self.box = scaled_box
        self.network = network

    def set_grid(self, gb: pp.GridBucket):
        """ Set a new grid
        """
        self.gb = gb
        pp.contact_conditions.set_projections(self.gb)
        self.n_frac = gb.get_grids(lambda _g: _g.dim == self.Nd - 1).size
        self.gb.add_node_props(keys=["name"])  # Add 'name' as node prop to all grids.

        # Set fracture grid names
        if self.n_frac > 0:
            fracture_grids = self.gb.get_grids(lambda g: g.dim == self.Nd - 1)
            for i, sz_name in enumerate(self.shearzone_names):
                self.gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)

    def grids_by_name(self, name, key="name") -> np.ndarray:
        """ Get grid by grid bucket node property 'name'

        """
        gb = self.gb
        grids = gb.get_grids(lambda g: gb.node_props(g, key) == name)

        return grids

    def faces_to_fix(self, g: pp.Grid):
        """ Fix some boundary faces to dirichlet to ensure unique solution to problem.

        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.

        Credits: Keilegavlen et al (2019) - Source code.
        """
        all_bf, *_ = self.domain_boundary_sides(g)
        point = np.array(
            [
                [(self.box["xmin"] + self.box["xmax"]) / 2],
                [(self.box["ymin"] + self.box["ymax"]) / 2],
                [self.box["zmin"]],
            ]
        )
        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argpartition(distances, self.Nd)[: self.Nd]
        old_indexes = np.argsort(distances)
        assert np.allclose(
            np.sort(indexes), np.sort(old_indexes[: self.Nd])
        )  # Temporary: test new argpartition method
        faces = all_bf[indexes[: self.Nd]]
        return faces

    def bc_type(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """ We set Neumann values on all but a few boundary faces.
        Fracture faces are set to Dirichlet.

        Three boundary faces (see self.faces_to_fix())
        are set to 0 displacement (Dirichlet).
        This ensures a unique solution to the problem.
        Furthermore, the fracture faces are set to 0 displacement (Dirichlet).
        """

        all_bf, *_ = self.domain_boundary_sides(g)
        faces = self.faces_to_fix(g)
        bc = pp.BoundaryConditionVectorial(g, faces, ["dir"] * len(faces))
        fracture_faces = g.tags["fracture_faces"]
        bc.is_neu[:, fracture_faces] = False
        bc.is_dir[:, fracture_faces] = True
        return bc

    def bc_values(self, g: pp.Grid) -> np.array:
        """ Mechanical stress values as ISC

        All faces are Neumann, except 3 faces fixed
        by self.faces_to_fix(g), which are Dirichlet.
        """
        # Retrieve the domain boundary
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)

        # Boundary values
        bc_values = np.zeros((g.dim, g.num_faces))

        # --- mechanical state ---
        # Get outward facing normal vectors for domain boundary, weighted for face area

        # 1. Get normal vectors on the faces. These are already weighed by face area.
        bf_normals = g.face_normals
        # 2. Adjust direction so they face outwards
        flip_normal_to_outwards = np.where(g.cell_face_as_dense()[0, :] >= 0, 1, -1)
        outward_normals = bf_normals * flip_normal_to_outwards
        bf_stress = np.dot(self.stress, outward_normals[:, all_bf])
        bc_values[:, all_bf] += bf_stress / self.scalar_scale  # Mechanical stress

        # --- gravitational forces ---
        # See init-method to turn on/off gravity effects (Default: OFF)
        if self._gravity_bc:
            lithostatic_bc = self._adjust_stress_for_depth(g, outward_normals)

            # NEUMANN
            bc_values[:, all_bf] += lithostatic_bc[:, all_bf] / self.scalar_scale

        # DIRICHLET
        faces = self.faces_to_fix(g)
        bc_values[:, faces] = 0  # / self.length scale

        return bc_values.ravel("F")

    def _adjust_stress_for_depth(self, g: pp.Grid, outward_normals):
        """ Compute a stress tensor purely accounting for depth.

        The true_stress_depth determines at what depth we consider
        the given stress tensor (self.stress) to be equal to
        the given value. This can in principle be any number,
        but will usually be zmin <= true_stress_depth <= zmax

        Need the grid g, and outward_normals (see method above).

        Returns the marginal **traction** for the given face (g.face_centers)

        # TODO This could perhaps be integrated directly in the above method.
        """
        # TODO: Only do computations over 'all_bf'.
        # TODO: Test this method
        true_stress_depth = self.box["zmax"] * self.length_scale

        # We assume the relative sizes of all stress components scale with sigma_zz.
        # Except if sigma_zz = 0, then we don't scale.
        if np.abs(self.stress[2, 2]) < 1e-12:
            logger.critical("The stress scaler is set to 0 since stress[2, 2] = 0")
            stress_scaler = np.zeros(self.stress.shape)
        else:
            stress_scaler = self.stress / self.stress[2, 2]

        # All depths are translated in terms of the assumed depth
        # of the given stress tensor.
        relative_depths = g.face_centers[2] * self.length_scale - true_stress_depth
        rho_g_h = self.rock.lithostatic_pressure(relative_depths)
        lithostatic_stress = stress_scaler.dot(np.multiply(outward_normals, rho_g_h))
        return lithostatic_stress

    def source(self, g: pp.Grid) -> np.array:
        """ Gravity term.

        Gravity points downward, but we give the term
        on the RHS of the equation, thus we take the
        negative (i.e. the vector given will be
        pointing upwards)
        """
        # See init-method to turn on/off gravity effects (Default: OFF)
        if not self._gravity_src:
            return np.zeros(self.Nd * g.num_cells)

        # Gravity term
        values = np.zeros((self.Nd, g.num_cells))
        scaling = self.length_scale / self.scalar_scale
        values[2] = self.rock.lithostatic_pressure(g.cell_volumes) * scaling
        return values.ravel("F")

    def set_rock(self):
        """ Set rock properties of the ISC rock.
        """

        self.rock = GrimselGranodiorite()

    def _set_friction_coefficient(self, g: pp.Grid):
        """ The friction coefficient is uniform, and equal to 1.

        Assumes self.set_rock() is called
        """
        return np.ones(g.num_cells) * self.rock.FRICTION_COEFFICIENT

    def set_parameters(self):
        """
        Set the parameters for the simulation.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = self.rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = self.rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # BC and source values
                bc = self.bc_type(g)
                bc_val = self.bc_values(g)
                source_val = self.source(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        # "max_memory": 7e7,
                        # "inverter": python,
                    },
                )

            elif g.dim == self.Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )
        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    def set_viz(self):
        """ Set exporter for visualization """
        self.viz = pp.Exporter(
            self.gb, file_name=self.file_name, folder_name=self.viz_folder_name
        )
        # list of time steps to export with visualization.

        self.u_exp = "u_exp"
        self.traction_exp = "traction_exp"
        self.normal_frac_u = "normal_frac_u"
        self.tangential_frac_u = "tangential_frac_u"

        self.export_fields = [
            self.u_exp,
            self.traction_exp,
            self.normal_frac_u,
            self.tangential_frac_u,
        ]

    @timer(logger)
    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """

        self.create_grid()
        self.Nd = self.gb.dim_max()
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

    def export_step(self):
        """ Export a step

        Inspired by Keilegavlen 2019 (code)
        """

        self.save_frac_jump_data()  # Save fracture jump data to pp.STATE
        gb = self.gb
        Nd = self.Nd

        for g, d in gb:

            if g.dim != 2:  # We only define tangential jumps in 2D fractures
                d[pp.STATE][self.normal_frac_u] = np.zeros(g.num_cells)
                d[pp.STATE][self.tangential_frac_u] = np.zeros(g.num_cells)

            if g.dim == Nd:  # On matrix
                u = (
                    d[pp.STATE][self.displacement_variable]
                    .reshape((Nd, -1), order="F")
                    .copy()
                    * self.length_scale
                )

                if g.dim != 3:  # Only called if solving a 2D problem
                    u = np.vstack(u, np.zeros(u.shape[1]))

                d[pp.STATE][self.u_exp] = u

                d[pp.STATE][self.traction_exp] = np.zeros(d[pp.STATE][self.u_exp].shape)

            else:  # In fractures or intersection of fractures (etc.)
                g_h = gb.node_neighbors(g, only_higher=True)[
                    0
                ]  # Get the higher-dimensional neighbor
                if g_h.dim == Nd:  # In a fracture
                    data_edge = gb.edge_props((g, g_h))
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge=data_edge, from_iterate=True
                    ).copy()
                    u_mortar_local = u_mortar_local * self.length_scale

                    traction = d[pp.STATE][self.contact_traction_variable].reshape(
                        (Nd, -1), order="F"
                    )

                    if g.dim == 2:
                        d[pp.STATE][self.u_exp] = u_mortar_local
                        d[pp.STATE][self.traction_exp] = traction
                    # TODO: Check when this statement is actually called
                    else:
                        # Only called if solving a 2D problem
                        # (i.e. this is a 0D fracture intersection)
                        d[pp.STATE][self.u_exp] = np.vstack(
                            u_mortar_local, np.zeros(u_mortar_local.shape[1])
                        )
                else:  # In a fracture intersection
                    d[pp.STATE][self.u_exp] = np.zeros((Nd, g.num_cells))
                    d[pp.STATE][self.traction_exp] = np.zeros((Nd, g.num_cells))
        self.viz.write_vtk(
            data=self.export_fields, time_dependent=False
        )  # Write visualization

    def save_frac_jump_data(self):
        """ Save normal and tangential jumps to a class attribute
        Inspired by Keilegavlen 2019 (code)
        """
        if self.n_frac > 0:
            gb = self.gb
            Nd = self.Nd
            n = self.n_frac

            tangential_u_jumps = np.zeros((1, n))
            normal_u_jumps = np.zeros((1, n))

            for frac_num, frac_name in enumerate(self.shearzone_names):
                g_lst = gb.get_grids(lambda _g: gb.node_props(_g)["name"] == frac_name)
                assert (
                    len(g_lst) == 1
                )  # Currently assume each fracture is uniquely named.

                g = g_lst[0]
                g_h = gb.node_neighbors(g, only_higher=True)[
                    0
                ]  # Get higher-dimensional neighbor
                assert g_h.dim == Nd  # We only operate on fractures of dim Nd-1.

                data_edge = gb.edge_props((g, g_h))
                u_mortar_local = (
                    self.reconstruct_local_displacement_jump(
                        data_edge=data_edge, from_iterate=True
                    ).copy()
                    * self.length_scale
                )

                # Jump distances in each cell
                tangential_jump = np.linalg.norm(
                    u_mortar_local[:-1, :], axis=0
                )  # * self.length_scale inside norm.
                normal_jump = np.abs(u_mortar_local[-1, :])  # * self.length_scale

                # Save jumps to state
                d = gb.node_props(g)
                d[pp.STATE][self.normal_frac_u] = normal_jump
                d[pp.STATE][self.tangential_frac_u] = tangential_jump

                # TODO: "Un-scale" these quantities
                # Ad-hoc average normal and tangential jump "estimates"
                # TODO: Find a proper way to express the "total"
                #  displacement of a fracture
                avg_tangential_jump = np.sum(tangential_jump * g.cell_volumes) / np.sum(
                    g.cell_volumes
                )
                avg_normal_jump = np.sum(normal_jump * g.cell_volumes) / np.sum(
                    g.cell_volumes
                )

                tangential_u_jumps[0, frac_num] = avg_tangential_jump
                normal_u_jumps[0, frac_num] = avg_normal_jump

            self.u_jumps_tangential = np.concatenate(
                (self.u_jumps_tangential, tangential_u_jumps)
            )
            self.u_jumps_normal = np.concatenate((self.u_jumps_normal, normal_u_jumps))

    def after_newton_iteration(self, solution_vector):
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE]["previous_iterate"] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        OVERWRITES parent to remove writing to vtk.

        Parameters:
            assembler (pp.Assembler): assembler for self.gb.
            solution_vector (np.array): solution vector for the current iterate.

        Returns:
            (np.array): displacement solution vector for the Nd grid.

        """
        self.update_state(solution_vector)

    def after_newton_convergence(self, solution, errors, iteration_counter):
        """ What to do at the end of a step."""
        self.assembler.distribute_variable(solution)
        self.export_step()

    def _depth(self, coords):
        """
        Unscaled depth. We center the domain at 480m below the surface.
        (See Krietsch et al, 2018a)
        """
        return 480.0 * pp.METER - self.length_scale * coords[2]
