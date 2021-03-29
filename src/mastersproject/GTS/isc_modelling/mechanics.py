import logging
from typing import Dict, Tuple

import numpy as np

import porepy as pp
from GTS.isc_modelling.general_model import CommonAbstractModel
from GTS.isc_modelling.parameter import BaseParameters
from mastersproject.util.logging_util import timer
from porepy.params.data import add_nonpresent_dictionary

logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class Mechanics(CommonAbstractModel):
    def __init__(self, params: BaseParameters):
        """General mechanics model for static contact mechanics

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
        """Set homogeneous conditions on all boundary faces."""
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
        """The friction coefficient is uniform, and equal to 1.

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
                        "p_reference": np.zeros(g.num_cells),
                        # "max_memory": 7e7,
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
        """Assign variables to the nodes and edges of the grid bucket."""
        gb = self.gb
        primary_vars = pp.PRIMARY_VARIABLES
        var_m = self.displacement_variable
        var_contact = self.contact_traction_variable
        var_mortar = self.mortar_displacement_variable

        for g, d in gb:
            add_nonpresent_dictionary(d, primary_vars)
            if g.dim == self.Nd:
                d[primary_vars].update(
                    {
                        var_m: {"cells": self.Nd},
                    }
                )

            elif g.dim == self.Nd - 1:
                d[primary_vars].update(
                    {
                        var_contact: {"cells": self.Nd},
                    }
                )

        for e, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            g_l, g_h = gb.nodes_of_edge(e)
            if g_h.dim == self.Nd:
                d[primary_vars].update(
                    {
                        var_mortar: {"cells": self.Nd},
                    }
                )

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
                d[discr_key].update(
                    {
                        var_m: {"mpsa": mpsa},
                    }
                )

            elif g.dim == nd - 1:
                d[discr_key].update(
                    {
                        var_contact: {"empty": empty_discr},
                    }
                )

        # Define the contact condition on the mortar grid
        coloumb = pp.ColoumbContact(key_m, nd, mpsa)
        contact = pp.PrimalContactCoupling(key_m, mpsa, coloumb)

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            add_nonpresent_dictionary(d, coupling_discr_key)
            if g_h.dim == nd:
                d[coupling_discr_key].update(
                    {
                        self.friction_coupling_term: {
                            g_h: (var_m, "mpsa"),
                            g_l: (var_contact, "empty"),
                            e: (var_mortar, contact),
                        },
                    }
                )

    @timer(logger, level="INFO")
    def discretize(self) -> None:
        """Discretize all terms"""
        if not self.assembler:
            self.dof_manager = pp.DofManager(self.gb)
            self.assembler = pp.Assembler(self.gb, self.dof_manager)

        self.assembler.discretize()

    # --- Initial condition ---

    def initial_mechanics_condition(self) -> None:
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction,
        and -1 (that is, in contact) in the normal direction.

        We initialize pp.ITERATE for the contact traction and
        mortar displacement since they are to be updated every Newton
        iteration.

        """
        gb = self.gb
        var_m = self.displacement_variable
        var_contact = self.contact_traction_variable
        var_mortar = self.mortar_displacement_variable
        state, iterate = pp.STATE, pp.ITERATE

        for g, d in gb:
            add_nonpresent_dictionary(d, state)
            if g.dim == self.Nd:
                # Initialize displacement variable
                initial_displacement_value = np.zeros(g.num_cells * self.Nd)
                d[state].update(
                    {
                        var_m: initial_displacement_value,
                        iterate: {var_m: initial_displacement_value.copy()}
                    }
                )

            elif g.dim == self.Nd - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (np.zeros((g.dim, g.num_cells)), -1 * np.ones(g.num_cells))
                ).ravel(order="F")
                d[state].update(
                    {
                        iterate: {var_contact: traction},
                        var_contact: traction,
                    }
                )

        for e, d in self.gb.edges():
            add_nonpresent_dictionary(d, state)

            mg: pp.MortarGrid = d["mortar_grid"]
            if mg.dim == self.Nd - 1:
                size = mg.num_cells * self.Nd
                d[state].update(
                    {
                        var_mortar: np.zeros(size),
                        iterate: {var_mortar: np.zeros(size)},
                    }
                )

    # --- Simulation and solvers ---

    @timer(logger, level="INFO")
    def prepare_simulation(self) -> None:
        """Is run prior to a time-stepping scheme. Use this to initialize
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
        """Initialize linear solver

        Currently, we only consider the direct solver.
        See also self.assemble_and_solve_linear_system()
        """

        # # Compute exact condition number:
        # A, _ = self.assembler.assemble_matrix_rhs()
        # cond = spla.norm(A, 1) * spla.norm(spla.inv(A), 1)
        # logger.info(f"Exact condition number: {cond:.2e}")

        if self.params.linear_solver == "direct":
            """In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.

            """
            pass

        else:
            raise ValueError(f"Unknown linear solver {self.params.linear_solver}")

    def _check_convergence_mechanics(
        self, solution, prev_solution, init_solution, nl_params
    ):
        """ Check convergence and compute error of matrix displacement variable"""
        var_m = self.displacement_variable
        g_max = self._nd_grid()
        ls = self.params.length_scale

        # Extract convergence tolerance
        tol_convergence = nl_params.get("convergence_tol")
        converged = False
        diverged = False

        # Get the solution from current and previous iterates,
        # as well as the initial guess.
        mech_dof = self.dof_manager.dof_ind(g_max, var_m)
        u_mech_now = solution[mech_dof] * ls
        u_mech_prev = prev_solution[mech_dof] * ls
        u_mech_init = init_solution[mech_dof] * ls

        # Calculate norms
        difference_in_iterates_mech = np.sqrt(np.sum((u_mech_now - u_mech_prev) ** 2)) / u_mech_now.size
        difference_from_init_mech = np.sqrt(np.sum((u_mech_now - u_mech_init) ** 2)) / u_mech_now.size

        # Calculate errors
        scaled_convergence_tol = tol_convergence * ls
        absolute_convergence = difference_in_iterates_mech < scaled_convergence_tol
        relative_convergence = difference_in_iterates_mech < tol_convergence * difference_from_init_mech
        abs_error = difference_in_iterates_mech
        rel_error = difference_in_iterates_mech / difference_from_init_mech
        if absolute_convergence:
            converged = True
            error_mech = abs_error
        elif relative_convergence:
            converged = True
            error_mech = rel_error
        else:
            error_mech = rel_error

        logger.info(
            f"3D displacement error: "
            f"absolute={abs_error:.2e} {'<' if absolute_convergence else '>'} {scaled_convergence_tol:.2e}. "
            f"relative={rel_error:.2e} {'<' if relative_convergence else '>'} {tol_convergence:.2e} "
            f"({'Converged' if (absolute_convergence or relative_convergence) else 'Did not converge'})."
        )

        if difference_in_iterates_mech > 1e30:
            diverged = True

        return error_mech, converged, diverged

    def _check_convergence_contact(
        self, solution, prev_solution, init_solution, nl_params
    ):
        """ Check convergence and compute error of contact traction variable"""

        contact_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)
            if g_h.dim == self.Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.dof_manager.dof_ind(e[1], self.contact_traction_variable),
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
        difference_in_iterates_contact = (
            np.sum((contact_now - contact_prev) ** 2) * ss * ls ** 2
        )
        difference_from_init_contact = (
            np.sum((contact_now - contact_init) ** 2) * ss * ls ** 2
        )

        tol_convergence = nl_params["convergence_tol"]

        converged = False
        diverged = False
        error_type = "relative"

        # The if is intended to avoid division through zero
        if (
            contact_norm < tol_convergence
            and difference_in_iterates_contact < tol_convergence
        ):
            converged = True
            error_contact = difference_in_iterates_contact
            error_type = "absolute"
        else:
            error_contact = (
                difference_in_iterates_contact / difference_from_init_contact
            )

        logger.info(f"Error in contact force is {error_contact:.6e} ({error_type}).")
        logger.info(
            f"Contact force {'converged' if converged else 'did not converge'}."
        )

        logger.info("DISABLE CONVERGENCE CHECK FOR CONTACT FORCE PENDING DEBUGGING")
        converged = True
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
            solution,
            prev_solution,
            init_solution,
            nl_params,
        )
        _, converged_contact, diverged_contact = self._check_convergence_contact(
            solution,
            prev_solution,
            init_solution,
            nl_params,
        )

        converged = converged_mech and converged_contact
        diverged = diverged_mech or diverged_contact

        # Only return matrix displacement error for now
        return error_mech, converged, diverged

    # --- Newton iterations ---

    def before_newton_loop(self) -> None:
        """Will be run before entering a Newton loop.

        Discretize time-dependent quantities etc.
        """
        self.set_mechanics_parameters()

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        term_filter = pp.assembler_filters.ListFilter(
            term_list=[self.friction_coupling_term]
        )
        self.assembler.discretize(term_filter)

    @pp.time_logger(sections=module_sections)
    def update_iterate(self, solution_vector: np.ndarray) -> None:
        """Update variables for the current Newton iteration.

        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE]
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
        var_displacement = self.displacement_variable

        dof_manager = self.dof_manager
        variable_names = []
        for pair in dof_manager.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in dof_manager.block_dof.items():
                g, name = pair
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    if name == var_mortar:
                        mortar_u = (solution_vector[dof[bi] : dof[bi + 1]]).copy()
                        data = self.gb.edge_props(g)
                        data[pp.STATE][pp.ITERATE][var_mortar] = mortar_u
                else:
                    # g is a node/grid (not edge)
                    data = self.gb.node_props(g)
                    if (g.dim == self.Nd) and name == var_displacement:
                        # In the matrix, update displacement
                        displacement = solution_vector[dof[bi] : dof[bi + 1]]
                        data[pp.STATE][pp.ITERATE][var_displacement] = displacement.copy()
                    elif (g.dim < self.Nd) and (name == var_contact):
                        # For the fractures, update the contact force
                        contact = (solution_vector[dof[bi] : dof[bi + 1]]).copy()
                        data[pp.STATE][pp.ITERATE][var_contact] = contact

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.gb.dim_min() < self.Nd

    # --- Helper methods ---

    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """Compute the stress in the highest-dimensional grid based on the displacement
        states in that grid, adjacent interfaces and global boundary conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]["stress"].

        Parameters:
            previous_iterate : bool
                If True, use values from previous iteration to compute the stress.
                Default: False.

        """
        # TODO: Currently 'reconstruct_stress' does not work if 'previous_iterate = True'
        #  since the displacement variable on Nd-grid is not stored in pp.ITERATE.
        if previous_iterate is True:
            raise ValueError("Not yet implemented.")
        g = self._nd_grid()
        d = self.gb.node_props(g)
        key_m = self.mechanics_parameter_key
        var_m = self.displacement_variable
        var_mortar = self.mortar_displacement_variable

        mpsa = pp.Mpsa(self.mechanics_parameter_key)

        if previous_iterate:
            u = d[pp.STATE][pp.ITERATE][var_m]
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

                stress += (
                    bound_stress_discr * mg.mortar_to_primary_avg(nd=self.Nd) * u_e
                )

        d[pp.STATE]["stress"] = stress

    def reconstruct_local_displacement_jump(
        self,
        data_edge: Dict,
        projection: pp.TangentialNormalProjection,
        from_iterate: bool = True,
    ):
        """Reconstruct the displacement jump in local coordinates.

        Parameters:
            data_edge : Dict
                The dictionary on the gb edge. Should contain a mortar grid.
            projection : pp.TangentialNormalProjection
                projection operator. Stored in lower-dimensional grid data.
                Computed with pp.contact_conditions.set_projections(gb)
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
            mortar_u = data_edge[pp.STATE][pp.ITERATE][var_mortar]
        else:
            mortar_u = data_edge[pp.STATE][var_mortar]

        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=nd)
            * mg.sign_of_mortar_sides(nd=nd)
            * mortar_u
        )

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
        self.fracture_state = "fracture_state"  # noqa
        self.tangential_frac_traction = "tangential_frac_traction"  # noqa
        self.normal_frac_traction = "normal_frac_traction"  # noqa
        self.slip_tendency = "slip_tendency"  # noqa
        self.cell_volumes = "cell_volumes"  # noqa

        self.export_fields.extend(
            [
                self.u_exp,
                self.traction_exp,
                self.normal_frac_u,
                self.tangential_frac_u,
                self.fracture_state,
                self.tangential_frac_traction,
                self.normal_frac_traction,
                self.slip_tendency,
                self.cell_volumes,
                # Cannot save variables that are defined on faces:
                # self.stress_exp,
            ]
        )

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

    def save_fracture_cell_state(self):
        """ Save state of fracture cells (open, sticking, gliding)"""
        gb = self.gb
        for g, d in gb:
            state = d[pp.STATE]
            if g.dim == self.Nd - 1:
                iterate = state[pp.ITERATE]
                penetration = iterate["penetration"]
                sliding = iterate["sliding"]
                _sliding = np.logical_and(sliding, penetration)
                _sticking = np.logical_and(np.logical_not(sliding), penetration)
                _open = np.logical_not(penetration)
                fracture_state = np.zeros(g.num_cells)
                fracture_state[_open] = 0
                fracture_state[_sticking] = 1
                fracture_state[_sliding] = 2
            else:
                fracture_state = np.zeros(g.num_cells)
            state[self.fracture_state] = fracture_state

    def save_frac_jump_data(self, from_iterate: bool = False) -> None:
        """Save upscaled normal and tangential jumps to a class attribute
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
                projection = d["tangential_normal_projection"]

                u_mortar_local = (
                    self.reconstruct_local_displacement_jump(
                        data_edge, projection, from_iterate
                    ).copy()
                    * ls
                )

                # Jump distances in each cell
                tangential_jump = np.linalg.norm(u_mortar_local[:-1, :], axis=0)
                normal_jump = np.abs(u_mortar_local[-1, :])
            else:
                tangential_jump = np.zeros(g.num_cells)
                normal_jump = np.zeros(g.num_cells)

            d[pp.STATE][self.normal_frac_u] = normal_jump
            d[pp.STATE][self.tangential_frac_u] = tangential_jump

    def save_global_displacements(self) -> None:
        """ Save upscaled global displacements"""
        gb = self.gb
        nd = self.Nd
        var_m, var_mortar = (
            self.displacement_variable,
            self.mortar_displacement_variable,
        )
        ls = self.params.length_scale

        for g, d in gb:
            state = d[pp.STATE]  # type: Dict[str, np.ndarray]
            if g.dim == nd:
                u_exp = state[var_m].reshape((nd, -1), order="F").copy() * ls
            elif g.dim == nd - 1:
                # Save fracture displacements in global coordinates
                data_edge = gb.edge_props((g, self._nd_grid()))
                mg = data_edge["mortar_grid"]
                mortar_u = data_edge[pp.STATE][var_mortar]
                displacement_jump_global_coord = (
                    mg.mortar_to_secondary_avg(nd=nd)
                    * mg.sign_of_mortar_sides(nd=nd)
                    * mortar_u
                )
                u_mortar_global = displacement_jump_global_coord.reshape(
                    (nd, -1),
                    order="F",
                )
                u_exp = u_mortar_global * ls
            else:
                u_exp = np.zeros((nd, g.num_cells))
            state[self.u_exp] = u_exp

    def save_contact_traction(self) -> None:
        """ Save upscaled fracture contact traction"""
        gb = self.gb
        var_contact = self.contact_traction_variable
        ls = self.params.length_scale
        ss = self.params.scalar_scale
        scale = ss * (ls ** 2)

        for g, d in gb:
            state = d[pp.STATE]  # type: Dict[str, np.ndarray]
            if g.dim == self.Nd - 1:
                # Traction is already in local coordinates.
                traction = state[var_contact].reshape((self.Nd, -1), order="F")
                traction_exp = traction * scale
                # Extract norms of normal and tangential components
                tangential_traction_exp = np.linalg.norm(traction_exp[:-1, :], axis=0)
                normal_traction_exp = np.abs(traction_exp[-1, :])

            else:
                traction_exp = np.zeros((self.Nd, g.num_cells))
                tangential_traction_exp = np.zeros(g.num_cells)
                normal_traction_exp = np.zeros(g.num_cells)
            state[self.traction_exp] = traction_exp
            state[self.tangential_frac_traction] = tangential_traction_exp
            state[self.normal_frac_traction] = normal_traction_exp
            slip_tendency = np.divide(
                tangential_traction_exp,
                normal_traction_exp,
                out=np.zeros_like(tangential_traction_exp),
                where=normal_traction_exp != 0,
            )

            state[self.slip_tendency] = slip_tendency

    def export_step(self, write_vtk: bool = True) -> None:
        """ Export a visualization step"""
        super().export_step(write_vtk=False)
        self.save_frac_jump_data()
        self.save_global_displacements()
        self.save_contact_traction()
        self.save_matrix_stress()
        self.save_fracture_cell_state()

        # Export cell volumes
        for g, d in self.gb:
            volume_scale = self.params.length_scale ** self.Nd
            d[pp.STATE][self.cell_volumes] = g.cell_volumes * volume_scale

        if write_vtk:
            self.viz.write_vtk(data=self.export_fields, time_dependent=False)

    def after_simulation(self):
        """ Called after a completed simulation """
        logger.info(f"Solution exported to folder \n {self.params.folder_name}")
