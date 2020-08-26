import logging
from typing import Dict, Tuple

import numpy as np

import porepy as pp
from GTS.isc_modelling.flow import Flow
from GTS.isc_modelling.mechanics import Mechanics
from GTS.isc_modelling.parameter import BiotParameters
from mastersproject.util.logging_util import timer

logger = logging.getLogger(__name__)


class ContactMechanicsBiotBase(Flow, Mechanics):
    def __init__(self, params: BiotParameters):
        super().__init__(params)
        self.params = params

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

    # --- Set parameters ---

    def biot_alpha(self, g: pp.Grid) -> float:
        if g.dim == self.Nd:
            return self.params.alpha
        else:
            return 1

    def set_biot_parameters(self) -> None:
        """ Set parameters for the simulation"""
        self.set_scalar_parameters()
        self.set_mechanics_parameters()
        key_m = self.mechanics_parameter_key
        key_s = self.scalar_parameter_key

        for g, d in self.gb:
            params: pp.Parameters = d[pp.PARAMETERS]

            alpha = self.biot_alpha(g)
            mech_params = {
                "biot_alpha": alpha,
                "time_step": self.time_step,
            }
            scalar_params = {"biot_alpha": alpha}

            params.update_dictionaries(
                [key_m, key_s], [mech_params, scalar_params],
            )

    # --- Primary variables and discretizations ---

    def assign_biot_variables(self) -> None:
        """ Assign variables to the nodes and edges of the grid bucket"""
        self.assign_mechanics_variables()
        self.assign_scalar_variables()

    def assign_biot_discretizations(self) -> None:
        """ Assign discretizations to the nodes and edges of the grid bucket"""
        # Shorthand
        key_s, key_m = self.scalar_parameter_key, self.mechanics_parameter_key
        var_s, var_m = self.scalar_variable, self.displacement_variable
        var_mortar = self.mortar_displacement_variable
        discr_key, coupling_discr_key = pp.DISCRETIZATION, pp.COUPLING_DISCRETIZATION
        gb = self.gb

        # Assign flow and mechanics discretizations
        self.assign_scalar_discretizations()
        self.assign_mechanics_discretizations()

        # Coupling discretizations
        # All dimensions
        div_u_disc = pp.DivU(
            mechanics_keyword=key_m,
            flow_keyword=key_s,
            variable=var_m,
            mortar_variable=var_mortar,
        )
        # Nd
        grad_p_disc = pp.GradP(keyword=key_m)
        stabilization_disc_s = pp.BiotStabilization(keyword=key_s, variable=var_s)

        # Assign node discretizations
        for g, d in gb:
            if g.dim == self.Nd:
                d[discr_key][var_s].update(
                    {"stabilization": stabilization_disc_s,}
                )

                d[discr_key].update(
                    {
                        var_m + "_" + var_s: {"grad_p": grad_p_disc},
                        var_s + "_" + var_m: {"div_u": div_u_disc},
                    }
                )

        # Define edge discretizations for the mortar grid

        # fetch the previously created discretizations
        d_ = gb.node_props(self._nd_grid())
        mass_disc_s = d_[discr_key][var_s]["mass"]

        # Account for the mortar displacements effect on scalar balance in the matrix,
        # as an internal boundary contribution, fracture, aperture changes appear as a
        # source contribution.
        div_u_coupling = pp.DivUCoupling(var_m, div_u_disc, div_u_disc)
        # Account for the pressure contributions to the force balance on the fracture
        # (see contact_discr).
        # This discretization needs the keyword used to store the grad p discretization:
        grad_p_key = key_m
        matrix_scalar_to_force_balance = pp.MatrixScalarToForceBalance(
            grad_p_key, mass_disc_s, mass_disc_s
        )
        if self.subtract_fracture_pressure:
            fracture_scalar_to_force_balance = pp.FractureScalarToForceBalance(
                mass_disc_s, mass_disc_s
            )

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            if g_h.dim == self.Nd:
                d[coupling_discr_key].update(
                    {
                        "div_u_coupling": {
                            g_h: (
                                var_s,
                                "mass",
                            ),  # This is really the div_u, but this is not implemented
                            g_l: (var_s, "mass"),
                            e: (var_mortar, div_u_coupling),
                        },
                        "matrix_scalar_to_force_balance": {
                            g_h: (var_s, "mass"),
                            g_l: (var_s, "mass"),
                            e: (var_mortar, matrix_scalar_to_force_balance),
                        },
                    }
                )

                if self.subtract_fracture_pressure:
                    d[coupling_discr_key].update(
                        {
                            "fracture_scalar_to_force_balance": {
                                g_h: (var_s, "mass"),
                                g_l: (var_s, "mass"),
                                e: (
                                    var_mortar,
                                    fracture_scalar_to_force_balance,  # noqa
                                ),
                            }
                        }
                    )

    def _discretize_biot(self) -> None:
        """
        To save computational time, the full Biot equation (without contact mechanics)
        is discretized once. This is to avoid computing the same terms multiple times.
        """
        g = self._nd_grid()
        d = self.gb.node_props(g)
        biot = pp.Biot(
            mechanics_keyword=self.mechanics_parameter_key,
            flow_keyword=self.scalar_parameter_key,
            vector_variable=self.displacement_variable,
            scalar_variable=self.scalar_variable,
        )
        biot.discretize(g, d)

    @timer(logger, level="INFO")
    def discretize(self) -> None:
        """ Discretize all terms
        """
        if not self.assembler:
            self.assembler = pp.Assembler(self.gb)

        g_max = self.gb.grids_of_dimension(self.Nd)[0]

        logger.info("Discretize")

        # Discretization is a bit cumbersome, as the Biot discetization removes the
        # one-to-one correspondence between discretization objects and blocks in the matrix.
        # First, Discretize with the biot class
        self._discretize_biot()

        # Next, discretize term on the matrix grid not covered by the Biot discretization,
        # i.e. the source term
        # Here, we also discretize the edge terms in the entire gb
        self.assembler.discretize(grid=g_max, term_filter=["source"])

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for g, _ in self.gb:
            if g.dim < self.Nd:
                # No need to discretize edges here, this was done above.
                self.assembler.discretize(grid=g, edges=False)

    # --- Initial condition ---

    def initial_biot_condition(self) -> None:
        """ Set initial guess for the variables"""
        self.initial_scalar_condition()
        self.initial_mechanics_condition()

        # Set initial guess for mechanical bc values
        for g, d in self.gb:
            if g.dim == self.Nd:
                bc_values = d[pp.PARAMETERS][self.mechanics_parameter_key]["bc_values"]
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

    # --- Simulation and solvers ---

    @timer(logger)
    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self._prepare_grid()

        self.set_biot_parameters()
        self.assign_biot_variables()
        self.assign_biot_discretizations()
        self.initial_biot_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict,
    ) -> Tuple[np.ndarray, bool, bool]:
        error_s, converged_s, diverged_s = super(
            ContactMechanicsBiotBase, self
        ).check_convergence(solution, prev_solution, init_solution, nl_params,)
        error_m, converged_m, diverged_m = super(Flow, self).check_convergence(
            solution, prev_solution, init_solution, nl_params,
        )

        converged = converged_m and converged_s
        diverged = diverged_m or diverged_s

        # Return matrix displacement error only
        return error_m, converged, diverged

    # --- Newton iterations ---

    def before_newton_loop(self) -> None:
        """ Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_biot_parameters()

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        self.assembler.discretize(term_filter=[self.friction_coupling_term])

    def after_newton_convergence(self, solution, errors, iteration_counter) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        self.save_mechanical_bc_values()

    def save_mechanical_bc_values(self) -> None:
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key = self.mechanics_parameter_key
        g = self.gb.grids_of_dimension(self.Nd)[0]
        d = self.gb.node_props(g)
        d[pp.STATE][key]["bc_values"] = d[pp.PARAMETERS][key]["bc_values"].copy()

    # --- Exporting and visualization ---

    def export_step(self, write_vtk=True):
        super().export_step(write_vtk=False)

        if write_vtk:
            self.viz.write_vtk(
                data=self.export_fields, time_step=self.time
            )  # Write visualization
            self.export_times.append(self.time)

    # --- Helper methods ---

    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """
        Compute the stress in the highest-dimensional grid based on the displacement
        and pressure states in that grid, adjacent interfaces and global boundary
        conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]['stress'].

        Parameters:
            previous_iterate (boolean, optional): If True, use values from previous
                iteration to compute the stress. Defaults to False.

        """
        # First the mechanical part of the stress
        super().reconstruct_stress(previous_iterate)

        g = self._nd_grid()
        d = self.gb.node_props(g)
        biot = pp.Biot()

        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]

        if previous_iterate:
            p = d[pp.STATE][pp.ITERATE][self.scalar_variable]
        else:
            p = d[pp.STATE][self.scalar_variable]

        # Stress contribution from the scalar variable
        d[pp.STATE]["stress"] += matrix_dictionary[biot.grad_p_matrix_key] * p

        # Is it correct there is no contribution from the global boundary conditions?
