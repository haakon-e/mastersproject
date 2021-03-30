import logging
from typing import Dict, List

import numpy as np

import porepy as pp
from GTS.isc_modelling.general_model import CommonAbstractModel
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.parameter import BaseParameters, FlowParameters
from porepy.params.data import add_nonpresent_dictionary
from porepy.utils.derived_discretizations import implicit_euler
from util.logging_util import timer, trace

logger = logging.getLogger(__name__)


class Flow(CommonAbstractModel):
    """ General flow model for time-dependent Darcy Flow for fractured porous media"""

    def __init__(self, params: FlowParameters):
        """General flow model for time-dependent Darcy Flow

        Parameters
        ----------
        params : BaseParameters
        """
        super().__init__(params)
        self.params = params

        # Time (must be kept for compatibility with pp.run_time_dependent_model)
        self.time = self.params.time
        self.time_step = self.params.time_step
        self.end_time = self.params.end_time

        # Pressure
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

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

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """
        pass

    # --- Boundary condition and source terms ---

    def bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, ["dir"] * all_bf.size)

    def bc_values_scalar(self, g: pp.Grid) -> np.ndarray:  # noqa
        """
        Note that Dirichlet values should be divided by scalar_scale.
        """
        return np.zeros(g.num_faces)

    @property
    def source_flow_rate(self) -> float:
        """ Scaled source flow rate """
        injection_rate = self.params.injection_protocol.active_rate(
            self.time
        )  # injection rate [l / s], unscaled
        return (
            injection_rate * pp.MILLI * (pp.METER / self.params.length_scale) ** self.Nd
        )

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        """ Well-bore source (scaled)"""
        flow_rate = self.source_flow_rate  # scaled
        values = flow_rate * g.tags["well_cells"] * self.time_step
        return values

    # --- aperture, thickness, and specific volume ---

    def thickness(
        self,
        g: pp.Grid,
        scaled: bool,
        **kwargs,
    ) -> np.ndarray:
        """Compute the fault thickness `b` for the grid `g`.

        Parameters
        ----------
        g : pp.Grid
            grid
        scaled : bool
            whether to scale the aperture, which has units [m]
        **kwargs:
            Optional keyword arguments

        In the matrix, b=1
        In fractures/faults, b = b_0 + da
            where `b_0` is the initial thickness,
            and `da` is the change in aperture
        In fracture intersections, `b` is the mean of the
        neighboring faults.
        See also `specific_volume`.
        """
        b_init = self.compute_initial_thickness(g, scaled=scaled)

        b_total = b_init
        return b_total

    def aperture(
        self,
        g: pp.Grid,
        scaled: bool,
        **kwargs,
    ) -> np.ndarray:
        """Compute the total aperture of each cell on a grid

        Parameters
        ----------
        g : pp.Grid
            grid
        scaled : bool
            whether to scale the aperture, which has units [m]
        **kwargs:
            Optional keyword arguments

        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        a_init = self.compute_initial_aperture(g, scaled=scaled)

        a_total = a_init
        return a_total

    def compute_initial_thickness(self, g: pp.Grid, scaled: bool) -> np.ndarray:
        """Fetch the initial thickness `b_0`

        For 3d-matrix: unitary (thickness is not defined in 3d)
        For 2d-fracture: fetch from the parameter dictionary
        For 1d-intersection: Get the mean from the two adjacent faults
        """
        nd = self.Nd
        gb = self.gb
        b = np.ones(g.num_cells)

        # Get the aperture in the corresponding fracture (is 1 for 3D matrix)
        if g.dim == nd:
            return b
        elif g.dim == nd - 1:
            fault = gb.node_props(g, "name")
            b *= self.params.initial_fault_thickness(g, fault)
        elif g.dim == nd - 2:
            primary_grids = gb.node_neighbors(g, only_higher=True)
            primary_aps = [
                self.compute_initial_aperture(g, scaled=scaled) for g in primary_grids
            ]
            mean_b = self.mean_from_neighbors(g, primary_aps)
            b *= mean_b
        else:
            raise ValueError("Not implemented 0d intersection points")

        if scaled:
            b *= pp.METER / self.params.length_scale

        return b

    def compute_initial_aperture(self, g: pp.Grid, scaled: bool) -> np.ndarray:
        """Fetch the initial aperture `a_0`

        For 3d-matrix: unitary (aperture isn't really defined in 3d)
        For 2d-fracture: fetch from the parameter dictionary
        For 1d-intersection: Get the max from the two adjacent fractures
        """
        aperture = np.ones(g.num_cells)
        nd = self.Nd
        gb = self.gb

        # Get the aperture in the corresponding fracture (is 1 for 3D matrix)
        if g.dim == nd:
            return aperture
        elif g.dim == nd - 1:
            fracture = gb.node_props(g, "name")
            aperture *= self.params.initial_aperture(g, fracture)
        elif g.dim == nd - 2:
            primary_grids = gb.node_neighbors(g, only_higher=True)
            primary_aps = [
                self.compute_initial_aperture(g, scaled=scaled) for g in primary_grids
            ]
            mean_aps = self.mean_from_neighbors(g, primary_aps)
            aperture *= mean_aps
        else:
            raise ValueError("Not implemented 0d intersection points")

        if scaled:
            aperture *= pp.METER / self.params.length_scale

        return aperture

    def mean_from_neighbors(self, g: pp.Grid, quantity: List[np.ndarray]):
        """Compute fracture intersection quantity by taking mean from higher-dim neighbors

        The quantity, typically aperture or thickness, is computed by averaging the values
        in adjacent higher-dimensional cells (i.e. from the two faults producing the
        intersection).
        """
        gb = self.gb

        primary_grids = gb.node_neighbors(g, only_higher=True)
        primary_edges = [(g, g_h) for g_h in primary_grids]
        primary_cell_faces = [g_h.cell_faces for g_h in primary_grids]
        mortar_grids = [
            gb.edge_props(edge)["mortar_grid"] for edge in primary_edges
        ]
        projected_quants = [
            mg.mortar_to_secondary_int()
            * mg.primary_to_mortar_int()
            * np.abs(cell_face)
            * q
            for mg, q, cell_face in zip(
                mortar_grids, quantity, primary_cell_faces
            )
        ]
        quantities = np.vstack(projected_quants)
        return np.mean(quantities, axis=0)

    def specific_volume(self, g: pp.Grid, scaled: bool, **kwargs) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the thickness in codimension 1 and the square/cube
        of thickness in codimensions 2 and 3.
        """
        a = self.thickness(g, scaled, **kwargs)
        return np.power(a, self.Nd - g.dim)

    # --- Set parameters ---

    def set_scalar_parameters(self) -> None:
        """Set scalar parameters for the simulation

        We set
            * boundary conditions and values
            * mass_weight (aka storage / compressibility term)
            * source (injection/production in the domains)
            * time_step (needed for some discretization methods)
            * permeability (see self.set_permeability_from_aperture)
            * gravity source term (see self.vector_source)
        """
        gb = self.gb

        # Set to 0 for steady state
        compressibility: float = self.params.fluid.COMPRESSIBILITY * (
            self.params.scalar_scale / pp.PASCAL
        )  # scaled. [1/Pa]
        for g, d in gb:
            porosity: np.ndarray = self.porosity(g)  # Unit [-]
            # specific volume
            specific_volume: np.ndarray = self.specific_volume(g, scaled=True)

            # Boundary and source conditions
            bc: pp.BoundaryCondition = self.bc_type_scalar(g)
            bc_values: np.ndarray = self.bc_values_scalar(g)  # Already scaled
            source_values: np.ndarray = self.source_scalar(g)  # Already scaled

            # Mass weight
            mass_weight = compressibility * porosity * specific_volume

            # Initialize data
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "source": source_values,
                    "time_step": self.time_step,
                },
            )

        # Set permeability on grid, fracture and mortar grids.
        self.set_permeability_from_aperture()
        # Set gravitational effects
        self.vector_source()

    def permeability(self, g, scaled, **kwargs) -> np.ndarray:
        """ Set (uniform) permeability in a subdomain"""
        # intact rock gets permeability from rock
        if g.dim == self.Nd:
            k = self.params.rock.PERMEABILITY * np.ones(g.num_cells)
            if scaled:
                k *= (pp.METER / self.params.length_scale) ** 2
        # fractures get permeability from cubic law
        else:
            aperture = self.aperture(g, scaled=scaled, **kwargs)
            k = self.params.cubic_law(aperture)
        return k

    def porosity(self, g) -> np.ndarray:
        porosity = self.params.rock.POROSITY if g.dim == self.Nd else 1.0
        return np.ones(g.num_cells) * porosity

    def set_permeability_from_aperture(self, **kwargs) -> None:
        """Set permeability by cubic law in fractures.

        **kwargs is passed to
            `permeability`, `specific_volume`, `thickness`
        """
        gb = self.gb
        scalar_key = self.scalar_parameter_key

        # Scaled dynamic viscosity
        viscosity = self.params.fluid.dynamic_viscosity * (
            pp.PASCAL / self.params.scalar_scale
        )
        for g, d in gb:
            k: np.ndarray = self.permeability(
                g, scaled=True, **kwargs,
            )  # permeability [m2] (scaled)

            # Multiply by the volume of the flattened dimension (specific volume)
            k *= self.specific_volume(g, scaled=True, **kwargs)

            kxx = k / viscosity
            diffusivity = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][scalar_key]["second_order_tensor"] = diffusivity

        # Normal permeability inherited from the neighboring fracture g_l
        for e, data_edge in gb.edges():
            mg = data_edge["mortar_grid"]
            g_l, g_h = gb.nodes_of_edge(e)  # get the neighbors

            # get thickness from lower dim neighbour
            b_l = self.thickness(g_l, scaled=True, **kwargs)  # one value per grid cell

            # Take trace of and then project specific volumes from g_h to mg
            V_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self.specific_volume(g_h, scaled=True)
            )

            # Compute diffusivity on g_l
            diffusivity = self.permeability(g_l, scaled=True, **kwargs) / viscosity

            # Division through half the thickness represents taking the (normal) gradient
            # Then, project to mg.
            normal_diffusivity = mg.secondary_to_mortar_int() * np.divide(
                diffusivity, b_l / 2
            )

            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= V_h

            # Set the data
            pp.initialize_data(
                mg,
                data_edge,
                scalar_key,
                {"normal_diffusivity": normal_diffusivity},
            )

    # --- Gravity-related methods ---

    def vector_source(self):
        """ Set gravity as a vector source term in the fluid flow equations"""
        if not self.params.gravity:
            return

        gb = self.gb
        scalar_key = self.scalar_parameter_key
        ls, ss = self.params.length_scale, self.params.scalar_scale
        for g, d in gb:
            # minus sign to convert from positive z downward (depth) to positive upward.
            gravity = -pp.GRAVITY_ACCELERATION * self.density(g) * (ls / ss)
            vector_source = np.zeros((self.Nd, g.num_cells))
            vector_source[-1, :] = gravity
            vector_params = {
                "vector_source": vector_source.ravel("F"),
                "ambient_dimension": self.Nd,
            }
            pp.initialize_data(g, d, scalar_key, vector_params)

        for e, de in gb.edges():
            mg: pp.MortarGrid = de["mortar_grid"]
            g_l, _ = gb.nodes_of_edge(e)
            b_l = self.thickness(g_l, scaled=True)

            # Compute gravity on the secondary grid
            rho_g = -pp.GRAVITY_ACCELERATION * self.density(g_l) * (ls / ss)

            # Multiply by (b/2) to "cancel out" the normal gradient of the diffusivity
            # (see also self.set_permeability_from_aperture)
            gravity_l = rho_g * (b_l / 2)

            # Take the gravity from the secondary grid and project to the interface
            gravity_mg = mg.secondary_to_mortar_avg() * gravity_l

            vector_source = np.zeros((self.Nd, mg.num_cells))
            vector_source[-1, :] = gravity_mg
            gravity = vector_source.ravel("F")

            pp.initialize_data(mg, de, scalar_key, {"vector_source": gravity})

    def density(self, g: pp.Grid) -> np.ndarray:
        """Compute unscaled fluid density

        Either use constant density, or compute as an exponential function of pressure.
        See also FlowParameters.

        rho = rho0 * exp( c * (p - p0) )

        where rho0 and p0 are reference values.

        For reference values, see:
            Berre et al. (2018): Three-dimensional numerical modelling of fracture
                       reactivation due to fluid injection in geothermal reservoirs
        """
        c = self.params.fluid.COMPRESSIBILITY
        ss = self.params.scalar_scale
        d = self.gb.node_props(g)

        # For reference values, see: Berre et al. (2018):
        # Three-dimensional numerical modelling of fracture
        # reactivation due to fluid injection in geothermal reservoirs
        rho0 = 1014 * np.ones(g.num_cells) * (pp.KILOGRAM / pp.METER ** 3)
        p0 = 1 * pp.ATMOSPHERIC_PRESSURE

        if self.params.constant_density:
            # This sets density to rho0.
            p = p0
        else:
            # Use pressure in STATE to approximate density
            p = d[pp.STATE][self.scalar_variable] * ss if pp.STATE in d else p0

        rho = rho0 * np.exp(c * (p - p0))

        return rho

    def hydrostatic_pressure(self, g: pp.Grid, scaled: bool) -> np.ndarray:
        """Hydrostatic pressure in the grid g

        If gravity is active, the hydrostatic pressure depends on depth.
        Otherwise, we set to atmospheric pressure.
        """
        params = self.params
        if params.gravity:
            depth = self.depth(g.cell_centers)
        else:
            depth = self.depth(np.zeros((self.Nd, g.num_cells)))
        hydrostatic = params.fluid.hydrostatic_pressure(depth)
        if scaled:
            hydrostatic /= params.scalar_scale

        return hydrostatic

    def depth(self, coords):
        """Unscaled depth. We center the domain at 480m below the surface.
        (See Krietsch et al, 2018a)
        """
        assert np.atleast_2d(coords).shape[0] == self.Nd
        return self.params.depth - self.params.length_scale * coords[2]

    # --- Primary variables and discretizations ---

    def assign_scalar_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        primary_vars = pp.PRIMARY_VARIABLES

        # First for the nodes
        for _, d in gb:
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update(
                {
                    self.scalar_variable: {"cells": 1},
                }  # noqa: E231
            )

        # Then for the edges
        for _, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update(
                {
                    self.mortar_scalar_variable: {"cells": 1},
                }
            )

    def assign_scalar_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        gb = self.gb
        # Shorthand
        key_s = self.scalar_parameter_key
        var_s = self.scalar_variable
        discr_key, coupling_discr_key = pp.DISCRETIZATION, pp.COUPLING_DISCRETIZATION

        # Scalar discretizations (all dimensions)
        diff_disc_s = implicit_euler.ImplicitMpfa(key_s)
        mass_disc_s = implicit_euler.ImplicitMassMatrix(key_s, var_s)
        source_disc_s = pp.ScalarSource(key_s)

        # Assign node discretizations
        for _, d in gb:
            add_nonpresent_dictionary(d, discr_key)

            d[discr_key].update(
                {
                    var_s: {
                        "diffusion": diff_disc_s,
                        "mass": mass_disc_s,
                        "source": source_disc_s,
                    },
                }
            )

        # Assign edge discretizations
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            add_nonpresent_dictionary(d, coupling_discr_key)

            d[coupling_discr_key].update(
                {
                    self.scalar_coupling_term: {
                        g_h: (var_s, "diffusion"),
                        g_l: (var_s, "diffusion"),
                        e: (
                            self.mortar_scalar_variable,
                            pp.RobinCoupling(key_s, diff_disc_s),
                        ),
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

    def initial_scalar_condition(self) -> None:
        """Initial scalar conditions for pressure and mortar flux

        The scalar pressure depends on gravity.
        """
        gb = self.gb

        for g, d in gb:
            add_nonpresent_dictionary(d, pp.STATE)
            # Initial value for the scalar variable.
            initial_scalar_value = self.hydrostatic_pressure(g, scaled=True)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})

        for _, d in gb.edges():
            add_nonpresent_dictionary(d, pp.STATE)
            mg = d["mortar_grid"]
            initial_scalar_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_scalar_value

    # --- Simulation and solvers ---

    @timer(logger)
    def prepare_simulation(self):
        """Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self._prepare_grid()

        self.set_scalar_parameters()
        self.assign_scalar_variables()
        self.assign_scalar_discretizations()
        self.initial_scalar_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

    def _prepare_grid(self):
        """ Wrapper to create grid"""
        self.create_grid()

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict,
    ):
        # Convergence check for linear problems
        if not self._is_nonlinear_problem():
            return super().check_convergence(
                solution, prev_solution, init_solution, nl_params
            )

        # -- Calculate the scalar error for non-linear simulations --
        # This code will only be executed if called in a coupled problem.

        var_s = self.scalar_variable
        ss = self.params.scalar_scale

        # Extract convergence tolerance
        tol_convergence = nl_params.get("convergence_tol")
        converged = False
        diverged = False

        # Find indices for pressure variables
        scalar_dof = np.array([], dtype=np.int)
        for g, _ in self.gb:
            scalar_dof = np.hstack((scalar_dof, self.dof_manager.dof_ind(g, var_s)))

        # Unscaled pressure solutions
        scalar_now = solution[scalar_dof] * ss
        scalar_prev = prev_solution[scalar_dof] * ss
        scalar_init = init_solution[scalar_dof] * ss

        # Calculate norms
        # scalar_norm = np.sum(scalar_now ** 2)
        difference_in_iterates_scalar = (
            np.sqrt(np.sum((scalar_now - scalar_prev) ** 2)) / scalar_now.size
        )
        difference_from_init_scalar = (
            np.sqrt(np.sum((scalar_now - scalar_init) ** 2)) / scalar_now.size
        )

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        scaled_convergence_tol = tol_convergence * ss
        absolute_convergence = difference_in_iterates_scalar < scaled_convergence_tol
        relative_convergence = (
            difference_in_iterates_scalar
            < tol_convergence * difference_from_init_scalar
        )
        abs_error = difference_in_iterates_scalar
        rel_error = difference_in_iterates_scalar / difference_from_init_scalar
        if absolute_convergence:
            converged = True
            error_scalar = abs_error
        elif relative_convergence:
            converged = True
            error_scalar = rel_error
        else:  # If not convergence, report relative error
            error_scalar = rel_error

        logger.info(
            f"Pressure error: "
            f"absolute={abs_error:.2e} {'<' if absolute_convergence else '>'} {scaled_convergence_tol:.2e}. "
            f"relative={rel_error:.2e} {'<' if relative_convergence else '>'} {tol_convergence:.2e} "
            f"({'Converged' if (absolute_convergence or relative_convergence) else 'Did not converge'})."
        )

        if difference_in_iterates_scalar > 1e30:
            diverged = True

        return error_scalar, converged, diverged

    @timer(logger, level="INFO")
    def initialize_linear_solver(self):
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

    # --- Newton iterations ---

    def before_newton_loop(self):
        """Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_scalar_parameters()

    def after_simulation(self):
        """Called after a time-dependent problem"""
        self.export_pvd()
        logger.info(f"Solution exported to folder \n {self.params.folder_name}")

    # --- Exporting and visualization ---

    def set_viz(self):
        """ Set exporter for visualization """
        super().set_viz()
        # list of time steps to export with visualization.
        self.export_times = []  # noqa

        self.p_exp = "p_exp"  # noqa
        self.p_perturbation = "p_perturb_from_t0"  # noqa
        self.aperture_exp = "aperture"  # noqa
        self.transmissivity_exp = "transmissivity"  # noqa

        # fmt: off
        self.export_fields.extend([
            self.p_exp,
            self.p_perturbation,
            self.aperture_exp,
            self.transmissivity_exp,
        ])
        # fmt: on

    def export_pressure_perturbation(self, d: dict):
        """ Export pressure perturbation relative to pressure at t=0"""
        p = d[pp.STATE][self.scalar_variable]
        if np.isclose(self.time, 0):
            d[pp.STATE]["p_ref"] = np.copy(p)
        p_ref = d[pp.STATE].get("p_ref", np.zeros_like(p))
        return (p - p_ref) * self.params.scalar_scale

    def export_step(self, write_vtk=True):
        """ Export a time step step with pressures, apertures and transmissivities"""
        super().export_step(write_vtk=False)
        for g, d in self.gb:
            state = d[pp.STATE]
            # Export aperture
            aperture = self.aperture(g, scaled=False)
            state[self.aperture_exp] = aperture
            # Export transmissivity
            thickness = self.thickness(g, scaled=False)
            transmissivity = self.params.T_from_a_b(aperture, thickness)
            state[self.transmissivity_exp] = transmissivity
            # Export pressure variable
            if self.scalar_variable in state:
                state[self.p_exp] = (
                    state[self.scalar_variable].copy() * self.params.scalar_scale
                )
            else:
                state[self.p_exp] = np.zeros((self.Nd, g.num_cells))

            # Export pressure perturbation
            state[self.p_perturbation] = self.export_pressure_perturbation(d)

        if write_vtk:
            self.viz.write_vtu(
                data=self.export_fields, time_step=self.time
            )  # Write visualization
            self.export_times.append(self.time)

    def export_pvd(self):
        """ Implementation of export pvd"""
        self.viz.write_pvd(np.array(self.export_times))


class FlowISC(Flow):
    """ Flow model for fractured porous media. Specific to GTS-ISC project."""

    def __init__(self, params: FlowParameters):
        """Initialize the flow model

        Parameters
        ----------
        params : FlowParameters

        """
        super().__init__(params)
        self.params = params

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            bounding_box (dict): The SCALED bounding box of the domain,
                defined through minimum and maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.
            network (pp.FractureNetwork3d): The fracture network associated to
                the domain.

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """

        # Create grid
        gb, network = create_grid(
            **self.params.dict(
                include={
                    "mesh_args",
                    "length_scale",
                    "bounding_box",
                    "fractures",
                    "folder_name",
                }
            )
        )
        self.gb = gb
        self.bounding_box = gb.bounding_box(as_dict=True)
        self.network = network

    @property
    def gb(self) -> pp.GridBucket:
        return self._gb

    @gb.setter
    def gb(self, gb: pp.GridBucket):
        """Set a grid bucket to the class"""
        self._gb = gb
        if gb is None:
            return
        pp.contact_conditions.set_projections(self.gb)
        self.gb.add_node_props(keys=["name"])  # Add 'name' as node prop to all grids.

        # Set the bounding box
        self.bounding_box = gb.bounding_box(as_dict=True)

        # Set Nd grid name
        self.gb.set_node_prop(self._nd_grid(), key="name", val=self.params.intact_name)

        # Set fracture grid names
        if self.params.n_frac > 0:
            fracture_grids = self.gb.get_grids(lambda g: g.dim == self.Nd - 1)
            assert (
                len(fracture_grids) == self.params.n_frac
            ), "There should be equal number of Nd-1 fractures as shearzone names"
            # We assume that order of fractures on grid creation (self.create_grid)
            # is preserved.
            for i, sz_name in enumerate(self.params.fractures):
                self.gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)

    def grids_by_name(self, name, key="name") -> np.ndarray:
        """Get grid by grid bucket node property 'name'"""
        gb = self.gb
        grids = gb.get_grids(lambda g: gb.node_props(g, key) == name)

        return grids

    @trace(logger, timeit=False, level="INFO")
    def well_cells(self) -> None:
        """
        Tag well cells with unity values, positive for injection cells and
        negative for production cells.
        """
        # Initiate all tags to zero
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

        # Set injection cells
        if self.params.well_cells:
            self.params.well_cells(self.params, self.gb)

    # --- Simulation and solvers ---

    def _prepare_grid(self):
        """Tag well cells right after creation.
        Called by self.prepare_simulation()
        """
        if self.gb is None:
            super()._prepare_grid()
        self.well_cells()  # tag well cells

    # -- For testing --

    def after_simulation(self):
        super().after_simulation()

        # # --- Print (and save to file) a summary of the simulation ---
        #
        # # Intro:
        # summary_intro = f"Time of simulation: {time.asctime()}\n"
        #
        # # Get negative values
        # g: pp.Grid = self._nd_grid()
        # d = self.gb.node_props(g)
        # p: np.ndarray = d[pp.STATE][self.p_exp]
        # neg_ind = np.where(p < 0)[0]
        # negneg_ind = np.where(p < -1e-10)[0]
        #
        # p_neg = p[neg_ind]
        #
        # summary_p_common = (
        #     f"\nInformation on negative values:\n"
        #     f"pressure values. "
        #     f"max: {np.max(p):.2e}. "
        #     f"Mean: {np.mean(p):.2e}. "
        #     f"Min: {np.min(p):.2e}\n"
        # )
        # if neg_ind.size > 0:
        #     summary_p = (
        #         f"{summary_p_common}"
        #         f"all negative indices: p<0: count:{neg_ind.size}, indices: {neg_ind}\n"
        #         f"very negative indices: p<-1e-10: count: {negneg_ind.size}, "
        #         f"indices: {negneg_ind}\n"
        #         f"neg pressure range: [{p_neg.min():.2e}, {p_neg.max():.2e}]\n"
        #     )
        # else:
        #     summary_p = (
        #         f"{summary_p_common}"
        #         f"No negative pressure values. count:{neg_ind.size}\n"
        #     )
        #
        # self.neg_ind = neg_ind  # noqa
        # self.negneg_ind = negneg_ind  # noqa
        #
        # # Condition number
        # A, _ = self.assembler.assemble_matrix_rhs()  # noqa
        # row_sum = np.sum(np.abs(A), axis=1)
        # pp_cond = np.max(row_sum) / np.min(row_sum)
        # diag = np.abs(A.diagonal())
        # umfpack_cond = np.max(diag) / np.min(diag)
        #
        # summary_param = (
        #     f"\nSummary of relevant parameters:\n"
        #     f"length scale: {self.params.length_scale:.2e}\n"
        #     f"scalar scale: {self.params.scalar_scale:.2e}\n"
        #     f"time step: {self.time_step / pp.HOUR:.4f} hours\n"
        #     f"3d cells: {g.num_cells}\n"
        #     f"pp condition number: {pp_cond:.2e}\n"
        #     f"umfpack condition number: {umfpack_cond:.2e}\n"
        # )
        #
        # scalar_parameters = d[pp.PARAMETERS][self.scalar_parameter_key]
        # diffusive_term = scalar_parameters["second_order_tensor"].values[0, 0, 0]
        # mass_term = scalar_parameters["mass_weight"][0]
        # source_term = scalar_parameters["source"]
        # nnz_source = np.where(source_term != 0)[0].size
        # cv = g.cell_volumes
        # summary_terms = (
        #     f"\nEstimates on term sizes, 3d grid:\n"
        #     f"diffusive term: {diffusive_term:.2e}\n"
        #     f"mass term: {mass_term:.2e}\n"
        #     f"source; max: {source_term.max():.2e}; "
        #     f"number of non-zero sources: {nnz_source}\n"
        #     f"cell volumes. "
        #     f"max:{cv.max():.2e}, "
        #     f"min:{cv.min():.2e}, "
        #     f"mean:{cv.mean():.2e}\n"
        # )
        #
        # # Write summary to file
        # summary_path = self.params.folder_name / "summary.txt"
        # summary_text = summary_intro + summary_p + summary_param + summary_terms
        # logger.info(summary_text)
        # with summary_path.open(mode="w") as f:
        #     f.write(summary_text)
