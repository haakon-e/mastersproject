import time
from typing import Dict, Optional

import porepy as pp
from porepy.models.abstract_model import AbstractModel
from porepy.params.data import add_nonpresent_dictionary
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations
import numpy as np
import scipy.sparse.linalg as spla

from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.parameter import BaseParameters, FlowParameters

# --- LOGGING UTIL ---
from util.logging_util import timer, trace

import logging

logger = logging.getLogger(__name__)


class Flow(AbstractModel):
    """ General flow model for time-dependent Darcy Flow for fractured porous media"""

    def __init__(self, params: BaseParameters):
        """ General flow model for time-dependent Darcy Flow

        Parameters
        ----------
        params : BaseParameters
        """
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

        # Grid
        self.gb: Optional[pp.GridBucket] = None
        self.Nd: Optional[int] = None
        self.bounding_box: Optional[
            Dict[str, float]
        ] = None  # Keep this as it will change due to scaling

        # Initialize assembler
        self.assembler = None

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            bounding_box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

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

    def bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """
        Note that Dirichlet values should be divided by scalar_scale.
        """
        return np.zeros(g.num_faces)

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    # --- aperture and specific volume ---

    def aperture(self, g: pp.Grid, scaled) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.Nd:
            aperture *= 0.1

        if scaled:
            aperture *= pp.METER / self.params.length_scale
        return aperture

    def specific_volume(self, g: pp.Grid, scaled) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self.aperture(g, scaled)
        return np.power(a, self.Nd - g.dim)

    # --- Set parameters ---

    def set_scalar_parameters(self) -> None:
        """ Set scalar parameters for the simulation
        """
        gb = self.gb

        # Set to 0 for steady state
        compressibility = self.params.fluid.COMPRESSIBILITY * (
            self.params.scalar_scale / pp.PASCAL
        )  # scaled. [1/Pa]
        for g, d in gb:
            porosity = self.porosity(g)  # Default: 1 [-]
            # specific volume
            specific_volume = self.specific_volume(g, scaled=True)

            # Boundary and source conditions
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)  # Already scaled
            source_values = self.source_scalar(g)  # Already scaled

            # Mass weight  # TODO: Simplified version of mass_weight?
            mass_weight = (
                compressibility * porosity * specific_volume * np.ones(g.num_cells)
            )

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

    def permeability(self, g):
        return 1

    def porosity(self, g):
        return 1

    def set_permeability_from_aperture(self) -> None:
        """ Set permeability by cubic law in fractures.

        Currently, we simply set the initial permeability.
        """
        gb = self.gb
        scalar_key = self.scalar_parameter_key

        # Scaled dynamic viscosity
        viscosity = self.params.fluid.dynamic_viscosity() * (
            pp.PASCAL / self.params.scalar_scale
        )
        for g, d in gb:
            # permeability [m2] (scaled)
            k = self.permeability(g) * (pp.METER / self.params.length_scale) ** 2
            logger.info(f"Scaled permeability in dim {g.dim} set to {k:.3e}")

            # Multiply by the volume of the flattened dimension (specific volume)
            k *= self.specific_volume(g, scaled=True)

            kxx = k / viscosity
            diffusivity = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][scalar_key]["second_order_tensor"] = diffusivity

        # Normal permeability inherited from the neighboring fracture g_l
        for e, data_edge in gb.edges():
            mg = data_edge["mortar_grid"]
            g_l, g_h = gb.nodes_of_edge(e)  # get the neighbors

            # get aperture data from lower dim neighbour
            aperture_l = self.aperture(g_l, scaled=True)  # one value per grid cell

            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.master_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self.specific_volume(g_h, scaled=True)
            )

            # Get diffusivity from lower-dimensional neighbour
            data_l = gb.node_props(g_l)
            diffusivity = data_l[pp.PARAMETERS][scalar_key][
                "second_order_tensor"
            ].values[0, 0]

            # Division through half the aperture represents taking the (normal) gradient
            normal_diffusivity = mg.slave_to_mortar_int() * np.divide(
                diffusivity, aperture_l / 2
            )
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h

            # Set the data
            pp.initialize_data(
                mg, data_edge, scalar_key, {"normal_diffusivity": normal_diffusivity},
            )

    # --- Primary variables and discretizations ---

    def assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        primary_vars = pp.PRIMARY_VARIABLES

        # First for the nodes
        for _, d in gb:
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update(
                {self.scalar_variable: {"cells": 1},}
            )

        # Then for the edges
        for _, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update(
                {self.mortar_scalar_variable: {"cells": 1},}
            )

    def assign_discretizations(self) -> None:
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
        diff_disc_s = IE_discretizations.ImplicitMpfa(key_s)
        mass_disc_s = IE_discretizations.ImplicitMassMatrix(key_s, var_s)
        source_disc_s = pp.ScalarSource(key_s)

        # Assign node discretizations
        for g, d in gb:
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
        """ Discretize all terms
        """
        if not self.assembler:
            self.assembler = pp.Assembler(self.gb)

        self.assembler.discretize()

    # --- Initial condition ---

    def initial_condition(self) -> None:
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        gb = self.gb

        for g, d in gb:
            add_nonpresent_dictionary(d, pp.STATE)
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})

        for _, d in gb.edges():
            add_nonpresent_dictionary(d, pp.STATE)
            mg = d["mortar_grid"]
            initial_scalar_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_scalar_value

    # --- Simulation and solvers ---

    @timer(logger)
    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.
        """
        self._prepare_grid()

        self.Nd = self.gb.dim_max()
        self.set_scalar_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
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
        nl_params: Dict = None,
    ):

        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = np.any(np.isnan(solution))
            converged = not diverged
            error = np.nan if diverged else 0
            return error, converged, diverged

        # -- Calculate the scalar error for non-linear simulations --
        # This code will only be executed if called in a coupled problem.

        # Extract convergence tolerance
        tol_convergence = nl_params.get("nl_convergence_tol")

        converged = False
        diverged = False

        # Find indices for pressure variables
        scalar_dof = np.array([], dtype=np.int)
        for g, _ in self.gb:
            scalar_dof = np.hstack(
                (scalar_dof, self.assembler.dof_ind(g, self.scalar_variable))
            )

        # Unscaled pressure solutions
        scalar_now = solution[scalar_dof] * self.params.scalar_scale
        scalar_prev = prev_solution[scalar_dof] * self.params.scalar_scale
        scalar_init = init_solution[scalar_dof] * self.params.scalar_scale

        # Calculate norms
        scalar_norm = np.sum(scalar_now ** 2)
        difference_in_iterates_scalar = np.sum((scalar_now - scalar_prev) ** 2)
        difference_from_init_scalar = np.sum((scalar_now - scalar_init) ** 2)

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        if (
            difference_in_iterates_scalar < tol_convergence
        ):  # and scalar_norm < tol_convergence
            converged = True
            error_scalar = difference_in_iterates_scalar
            logger.info(f"pressure converged absolutely")
        else:
            # Relative convergence criterion:
            if (
                difference_in_iterates_scalar
                < tol_convergence * difference_from_init_scalar
            ):
                converged = True
                logger.info(f"pressure converged relatively")

            error_scalar = difference_in_iterates_scalar / difference_from_init_scalar

        logger.info(f"Error in pressure is {error_scalar:.6e}.")

        return error_scalar, converged, diverged

    @timer(logger, level="INFO")
    def initialize_linear_solver(self):
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

    def assemble_and_solve_linear_system(self, tol):
        """ Assemble a solve the linear system"""

        A, b = self.assembler.assemble_matrix_rhs()

        # Estimate condition number
        logger.info(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.info(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and "
            f"min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )

        # UMFPACK Estimate of condition number
        logger.info(
            f"UMFPACK Condition number estimate: "
            f"{np.min(np.abs(A.diagonal())) / np.max(np.abs(A.diagonal())) :.2e}"
        )

        if self.params.linear_solver == "direct":
            tic = time.time()
            logger.info("Solve Ax=b using scipy")
            sol = spla.spsolve(A, b)
            logger.info(f"Done. Elapsed time {time.time() - tic}")
            logger.info(f"||b-Ax|| = {np.linalg.norm(b - A * sol)}")
            logger.info(
                f"||b-Ax|| / ||b|| = {np.linalg.norm(b - A * sol) / np.linalg.norm(b)}"
            )
            return sol

        else:
            raise ValueError(f"Unknown linear solver {self.params.linear_solver}")

    # --- Newton iterations ---

    def before_newton_loop(self):
        """ Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_scalar_parameters()

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        # (flow is linear)
        pass

    def after_newton_iteration(self, solution: np.ndarray) -> None:
        pass

    def after_newton_convergence(self, solution, errors, iteration_counter):
        """ Overwrite from parent to export solution steps."""
        self.assembler.distribute_variable(solution)
        self.export_step()

    def after_simulation(self):
        """ Called after a time-dependent problem
        """
        self.export_pvd()
        logger.info(f"Solution exported to folder \n {self.params.folder_name}")

    def after_newton_failure(self, solution, errors, iteration_counter):
        """ Instead of raising error on failure, save and return available data.
        """
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    # --- Helper methods ---

    def domain_boundary_sides(self, g):
        """
        Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries.
        """
        tol = 1e-10
        box = self.bounding_box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        north = g.face_centers[1] > box["ymax"] - tol
        south = g.face_centers[1] < box["ymin"] + tol
        if self.Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    def _nd_grid(self) -> pp.Grid:
        """ Get the grid of the highest dimension. Assumes self.gb is set.
        """
        return self.gb.grids_of_dimension(self.Nd)[0]

    def get_state_vector(self):
        """ Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state, as stored in the GridBucket.

        """
        size = self.assembler.num_dof()
        state = np.zeros(size)
        for g, var in self.assembler.block_dof.keys():
            # Index of
            ind = self.assembler.dof_ind(g, var)

            if isinstance(g, tuple):
                values = self.gb.edge_props(g)[pp.STATE][var]
            else:
                values = self.gb.node_props(g)[pp.STATE][var]
            state[ind] = values

        return state

    def _is_nonlinear_problem(self):
        """ flow problems are linear even with fractures """
        return False

    # --- Exporting and visualization ---

    def set_viz(self):
        """ Set exporter for visualization """
        self.viz = pp.Exporter(
            self.gb,
            file_name=self.params.viz_file_name,
            folder_name=self.params.folder_name,
        )
        # list of time steps to export with visualization.
        self.export_times = []

        self.p_exp = "p_exp"

        self.export_fields = [
            self.p_exp,
        ]

    def export_step(self, write_vtk=True):
        """ Export a step with pressures """
        for g, d in self.gb:
            # Export pressure variable
            if self.scalar_variable in d[pp.STATE]:
                d[pp.STATE][self.p_exp] = (
                    d[pp.STATE][self.scalar_variable].copy() * self.params.scalar_scale
                )
            else:
                d[pp.STATE][self.p_exp] = np.zeros((self.Nd, g.num_cells))

        if write_vtk:
            self.viz.write_vtk(
                data=self.export_fields, time_step=self.time
            )  # Write visualization
            self.export_times.append(self.time)

    def export_pvd(self):
        """ Implementation of export pvd"""
        self.viz.write_pvd(self.export_times)


class FlowISC(Flow):
    """ Flow model for fractured porous media. Specific to GTS-ISC project."""

    def __init__(self, params: FlowParameters):
        """ Initialize the flow model

        Parameters
        ----------
        params : FlowParameters

        """
        super().__init__(params)
        self.params = params

        # --- FRACTURES ---
        # self.shearzone_names: List[str] = params.get("shearzone_names")
        # self.n_frac = len(self.shearzone_names) if self.shearzone_names else 0

        # --- PHYSICAL PARAMETERS ---

        # * Source injection *
        # self.source_scalar_borehole_shearzone = params.get(
        #     "source_scalar_borehole_shearzone"
        # )

        # * Permeability and aperture *

        # For now, constant permeability in fractures
        initial_frac_permeability = (
            {sz: params.frac_permeability for sz in params.shearzone_names}
            if params.shearzone_names
            else {}
        )
        initial_intact_permeability = {params.intact_name: params.intact_permeability}
        self.initial_permeability = {
            **initial_frac_permeability,
            **initial_intact_permeability,
        }

        # Use cubic law to compute initial apertures in fractures.
        # k = a^2 / 12 => a=sqrt(12k)
        self.initial_aperture = {
            sz: np.sqrt(12 * k) for sz, k in self.initial_permeability.items()
        }
        self.initial_aperture[params.intact_name] = 1  # Set 3D matrix aperture to 1.

        # --- COMPUTATIONAL MESH ---
        self._gb: Optional[pp.GridBucket] = None
        self.Nd: Optional[int] = None
        self.network = None

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            bounding_box (dict): The SCALED bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.
            network (pp.FractureNetwork3d): The fracture network associated to the domain.

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
                    "shearzone_names",
                    "folder_name",
                }
            )
        )
        self.gb = gb
        self.bounding_box = gb.bounding_box(as_dict=True)
        self.Nd = self.gb.dim_max()
        self.network = network

    @property
    def gb(self) -> pp.GridBucket:
        return self._gb

    @gb.setter
    def gb(self, gb: pp.GridBucket):
        """ Set a grid bucket to the class
        """
        self._gb = gb
        if gb is None:
            return
        pp.contact_conditions.set_projections(self.gb)
        self.Nd = gb.dim_max()
        self.gb.add_node_props(keys=["name"])  # Add 'name' as node prop to all grids.

        # Set the bounding box
        self.bounding_box = gb.bounding_box(as_dict=True)

        # Set Nd grid name
        self.gb.set_node_prop(self._nd_grid(), key="name", val=self.params.intact_name)

        # Set fracture grid names
        if self.params.n_frac > 0:
            fracture_grids = self.gb.get_grids(lambda g: g.dim == self.Nd - 1)
            # We assume that order of fractures on grid creation (self.create_grid) is preserved.
            for i, sz_name in enumerate(self.params.shearzone_names):
                self.gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)

    def grids_by_name(self, name, key="name") -> np.ndarray:
        """ Get grid by grid bucket node property 'name'

        """
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
        self.params.well_cells(self.params, self.gb)

    # --- Aperture related methods ---

    def aperture(self, g: pp.Grid, scaled) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        # TODO: This does not resolve what happens in 1D fractures
        aperture = np.ones(g.num_cells)

        if g.dim == self.Nd or g.dim == self.Nd - 1:
            # Get the aperture in the corresponding shearzone (is 1 for 3D matrix)
            shearzone = self.gb.node_props(g, "name")
            aperture *= self.initial_aperture[shearzone]
        else:
            # Temporary solution: Just take one of the higher-dim grids' aperture
            g_h = self.gb.node_neighbors(g, only_higher=True)[0]
            shearzone = self.gb.node_props(g_h, "name")
            aperture *= self.initial_aperture[shearzone]

        if scaled:
            aperture *= pp.METER / self.params.length_scale

        return aperture

    # --- Parameter related methods ---

    def permeability(self, g):
        """ Set (uniform) permeability in a subdomain"""

        # TODO: This does not resolve what happens in 1D fractures
        if g.dim == self.Nd or g.dim == self.Nd - 1:
            # get the shearzone
            shearzone = self.gb.node_props(g, "name")
            k = self.initial_permeability[shearzone]
        else:
            # Set the permeability in fracture intersections as
            # the average of the neighbouring fractures

            # Temporary solution: Just take one of the higher-dim grids' permeability
            g_h = self.gb.node_neighbors(g, only_higher=True)[0]
            shearzone = self.gb.node_props(g_h, "name")
            k = self.initial_permeability[shearzone]

        return k

    def porosity(self, g):
        # TODO: Set porosity in fractures and matrix. (Usually set by pp.Rock)
        return 1

    @property
    def source_flow_rate(self) -> float:
        """ Scaled source flow rate """
        injection_rate = (
            self.params.injection_rate
        )  # 10 / 60  # 10 l/min  # injection rate [l / s], unscaled
        return (
            injection_rate * pp.MILLI * (pp.METER / self.params.length_scale) ** self.Nd
        )

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        """ Well-bore source (scaled)"""
        flow_rate = self.source_flow_rate  # scaled
        values = flow_rate * g.tags["well_cells"] * self.time_step
        return values

    # --- Simulation and solvers ---

    def _prepare_grid(self):
        """ Tag well cells right after creation.
        Called by self.prepare_simulation()
        """
        if self.gb is None:
            super()._prepare_grid()
        self.well_cells()  # tag well cells

    # -- For testing --

    def after_simulation(self):
        super().after_simulation()

        # Intro:
        summary_intro = f"Time of simulation: {time.asctime()}\n"

        # Get negative values
        g: pp.Grid = self._nd_grid()
        d = self.gb.node_props(g)
        p: np.ndarray = d[pp.STATE][self.p_exp]
        neg_ind = np.where(p < 0)[0]
        negneg_ind = np.where(p < -1e-10)[0]

        p_neg = p[neg_ind]

        summary_p_common = (
            f"\nInformation on negative values:\n"
            f"pressure values. "
            f"max: {p.max():.2e}. "
            f"Mean: {p.mean():.2e}. "
            f"Min: {p.min():.2e}\n"
        )
        if neg_ind.size > 0:
            summary_p = (
                f"{summary_p_common}"
                f"all negative indices: p<0: count:{neg_ind.size}, indices: {neg_ind}\n"
                f"very negative indices: p<-1e-10: count: {negneg_ind.size}, indices: {negneg_ind}\n"
                f"neg pressure range: [{p_neg.min():.2e}, {p_neg.max():.2e}]\n"
            )
        else:
            summary_p = (
                f"{summary_p_common}"
                f"No negative pressure values. count:{neg_ind.size}\n"
            )

        self.neg_ind = neg_ind
        self.negneg_ind = negneg_ind

        # Condition number
        A, _ = self.assembler.assemble_matrix_rhs()
        row_sum = np.sum(np.abs(A), axis=1)
        pp_cond = np.max(row_sum) / np.min(row_sum)
        diag = np.abs(A.diagonal())
        umfpack_cond = np.max(diag) / np.min(diag)

        summary_param = (
            f"\nSummary of relevant parameters:\n"
            f"length scale: {self.params.length_scale:.2e}\n"
            f"scalar scale: {self.params.scalar_scale:.2e}\n"
            f"3d permeability: {self.initial_permeability[self.params.intact_name]:.2e}\n"
            f"time step: {self.time_step / pp.HOUR:.4f} hours\n"
            f"3d cells: {g.num_cells}\n"
            f"pp condition number: {pp_cond:.2e}\n"
            f"umfpack condition number: {umfpack_cond:.2e}\n"
        )

        scalar_parameters = d[pp.PARAMETERS][self.scalar_parameter_key]
        diffusive_term = scalar_parameters["second_order_tensor"].values[0, 0, 0]
        mass_term = scalar_parameters["mass_weight"][0]
        source_term = scalar_parameters["source"]
        nnz_source = np.where(source_term != 0)[0].size
        cv = g.cell_volumes
        summary_terms = (
            f"\nEstimates on term sizes, 3d grid:\n"
            f"diffusive term: {diffusive_term:.2e}\n"
            f"mass term: {mass_term:.2e}\n"
            f"source; max: {source_term.max():.2e}; "
            f"number of non-zero sources: {nnz_source}\n"
            f"cell volumes. "
            f"max:{cv.max():.2e}, "
            f"min:{cv.min():.2e}, "
            f"mean:{cv.mean():.2e}\n"
        )

        # Write summary to file
        summary_path = self.params.folder_name / "summary.txt"
        summary_text = summary_intro + summary_p + summary_param + summary_terms
        logger.info(summary_text)
        with summary_path.open(mode="w") as f:
            f.write(summary_text)
