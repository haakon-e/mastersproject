import time
from typing import Dict, List

import porepy as pp
from porepy.models.abstract_model import AbstractModel
from porepy.params.data import add_nonpresent_dictionary
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations
import numpy as np
import scipy.sparse.linalg as spla

from GTS.isc_modelling.ISCGrid import create_grid
from GTS.ISC_data.isc import ISCData

# --- LOGGING UTIL ---
from util.logging_util import timer, trace

import logging
logger = logging.getLogger(__name__)


class Flow(AbstractModel):
    """ General flow model for time-dependent Darcy Flow for fractured porous media"""

    def __init__(self, params: Dict):
        """ General flow model for time-dependent Darcy Flow

        Parameters
        ----------
        params : dict
            folder_name : str
                Path to where visualization results are to be stored.

            -- OPTIONAL --
            scalar_scale : float (default: 1)
                pressure scaling coefficient
            length_scale : float (default: 1)
                length scaling coefficient
            temperature : float (default: 11 C)
                Temperature of the fluid in pp.Water
            file_name : str (default: 'simulation_run')

        """
        self.params = params

        # File name
        self.file_name = params.get('file_name', 'simulation_run')
        self.viz_folder_name = params.get('folder_name')

        # Time
        self.time = 0
        self.time_step = 1
        self.end_time = 1

        # Pressure
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        # Scaling coefficients
        self.scalar_scale = params.get('scalar_scale', 1)
        self.length_scale = params.get('length_scale', 1)

        # Grid
        self.gb = None  # pp.GridBucket
        self.Nd = None  # int
        self.box = None  # Dict[float]

        # Parameters

        # Constant in-situ temperature. Default is ISC temp.
        temperature = params.get('temperature', 11)
        self.fluid = pp.Water(theta_ref=temperature)

        # Initialize simulation/solver options
        self.assembler = None
        self.linear_solver = None

    # --- Grid methods ---

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
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
            aperture *= (pp.METER / self.length_scale)
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

        compressibility = self.fluid.COMPRESSIBILITY * (self.scalar_scale / pp.PASCAL)  # scaled. [1/Pa]
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
                    compressibility
                    * porosity
                    * specific_volume
                    * np.ones(g.num_cells)
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
        viscosity = self.fluid.dynamic_viscosity() * (pp.PASCAL / self.scalar_scale)
        for g, d in gb:
            # permeability [m2] (scaled)
            k = self.permeability(g) * (pp.METER / self.length_scale) ** 2

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
            diffusivity = data_l[pp.PARAMETERS][scalar_key]["second_order_tensor"].values[0, 0]

            # Division through half the aperture represents taking the (normal) gradient
            normal_diffusivity = (
                    mg.slave_to_mortar_int()
                    * np.divide(diffusivity, aperture_l / 2)
            )
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h

            # Set the data
            pp.initialize_data(
                mg,
                data_edge,
                scalar_key,
                {"normal_diffusivity": normal_diffusivity},
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

            d[primary_vars].update({
                self.scalar_variable: {"cells": 1},
            })

        # Then for the edges
        for _, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update({
                self.mortar_scalar_variable: {"cells": 1},
            })

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

            d[discr_key].update({
                var_s: {
                    "diffusion": diff_disc_s,
                    "mass": mass_disc_s,
                    "source": source_disc_s,
                },
            })

        # Assign edge discretizations
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            add_nonpresent_dictionary(d, coupling_discr_key)

            d[coupling_discr_key].update({
                self.scalar_coupling_term: {
                    g_h: (var_s, "diffusion"),
                    g_l: (var_s, "diffusion"),
                    e: (
                        self.mortar_scalar_variable,
                        pp.RobinCoupling(key_s, diff_disc_s),
                    ),
                },
            })

    @timer(logger, level='INFO')
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
        self.create_grid()
        self.Nd = self.gb.dim_max()
        self.set_scalar_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

    def check_convergence(
            self, solution: np.ndarray, prev_solution: np.ndarray,
            init_solution: np.ndarray, nl_params: Dict = None,
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
                (
                    scalar_dof,
                    self.assembler.dof_ind(g, self.scalar_variable)
                )
            )

        # Unscaled pressure solutions
        scalar_now = solution[scalar_dof] * self.scalar_scale
        scalar_prev = prev_solution[scalar_dof] * self.scalar_scale
        scalar_init = init_solution[scalar_dof] * self.scalar_scale

        # Calculate norms
        scalar_norm = np.sum(scalar_now ** 2)
        difference_in_iterates_scalar = np.sum((scalar_now - scalar_prev) ** 2)
        difference_from_init_scalar = np.sum((scalar_now - scalar_init) ** 2)

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        if difference_in_iterates_scalar < tol_convergence:  # and scalar_norm < tol_convergence
            converged = True
            error_scalar = difference_in_iterates_scalar
            logger.info(f"pressure converged absolutely")
        else:
            # Relative convergence criterion:
            if difference_in_iterates_scalar < tol_convergence * difference_from_init_scalar:
                converged = True
                logger.info(f"pressure converged relatively")

            error_scalar = (difference_in_iterates_scalar / difference_from_init_scalar)

        logger.info(f"Error in pressure is {error_scalar:.6e}.")

        return error_scalar, converged, diverged

    @timer(logger, level='INFO')
    def initialize_linear_solver(self):
        """ Initialize linear solver

        Currently, we only consider the direct solver.
        See also self.assemble_and_solve_linear_system()
        """

        solver = self.params.get("linear_solver", "direct")

        if solver == "direct":
            """ In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.

            """
            self.linear_solver = "direct"

        else:
            raise ValueError(f"Unknown linear solver {solver}")

    def assemble_and_solve_linear_system(self, tol):
        """ Assemble a solve the linear system"""

        A, b = self.assembler.assemble_matrix_rhs()

        # Estimate condition number
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and "
            f"min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )

        if self.linear_solver == "direct":
            tic = time.time()
            logger.info("Solve Ax=b using scipy")
            sol = spla.spsolve(A, b)
            logger.info(f"Done. Elapsed time {time.time() - tic}")
            logger.info(f"||b-Ax|| = {np.linalg.norm(b - A * sol)}")
            logger.info(f"||b-Ax|| / ||b|| = {np.linalg.norm(b - A * sol) / np.linalg.norm(b)}")
            return sol

        else:
            raise ValueError(f"Unknown linear solver {self.linear_solver}")

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
        logger.info(f"Solution exported to folder \n {self.viz_folder_name}")

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
        box = self.box
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

    def _nd_grid(self):
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
        self.viz = pp.Exporter(self.gb, file_name=self.file_name, folder_name=self.viz_folder_name)
        # list of time steps to export with visualization.
        self.export_times = []

        self.p_exp = 'p_exp'

        self.export_fields = [
            self.p_exp,
        ]

    def export_step(self, write_vtk=True):
        """ Export a step with pressures """
        for g, d in self.gb:
            # Export pressure variable
            if self.scalar_variable in d[pp.STATE]:
                d[pp.STATE][self.p_exp] = d[pp.STATE][self.scalar_variable].copy() * self.scalar_scale
            else:
                d[pp.STATE][self.p_exp] = np.zeros((self.Nd, g.num_cells))

        if write_vtk:
            self.viz.write_vtk(data=self.export_fields, time_step=self.time)  # Write visualization
            self.export_times.append(self.time)

    def export_pvd(self):
        """ Implementation of export pvd"""
        self.viz.write_pvd(self.export_times)


class FlowISC(Flow):
    """ Flow model for fractured porous media. Specific to GTS-ISC project."""

    def __init__(self, params: Dict):
        """ Initialize the flow model

        Parameters
        ----------
        params : dict
            folder_name : str
                Path to where visualization results are to be stored.
            shearzone_names : List[str]
                List of shearzones to mesh
            source_scalar_borehole_shearzone : Dict[str, str]
                Which borehole-shearzone intersection to inject to.
            mesh_args : Dict[str, Float]
                Coefficients to mesh the domain
            bounding_box : Dict[str, Float]
                Bounding box of domain.

            -- OPTIONAL --
            scalar_scale : float (default: 1)
                pressure scaling coefficient
            length_scale : float (default: 1)
                length scaling coefficient
            temperature : float (default: 11 C)
                Temperature of the fluid in pp.Water
            file_name : str (default: 'simulation_run')

        """
        super().__init__(params)

        # --- FRACTURES ---
        self.shearzone_names: List[str] = params.get('shearzone_names')
        self.n_frac = len(self.shearzone_names) if self.shearzone_names else 0

        # --- PHYSICAL PARAMETERS ---

        # * Source injection *
        self.source_scalar_borehole_shearzone = params.get('source_scalar_borehole_shearzone')

        # * Permeability and aperture *

        # For now, constant permeability in fractures
        # For details on values, see "2020-04-21 Møtereferat" (Veiledningsmøte)
        mean_frac_permeability = 4.9e-16 * pp.METER ** 2
        mean_intact_permeability = 2e-20 * pp.METER ** 2
        self.initial_permeability = {  # Unscaled
            'S1_1': mean_frac_permeability,
            'S1_2': mean_frac_permeability,
            'S1_3': mean_frac_permeability,
            'S3_1': mean_frac_permeability,
            'S3_2': mean_frac_permeability,
            None: mean_intact_permeability,  # 3D matrix
        }

        # Use cubic law to compute initial apertures in fractures.
        # k = a^2 / 12 => a=sqrt(12k)
        self.initial_aperture = {
            sz: np.sqrt(12 * k) for sz, k in self.initial_permeability.items()
        }
        self.initial_aperture[None] = 1  # Set 3D matrix aperture to 1.

        # --- COMPUTATIONAL MESH ---
        self.mesh_args: Dict[str, float] = params.get('mesh_args')
        self.box: Dict[str, float] = params.get('bounding_box')
        self.gb = None
        self.Nd = None
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
            box (dict): The SCALED bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """

        # Create grid
        gb, scaled_box, network = create_grid(
            self.mesh_args, self.length_scale, self.box,
            self.shearzone_names, self.viz_folder_name,
        )
        self.gb = gb
        self.box = scaled_box
        self.network = network

        self.Nd = self.gb.dim_max()

    def grids_by_name(self, name, key='name') -> np.ndarray:
        """ Get grid by grid bucket node property 'name'

        """
        gb = self.gb
        grids = gb.get_grids(lambda g: gb.node_props(g, key) == name)

        return grids

    def well_cells(self) -> None:
        """
        Tag well cells with unity values, positive for injection cells and
        negative for production cells.
        """
        isc = ISCData()
        df = isc.borehole_plane_intersection()

        # Shorthand
        borehole = self.source_scalar_borehole_shearzone.get("borehole")
        shearzone = self.source_scalar_borehole_shearzone.get("shearzone")

        # Get the UNSCALED coordinates of the borehole - shearzone intersection.
        _mask = (df.shearzone == shearzone) & \
                (df.borehole == borehole)
        result = df.loc[_mask, ("x_sz", "y_sz", "z_sz")]
        if result.empty:
            raise ValueError("No intersection found.")

        # Scale the intersection coordinates by length_scale. (scaled)
        pts = result.to_numpy().T / self.length_scale
        assert pts.shape[1] == 1, "There should only be one intersection"

        # Tag all grid cells. Tag 1 for injection cell, 0 otherwise.
        tagged = False
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)

            grid_name = self.gb.node_props(g, "name")
            if grid_name == shearzone:
                logger.info(f"Tag injection cell on {grid_name!r} (dim: {g.dim}).")

                # Find closest cell
                ids, dsts = g.closest_cell(pts, return_distance=True)
                logger.info(f"Closest cell found has (unscaled) distance: {dsts[0] * self.length_scale:4f}")

                # Tag the injection cell
                tags[ids] = 1
                tagged = True

            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

        if not tagged:
            logger.warning("No injection cell was tagged.")

    # --- Aperture related methods ---

    def aperture(self, g: pp.Grid, scaled=False) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        # Get the aperture in the corresponding shearzone (is 1 for 3D matrix)
        shearzone = self.gb.node_props(g, 'name')
        aperture *= self.initial_aperture[shearzone]

        if scaled:
            aperture *= (pp.METER / self.length_scale)

        return aperture

    # --- Parameter related methods ---

    def permeability(self, g):
        """ Set (uniform) permeability in a subdomain"""
        # get the shearzone
        shearzone = self.gb.node_props(g, 'name')
        return self.initial_permeability[shearzone]

    def porosity(self, g):
        # TODO: Set porosity in fractures and matrix. (Usually set by pp.Rock)
        return 1

    @property
    def source_flow_rate(self) -> float:
        """ Scaled source flow rate """
        injection_rate = 1  # injection rate [l / s], unscaled
        return injection_rate * pp.MILLI * (pp.METER / self.length_scale) ** self.Nd

    def source_scalar(self, g: pp.Grid) -> np.ndarray:
        """ Well-bore source (scaled)"""
        self.well_cells()  # tag well cells
        flow_rate = self.source_flow_rate  # scaled
        values = flow_rate * g.tags["well_cells"] * self.time_step
        return values


