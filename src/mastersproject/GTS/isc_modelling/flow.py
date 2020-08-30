import logging
import time
from typing import Dict, Optional

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

    def __init__(self, params: BaseParameters):
        """ General flow model for time-dependent Darcy Flow

        Parameters
        ----------
        params : BaseParameters
        """
        super().__init__(params)

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

    def specific_volume(self, g: pp.Grid, scaled: bool) -> np.ndarray:
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

    def permeability(self, g, scaled) -> np.ndarray:
        k = np.ones(g.num_cells)
        if scaled:
            k *= (pp.METER / self.params.length_scale) ** 2
        return k

    def porosity(self, g) -> np.ndarray:
        return np.ones(g.num_cells)

    def set_permeability_from_aperture(self) -> None:
        """ Set permeability by cubic law in fractures.

        Currently, we simply set the initial permeability.
        """
        gb = self.gb
        scalar_key = self.scalar_parameter_key

        # Scaled dynamic viscosity
        viscosity = self.params.fluid.dynamic_viscosity * (
            pp.PASCAL / self.params.scalar_scale
        )
        for g, d in gb:
            # permeability [m2] (scaled)
            k: np.ndarray = self.permeability(g, scaled=True)
            # a = self.aperture(g, scaled=True)
            # logger.info(
            #     f"Scaled permeability and aperture in dim {g.dim} have values: "
            #     f"min [k={np.min(k):.2e}, a={np.min(a):.2e}]; "
            #     f"mean [k={np.mean(k):.2e}, a={np.mean(a):.2e}]; "
            #     f"max [k={np.max(k):.2e}, a={np.max(a):.2e}]"
            # )

            # Multiply by the volume of the flattened dimension (specific volume)
            k *= self.specific_volume(g, scaled=True)

            kxx = k / viscosity
            diffusivity = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][scalar_key]["second_order_tensor"] = diffusivity

        # Normal permeability inherited from the neighboring fracture g_l
        for e, data_edge in gb.edges():
            mg = data_edge["mortar_grid"]
            g_l, g_h = gb.nodes_of_edge(e)  # get the neighbors

            # get aperture from lower dim neighbour
            a_l = self.aperture(g_l, scaled=True)  # one value per grid cell

            # Take trace of and then project specific volumes from g_h to mg
            V_h = (
                mg.master_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self.specific_volume(g_h, scaled=True)
            )

            # Compute diffusivity on g_l
            diffusivity = self.permeability(g_l, scaled=True) / viscosity

            # Division through half the aperture represents taking the (normal) gradient
            # Then, project to mg.
            normal_diffusivity = mg.slave_to_mortar_int() * np.divide(
                diffusivity, a_l / 2
            )

            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= V_h

            # Set the data
            pp.initialize_data(
                mg, data_edge, scalar_key, {"normal_diffusivity": normal_diffusivity},
            )

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
                {self.scalar_variable: {"cells": 1},}  # noqa: E231
            )

        # Then for the edges
        for _, d in gb.edges():
            add_nonpresent_dictionary(d, primary_vars)

            d[primary_vars].update(
                {self.mortar_scalar_variable: {"cells": 1},}
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
        """ Discretize all terms
        """
        if not self.assembler:
            self.assembler = pp.Assembler(self.gb)

        self.assembler.discretize()

    # --- Initial condition ---

    def initial_scalar_condition(self) -> None:
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
            scalar_dof = np.hstack((scalar_dof, self.assembler.dof_ind(g, var_s)))

        # Unscaled pressure solutions
        scalar_now = solution[scalar_dof] * ss
        scalar_prev = prev_solution[scalar_dof] * ss
        scalar_init = init_solution[scalar_dof] * ss

        # Calculate norms
        # scalar_norm = np.sum(scalar_now ** 2)
        difference_in_iterates_scalar = np.sum((scalar_now - scalar_prev) ** 2)
        difference_from_init_scalar = np.sum((scalar_now - scalar_init) ** 2)

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        if (
            difference_in_iterates_scalar < tol_convergence
        ):  # and scalar_norm < tol_convergence
            converged = True
            error_scalar = difference_in_iterates_scalar
            logger.info(f"Pressure converged absolutely")
            logger.info(f"Absolute error in pressure is {error_scalar:.6e}.")
        else:
            # Relative convergence criterion:
            if (
                difference_in_iterates_scalar
                < tol_convergence * difference_from_init_scalar
            ):
                converged = True
                logger.info(f"Pressure converged relatively")

            error_scalar = difference_in_iterates_scalar / difference_from_init_scalar
            logger.info(
                f"Relative error in pressure is {error_scalar:.6e}. "
                f"(absolute error is {difference_in_iterates_scalar:.4e})"
            )

        if not converged:
            logger.info(f"Scalar pressure did not converge.")

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

    # --- Newton iterations ---

    def before_newton_loop(self):
        """ Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_scalar_parameters()

    def after_simulation(self):
        """ Called after a time-dependent problem
        """
        self.export_pvd()
        logger.info(f"Solution exported to folder \n {self.params.folder_name}")

    # --- Exporting and visualization ---

    def set_viz(self):
        """ Set exporter for visualization """
        super().set_viz()
        # list of time steps to export with visualization.
        self.export_times = []  # noqa

        self.p_exp = "p_exp"  # noqa
        self.aperture_exp = "aperture"  # noqa
        self.injection_cells = "injection_cells"  # noqa

        self.export_fields.extend([
            self.p_exp,
            self.aperture_exp,
            self.injection_cells,
        ])

    def export_step(self, write_vtk=True):
        """ Export a step with pressures """
        super().export_step(write_vtk=False)
        for g, d in self.gb:
            state = d[pp.STATE]
            state[self.aperture_exp] = self.aperture(g, scaled=False)
            # Export pressure variable
            if self.scalar_variable in state:
                state[self.p_exp] = (
                    state[self.scalar_variable].copy() * self.params.scalar_scale
                )
            else:
                state[self.p_exp] = np.zeros((self.Nd, g.num_cells))

            # Export injection cells
            state[self.injection_cells] = self.gb.node_props(g, "well")

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

        # --- PHYSICAL PARAMETERS ---

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
        self.network = None

        # ADJUST TIME STEP WITH PERMEABILITY
        # self.time_step = self.length_scale**2 / self.initial_permeability[None] / 1e10
        # self.end_time = self.time_step * 4

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
                    "shearzone_names",
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
        """ Set a grid bucket to the class
        """
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

    # def bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
    #     # Define boundary regions
    #     all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
    #     dir = np.where(east + west)[0]
    #     bc = pp.BoundaryCondition(g, dir, ["dir"] * dir.size)
    #     return bc
    #
    # def bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
    #     """
    #     Note that Dirichlet values should be divided by scalar_scale.
    #     """
    #     all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
    #     v = np.zeros(g.num_faces)
    #     v[west] = 1 * (pp.PASCAL / self.params.scalar_scale)
    #     return v

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
                f"very negative indices: p<-1e-10: count: {negneg_ind.size}, "
                f"indices: {negneg_ind}\n"
                f"neg pressure range: [{p_neg.min():.2e}, {p_neg.max():.2e}]\n"
            )
        else:
            summary_p = (
                f"{summary_p_common}"
                f"No negative pressure values. count:{neg_ind.size}\n"
            )

        self.neg_ind = neg_ind  # noqa
        self.negneg_ind = negneg_ind  # noqa

        # Condition number
        A, _ = self.assembler.assemble_matrix_rhs()  # noqa
        row_sum = np.sum(np.abs(A), axis=1)
        pp_cond = np.max(row_sum) / np.min(row_sum)
        diag = np.abs(A.diagonal())
        umfpack_cond = np.max(diag) / np.min(diag)

        summary_param = (
            f"\nSummary of relevant parameters:\n"
            f"length scale: {self.params.length_scale:.2e}\n"
            f"scalar scale: {self.params.scalar_scale:.2e}\n"
            f"3d permeability: "
            f"{self.initial_permeability[self.params.intact_name]:.2e}\n"
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
