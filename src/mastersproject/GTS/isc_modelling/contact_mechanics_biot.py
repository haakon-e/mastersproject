import logging
from typing import Dict, Tuple

import numpy as np

import porepy as pp
from GTS.isc_modelling.mechanics import ContactMechanicsISC
from GTS.isc_modelling.mechanics import Mechanics
from GTS.isc_modelling.flow import Flow
from GTS.isc_modelling.parameter import BaseParameters
from mastersproject.util.logging_util import timer, trace
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot

logger = logging.getLogger(__name__)


class ContactMechanicsBiotBase(Flow, Mechanics):
    def __init__(self, params: BaseParameters):
        super().__init__(params)

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure = True

    # --- Set parameters ---

    def biot_alpha(self, g: pp.Grid) -> float:  # noqa
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
                    {"stabilization": stabilization_disc_s, }
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
        ).check_convergence(solution, prev_solution, init_solution, nl_params, )
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
        self.set_scalar_parameters()
        self.set_mechanics_parameters()

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
            p = d[pp.STATE]["previous_iterate"][self.scalar_variable]
        else:
            p = d[pp.STATE][self.scalar_variable]

        # Stress contribution from the scalar variable
        d[pp.STATE]["stress"] += matrix_dictionary[biot.grad_p_matrix_key] * p

        # Is it correct there is no contribution from the global boundary conditions?


class ContactMechanicsBiotISC(ContactMechanicsISC, ContactMechanicsBiot):
    """ Biot model with contact mechanics for the Grimsel Test Site ISC experiment.

    This class defines the model setup for the In-Situ Stimulation and Circulation (ISC)
    experiment at the Grimsel Test Site (GTS).

    See Berge et al (2019):
        Finite volume discretization for poroelastic media with fractures
        modeled by contact mechanics
    in particular, Equation (1), for details.

    """

    def __init__(self, params: dict):
        """ Initialize the Contact Mechanics Biot

        Parameters
        ----------
        params : dict
            Should contain the following key-value pairs:
                viz_folder_name : str
                    Absolute path to folder where grid and results will be stored
                mesh_args : dict[str, int]
                    Arguments for meshing of domain.
                    Required keys: 'mesh_size_frac', 'mesh_size_min, 'mesh_size_bound'
                bounding_box : d[str, int]
                    Bounding box of domain
                    Required keys: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
                shearzone_names : List[str]
                    Which shear-zones to include in simulation
                source_scalar_borehole_shearzone : dict[str, str]
                    Which borehole and shear-zone intersection to do injection in.
                    Required keys: 'shearzone', 'borehole'
                length_scale, scalar_scale : float : Optional
                    Length scale and scalar variable scale.
                    Default: 100, pp.GIGA, respectively.
        """

        logger.info(f"Initializing contact mechanics biot on ISC dataset")

        # --- BOUNDARY, INITIAL, SOURCE CONDITIONS ---
        self.source_scalar_borehole_shearzone = params.get(
            "source_scalar_borehole_shearzone"
        )

        super().__init__(params=params)

        # Set file name of the pre-run first.
        self.file_name = "initialize_run"

        # Time
        self.prepare_initial_run()

        # Initialize phase and injection rate
        self.current_phase = 0
        self.current_injection_rate = 0

        # --- PHYSICAL PARAMETERS ---
        self.set_rock_and_fluid()

        # For now, constant permeability in fractures
        # For details on values, see "2020-04-21 Møtereferat" (Veiledningsmøte)
        mean_frac_permeability = 4.9e-16 * pp.METER ** 2
        mean_intact_permeability = 2e-20 * pp.METER ** 2
        self.initial_permeability = {  # Unscaled
            "S1_1": mean_frac_permeability,
            "S1_2": mean_frac_permeability,
            "S1_3": mean_frac_permeability,
            "S3_1": mean_frac_permeability,
            "S3_2": mean_frac_permeability,
            None: mean_intact_permeability,  # 3D matrix
        }

        # Use cubic law to compute initial apertures in fractures.
        # k = a^2 / 12 => a=sqrt(12k)
        self.initial_aperture = {
            sz: np.sqrt(12 * k) for sz, k in self.initial_permeability.items()
        }
        self.initial_aperture[None] = 1  # Set 3D matrix aperture to 1.

        #
        # --- ADJUST CERTAIN PARAMETERS FOR TESTING ---

        # # Turn on/off scalar gravity term
        # self._gravity_src_p = params.get("_gravity_src_p", False)

        # Turn on/off gravitational effects on (Dirichlet) scalar boundary conditions
        self._gravity_bc_p = params.get("_gravity_bc_p", False)

    def bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values on all but a few boundary faces.
        Fracture faces also set to Dirichlet.

        Three boundary faces (see method faces_to_fix(self, g))
        are set to 0 displacement (Dirichlet).
        This ensures a unique solution to the problem.
        Furthermore, the fracture faces are set to 0 displacement (Dirichlet).
        """
        return super().bc_type(g)

    def bc_values_mechanics(self, g) -> np.array:
        """ Scaled mechanical stress values as ISC
        """
        return super().bc_values(g)

    def bc_values_scalar(self, g) -> np.array:
        """ Boundary condition values - zero or hydrostatic.

        Prescribe either homogenous pressure values
        or
        hydrostatic (depth-dependent) pressure values.
        credit: porepy paper
        """
        # DIRICHLET
        all_bf, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)

        # Hydrostatic
        if self._gravity_bc_p:
            depth = self._depth(g.face_centers[:, all_bf])
            bc_values[all_bf] += (
                    self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
            )
        return bc_values

    def bc_type_scalar(self, g) -> pp.BoundaryCondition:
        """ Known boundary conditions (Dirichlet)

        Dirichlet scalar BCs are prescribed on all external boundaries
        """
        all_bf, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, ["dir"] * all_bf.size)

    def source_flow_rate(self) -> float:
        """ Scaled source flow rate

        [From Grimsel Experiment Description]:

        We simulate one part of the injection procedure (Phase 3).
        In this phase, they inject by flow rate.
        Four injection steps of 10, 15, 20 and 25 l/min, each for 10 minutes.

        Afterwards, the system is shut-in for 40 minutes.
        """
        self.simulation_protocol()
        injection_rate = self.current_injection_rate  # injection rate [l / s], unscaled
        return injection_rate * pp.MILLI * (pp.METER / self.length_scale) ** self.Nd

    def well_cells(self) -> None:
        """
        Tag well cells with unity values, positive for injection cells and
        negative for production cells.
        """
        # TODO: Use unscaled grid to find result.
        df = self.isc.borehole_plane_intersection()
        # Borehole-shearzone intersection of interest
        bh_sz = self.source_scalar_borehole_shearzone

        _mask = (df.shearzone == bh_sz["shearzone"]) & (
                df.borehole == bh_sz["borehole"]
        )

        # Get the intersection coordinates of the borehole and the shearzone. (unscaled)
        result = df.loc[_mask, ("x_sz", "y_sz", "z_sz")]
        if result.empty:
            raise ValueError("No intersection found.")

        # Scale the intersection coordinates by length_scale. (scaled)
        pts = result.to_numpy().T / self.length_scale
        assert pts.shape[1] == 1, "Should only be one intersection"

        # Loop through the grids. Tag all cells with 0.
        # The closest discrete cell to the intersection coordinates is tagged with 1.
        tagged = False
        for g, d in self.gb:
            # By default: All cells are tagged with 0 (not the closest point).
            tags = np.zeros(g.num_cells)

            # Get name of grid. See ContactMechanicsISC.create_grid() for details.
            grid_name = self.gb.node_props(g, "name")

            # We only tag cells in the desired fracture
            if grid_name == bh_sz["shearzone"]:
                logger.info(f"Tagging grid of name: {grid_name}, and dimension {g.dim}")
                logger.info(f"Setting non-zero source value for pressure")

                ids, dsts = g.closest_cell(pts, return_distance=True)
                # TODO: log distance in unscaled lengths
                logger.info(
                    f"Closest cell found has (unscaled) distance: "
                    f"{dsts[0] * self.length_scale:4f}"
                )

                # Tag the injection cell
                tags[ids] = 1
                tagged = True

            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

        if not tagged:
            logger.warning("No injection cell was tagged.")

    def source_scalar(self, g: pp.Grid) -> np.array:
        """ Well-bore source

        This is an example implementation of a borehole-fracture source.
        """
        flow_rate = self.source_flow_rate()  # Already scaled by self.length_scale
        values = flow_rate * g.tags["well_cells"] * self.time_step

        # TODO: This is wrong: scalar contribution has (integrated) units [m3 / s]
        #   Perhaps this should just be zero (ref porepy-paper code)
        # TODO: Q2: Is source flow rate on top of hydrostatic pressure or
        #  set absolutely?
        # Hydrostatic contribution
        # if self._gravity_src_p:
        #     depth = self._depth(g.cell_centers)
        #     values += self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
        return values

    def source_mechanics(self, g) -> np.array:
        """ Scaled gravity term. """
        return super().source(g)

    def set_permeability_from_aperture(self) -> None:
        """ Set permeability by cubic law in fractures.

        Currently, we simply set the initial permeability.
        """

        # Scaled dynamic viscosity
        viscosity = self.fluid.dynamic_viscosity() * (pp.PASCAL / self.scalar_scale)

        gb = self.gb
        param_key = self.scalar_parameter_key
        for g, d in gb:
            # get the shearzone
            shearzone = self.gb.node_props(g, "name")

            # permeability [m2] (scaled)
            k = (
                    self.initial_permeability[shearzone]
                    * (pp.METER / self.length_scale) ** 2
            )

            # Multiply by the volume of the flattened dimension (specific volume)
            k *= self.specific_volume(g, scaled=True)

            kxx = k / viscosity
            diffusivity = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][param_key]["second_order_tensor"] = diffusivity

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
            diffusivity = data_l[pp.PARAMETERS][param_key][
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
                mg, data_edge, param_key, {"normal_diffusivity": normal_diffusivity},
            )

    def aperture(self, g: pp.Grid, scaled=False) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        # Get the aperture in the corresponding shearzone (is 1 for 3D matrix)
        shearzone = self.gb.node_props(g, "name")
        aperture *= self.initial_aperture[shearzone]

        if scaled:
            aperture *= pp.METER / self.length_scale

        return aperture

    def specific_volume(self, g: pp.Grid, scaled=False) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self.aperture(g, scaled)
        return np.power(a, self.Nd - g.dim)

    def set_rock_and_fluid(self) -> None:
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature effects.
        Credits: PorePy paper
        """

        super().set_rock()

        # Fluid. Temperature at ISC is 11 degrees average.
        self.fluid = WaterISC(theta_ref=11)

    def set_parameters(self) -> None:
        """ Set biot parameters
        """
        self.set_mechanics_parameters()
        self.set_scalar_parameters()

    def set_mechanics_parameters(self) -> None:
        """ Set mechanics parameters for the simulation.
        """
        # TODO Consider calling super().set_parameters(),
        #  then set the remaining parameters here.
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = self.rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = self.rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self.bc_type_mechanics(g)
                # BC and source values
                bc_val = self.bc_values_mechanics(g)  # Already scaled
                source_val = self.source_mechanics(g)  # Already scaled

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        "time_step": self.time_step,
                        "biot_alpha": self.biot_alpha(g),
                    },
                )

            elif g.dim == self.Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction, "time_step": self.time_step},
                )

        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    def set_scalar_parameters(self) -> None:
        """ Set scalar parameters for the simulation
        """
        gb = self.gb

        compressibility = self.fluid.COMPRESSIBILITY * (
                self.scalar_scale / pp.PASCAL
        )  # scaled. [1/Pa]
        porosity = self.rock.POROSITY
        for g, d in gb:
            # specific volume
            specific_volume = self.specific_volume(g, scaled=True)

            # Boundary and source conditions
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)  # Already scaled
            source_values = self.source_scalar(g)  # Already scaled

            # Biot alpha
            alpha = self.biot_alpha(g)

            # Initialize data
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": (  # TODO: Simplified version off mass_weight?
                            compressibility
                            * porosity
                            * specific_volume
                            * np.ones(g.num_cells)
                    ),
                    "biot_alpha": alpha,
                    "source": source_values,
                    "time_step": self.time_step,
                },
            )

        # Set permeability on grid, fracture and mortar grids.
        self.set_permeability_from_aperture()

    def set_viz(self):
        """ Set exporter for visualization """
        self.viz = pp.Exporter(
            self.gb, file_name=self.file_name, folder_name=self.viz_folder_name
        )
        # list of time steps to export with visualization.
        self.export_times = []

        self.u_exp = "u_exp"
        self.p_exp = "p_exp"
        self.traction_exp = "traction_exp"
        self.normal_frac_u = "normal_frac_u"
        self.tangential_frac_u = "tangential_frac_u"

        self.export_fields = [
            self.u_exp,
            self.p_exp,
            # self.traction_exp,
            self.normal_frac_u,
            self.tangential_frac_u,
        ]

    @trace(logger, timeit=False)
    def export_step(self):
        """ Export a step

        Inspired by Keilegavlen 2019 (code)
        """

        self.save_frac_jump_data()  # Save fracture jump data to pp.STATE
        gb = self.gb
        Nd = self.Nd
        ss = self.scalar_scale
        ls = self.length_scale

        for g, d in gb:
            # Export pressure variable
            if self.scalar_variable in d[pp.STATE]:
                d[pp.STATE][self.p_exp] = d[pp.STATE][self.scalar_variable].copy() * ss
            else:
                d[pp.STATE][self.p_exp] = np.zeros((Nd, g.num_cells))

            if g.dim != 2:  # We only define tangential jumps in 2D fractures
                d[pp.STATE][self.normal_frac_u] = np.zeros(g.num_cells)
                d[pp.STATE][self.tangential_frac_u] = np.zeros(g.num_cells)

            if g.dim == Nd:  # On matrix
                u = (
                        d[pp.STATE][self.displacement_variable]
                        .reshape((Nd, -1), order="F")
                        .copy()
                        * ls
                )

                if g.dim != 3:  # Only called if solving a 2D problem
                    u = np.vstack(u, np.zeros(u.shape[1]))

                d[pp.STATE][self.u_exp] = u

                d[pp.STATE][self.traction_exp] = np.zeros(d[pp.STATE][self.u_exp].shape)

            else:  # In fractures or intersection of fractures (etc.)
                # Get the higher-dimensional neighbor
                g_h = gb.node_neighbors(g, only_higher=True)[0]
                if g_h.dim == Nd:  # In a fracture
                    data_edge = gb.edge_props((g, g_h))
                    # TODO: Should I instead export the fracture displacement
                    #  in global coordinates?
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge=data_edge, from_iterate=True
                    ).copy()
                    u_mortar_local = u_mortar_local * ls

                    traction = d[pp.STATE][self.contact_traction_variable].reshape(
                        (Nd, -1), order="F"
                    )

                    if g.dim == 2:
                        d[pp.STATE][self.u_exp] = u_mortar_local
                        d[pp.STATE][self.traction_exp] = traction * ss * ls ** 2
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
            data=self.export_fields, time_step=self.time
        )  # Write visualization
        self.export_times.append(self.time)

    def export_pvd(self):
        """ Implementation of export pvd"""
        self.viz.write_pvd(self.export_times)

    # def initial_condition(self):
    #     """
    #     Initial guess for Newton iteration, scalar variable and bc_values (for time
    #     discretization).
    #
    #     When stimulation phase is reached, we use displacements of last solution in
    #     initialize phase as initial condition for the cell displacements.
    #     """
    #     super().initial_condition()
    #     # TODO: Set hydrostatic initial condition
    #     for g, d in self.gb:
    #         depth = self._depth(g.cell_centers)
    #         initial_scalar_value = (
    #               self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
    #         )
    #         d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
    #
    #     # TODO What hydrostatic scalar initial condition should be set on
    #     #      mortar grid? lower dim value?
    #     for _, d in self.gb.edges():
    #         mg = d["mortar_grid"]
    #         initial_value = np.zeros(mg.num_cells)
    #         d[pp.STATE][self.mortar_scalar_variable] = initial_value
    #
    #     # TODO: Scale variables
    #     if self.current_phase > 0:  # Stimulation phase
    #
    #         for g, d in self.gb:
    #             if g.dim == self.Nd:
    #                 initial_displacements = d["initial_cell_displacements"]
    #                 pp.set_state(d, {
    #                     self.displacement_variable: initial_displacements
    #                 })
    #
    #         for e, d in self.gb.edges():
    #             if e[0].dim == self.Nd:
    #                 try:
    #                     initial_displacements = d["initial_cell_displacements"]
    #                 except KeyError:
    #                     logger.warning(
    #                         "We got KeyError on d['initial_cell_displacements']."
    #                     )
    #                     mg = d["mortar_grid"]
    #                     initial_displacements = np.zeros(mg.num_cells * self.Nd)
    #                 state = {
    #                     self.mortar_displacement_variable: initial_displacements,
    #                     "previous_iterate": {
    #                         self.mortar_displacement_variable: initial_displacements,
    #                     },
    #                 }
    #                 pp.set_state(d, state)

    @trace(logger)
    def before_newton_loop(self):
        """ Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self.set_parameters()
        # The following is expensive, as it includes Biot. Consider
        # making a custom  method discretizing only the term you need!

        # TODO: Discretize only the terms you need.
        # The LHS of mechanics equation has no time-dependent terms (3D)
        # The LHS of flow equation has no time-dependent terms (3D)
        # We don't update fracture aperture, so flow on the fracture
        # is also not time-dependent (LHS) (2D)
        # self.discretize()

    @trace(logger, timeit=False)
    def after_newton_convergence(self, solution, errors, iteration_counter):
        """ Overwrite from parent to export solution steps."""
        self.assembler.distribute_variable(solution)
        self.save_mechanical_bc_values()
        self.export_step()

    def after_simulation(self):
        """ Called after a time-dependent problem
        """
        self.export_pvd()
        logger.info(f"Solution exported to folder \n {self.viz_folder_name}")

    def after_newton_failure(self, solution, errors, iteration_counter):
        """ Instead of raising error on failure, save and return available data.
        """
        logger.error("Newton iterations did not converge")
        self.after_newton_convergence(solution, errors, iteration_counter)

        self.after_simulation()
        return self

    def prepare_initial_run(self):
        """
        Set time parameters for the preparation phase

        First, we run no flow for 6 hours to observe
        deformation due to mechanics itself.
        Then, [from Grimsel Experiment Description]:
        flow period is 40 minutes, followed by a shut-in period of 40 minutes.
        """

        # For the initialization phase, we use the following
        # # start time
        # self.time = - 72 * pp.HOUR
        # # time step
        # self.time_step = 24 * pp.HOUR
        # # end time
        # self.end_time = 0
        self.time = 0
        self.time_step = 1 * pp.DAY
        self.end_time = self.time_step * 3

    def prepare_main_run(self):
        """ Adjust parameters between initial run and main run

        Total time: 80 minutes.
        Time step: 5 minutes
        """

        # New file name for this run
        self.file_name = "main_run"
        self.set_viz()

        # We use the following time parameters
        # start time
        self.time = 0
        # time step
        self.time_step = 5 * pp.MINUTE
        # end time
        self.end_time = 20 * pp.MINUTE  # TODO: Change back to 40 minutes.

        # Store initial displacements
        for g, d in self.gb:
            if g.dim == 3:
                u = d[pp.STATE][self.displacement_variable]
                d["initial_cell_displacements"] = u

        for e, d in self.gb.edges():
            if e[0].dim == self.Nd:
                u = d[pp.STATE][self.mortar_displacement_variable]
                d["initial_cell_displacements"] = u

    def simulation_protocol(self) -> None:
        """ Adjust time step and other parameters for simulation protocol

                Here, we consider Doetsch et al (2018) [see e.g. p. 78/79 or App. J]
                Hydro Shearing Protocol:
                * Injection Cycle 3:
                    - Four injection steps of 10, 15, 20 and 25 l/min
                    - Each step lasts 10 minutes.
                    - Then, the interval is shut-in and monitored for 40 minutes.
                    - Venting was forseen at 20 minutes

                For this setup, we only consider Injection Cycle 3.

                Attributes set here:
                    current_phase : int
                        phase as a number (0 - 5)
                    current_injection_rate : float
                        fluid injection rate (l/min)
                """
        time_intervals = [
            # Phase 0: 0 l/min
            1e-10,
            # Phase 1: 10 l/min
            10 * pp.MINUTE,
            # Phase 2: 15 l/min
            20 * pp.MINUTE,
            # Phase 3: 20 l/min
            30 * pp.MINUTE,
            # Phase 4: 25 l/min
            40 * pp.MINUTE,
            # Phase 5: 0 l/min
        ]

        # TODO: TEMPORARY CONSTANT INJECTION
        injection_amount = [1e-6] * 6

        # injection_amount = [
        #     0,      # Phase 0
        #     10,     # Phase 1
        #     15,     # Phase 2
        #     20,     # Phase 3
        #     25,     # Phase 4
        #     0,      # Phase 5
        # ]
        next_phase = np.searchsorted(time_intervals, self.time, side="right")
        if next_phase > self.current_phase:
            logger.info(
                f"A new phase has started: Phase {next_phase}. "
                f"Injection set to {injection_amount[next_phase]} l/min"
            )

        # Current phase number:
        self.current_phase = next_phase

        # Current injection amount [litres / second]
        self.current_injection_rate = injection_amount[self.current_phase] / pp.MINUTE

    @timer(logger)
    def prepare_simulation(self):
        """ Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, linear solvers etc.


        ONLY CHANGE FROM PARENT:
        - Set self.viz with custom method.
        """
        self.create_grid()
        self.Nd = self.gb.dim_max()
        self.well_cells()  # Tag the well cells
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
        self.discretize()
        self.initialize_linear_solver()

        self.set_viz()

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

        mech_dof = self.assembler.dof_ind(g_max, self.displacement_variable)
        scalar_dof = self.assembler.dof_ind(g_max, self.scalar_variable)

        # Also find indices for the contact variables
        contact_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            if e[0].dim == self.Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.assembler.dof_ind(e[1], self.contact_traction_variable),
                    )
                )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        u_mech_now = solution[mech_dof] * self.length_scale
        u_mech_prev = prev_solution[mech_dof] * self.length_scale
        u_mech_init = init_solution[mech_dof] * self.length_scale

        # TODO: Check if scaling of contact variable
        contact_now = solution[contact_dof] * self.scalar_scale * self.length_scale ** 2
        contact_prev = (
                prev_solution[contact_dof] * self.scalar_scale * self.length_scale ** 2
        )
        contact_init = (
                init_solution[contact_dof] * self.scalar_scale * self.length_scale ** 2
        )

        # Pressure solution
        p_scalar_now = solution[scalar_dof] * self.scalar_scale
        p_scalar_prev = prev_solution[scalar_dof] * self.scalar_scale
        p_scalar_init = init_solution[scalar_dof] * self.scalar_scale

        # Calculate errors

        # Displacement error
        difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
        difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)

        logger.info(f"diff iter u = {difference_in_iterates_mech:.6e}")
        logger.info(f"diff init u = {difference_from_init_mech:.6e}")

        # Contact traction error
        # TODO: Unsure about units of contact traction
        # contact_norm = np.sum(contact_now ** 2)
        difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2)
        difference_from_init_contact = np.sum((contact_now - contact_init) ** 2)

        logger.info(f"diff iter contact = {difference_in_iterates_contact:.6e}")
        logger.info(f"diff init contact = {difference_from_init_contact:.6e}")

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

        # Converge in displacement and pressure on 3D grid
        converged_u = False
        converged_p = False

        # Check absolute convergence criterion
        if difference_in_iterates_mech < tol_convergence:
            # converged = True
            converged_u = True
            error_mech = difference_in_iterates_mech
            logger.info(f"u converged absolutely.")

        else:
            # Check relative convergence criterion
            if (
                    difference_in_iterates_mech
                    < tol_convergence * difference_from_init_mech
            ):
                # converged = True
                converged_u = True
                logger.info(f"u converged relatively")
            error_mech = difference_in_iterates_mech / difference_from_init_mech

        # The if is intended to avoid division through zero
        if difference_in_iterates_contact < 1e-10:
            # converged = True
            error_contact = difference_in_iterates_contact
            logger.info(f"contact variable converged absolutely")
        else:
            error_contact = (
                    difference_in_iterates_contact / difference_from_init_contact
            )

        # -- Scalar solution --
        # The if is intended to avoid division through zero
        if difference_in_iterates_scalar < tol_convergence:
            converged_p = True
            error_scalar = difference_in_iterates_scalar
            logger.info(f"pressure converged absolutely")
        else:
            # Relative convergence criterion:
            if (
                    difference_in_iterates_scalar
                    < tol_convergence * difference_from_init_scalar
            ):
                # converged = True
                converged_p = True
                logger.info(f"pressure converged relatively")

            error_scalar = difference_in_iterates_scalar / difference_from_init_scalar

        logger.info(f"Error in contact force is {error_contact:.6e}")
        logger.info(f"Error in matrix displacement is {error_mech:.6e}")
        logger.info(f"Error in pressure is {error_scalar:.6e}.")

        converged = converged_p and converged_u

        return error_mech, converged, diverged

    def _depth(self, coords):
        """
        Unscaled depth. We center the domain at 480m below the surface.
        (See Krietsch et al, 2018a)
        """
        return 480.0 * pp.METER - self.length_scale * coords[2]


# TODO: Extract the stimulation protocol from ContactMechanicsBiot by the principle of
#  separation of data from model.
class StimulationProtocol:
    """ This class describes a stimulation protocol for the Biot equations
    """

    def __init__(self):
        pass


class WaterISC(pp.Water):
    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)

    def aperture_from_transmissivity(self, T, b, theta=None) -> float:
        """ Compute hydraulic aperture [m] from transmissivity [m2/s]

        We use the following relation (derived from cubic law):
        a = sqrt( 12 * mu * T / (rho * g * b) )
        where mu is dynamic viscosity [Pa s], rho is density [kg/m3],
        g is gravitational acceleration [m/s2], and b is aquifer thickness [m]

        Assumes that self.fluid is set
        """
        mu = self.dynamic_viscosity(theta=theta)
        rho = self.density(theta=theta)
        g = pp.GRAVITY_ACCELERATION
        hydraulic_aperture = np.sqrt(12 * mu * T / (rho * g * b))
        return hydraulic_aperture

    def permeability_from_transmissivity(self, T, b, theta=None) -> float:
        """ Compute permeability [m2] from transmissivity [m2/s]

        We can relate permeability, k [m2] with transmissivity, T [m2/s]
        through the relation
        k = T * mu / (rho * g * b)
        where mu is dynamic viscosity [Pa s], rho is density [kg/m3],
        g in gravitational acceleration [m/s2] and b is aquifer thickness [m]

        Assumes that self.fluid is set
        """
        mu = self.dynamic_viscosity(theta=theta)
        rho = self.density(theta=theta)
        g = pp.GRAVITY_ACCELERATION
        k = T * mu / (rho * g * b)
        return k
