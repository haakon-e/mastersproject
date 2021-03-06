import logging
import time
from typing import Dict

import numpy as np

import porepy as pp
from GTS import ContactMechanicsBiotBase
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.parameter import BiotParameters
from mastersproject.util.logging_util import timer, trace

logger = logging.getLogger(__name__)


class ISCBiotContactMechanics(ContactMechanicsBiotBase):
    def __init__(self, params: BiotParameters):
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
            for i, sz_name in enumerate(self.params.shearzone_names):
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
            pp.set_state(d, {"well": tags})

        # Set injection cells
        if self.params.well_cells:
            self.params.well_cells(self.params, self.gb)

    # ---- FLOW -----

    def biot_alpha(self, g: pp.Grid) -> float:
        return self.params.alpha if g.dim == self.Nd else 1.0

    # --- Aperture related methods ---

    def specific_volume(self, g: pp.Grid, scaled, from_iterate=True) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self.aperture(g, scaled, from_iterate)
        return np.power(a, self.Nd - g.dim)

    def aperture(
        self, g: pp.Grid, scaled: bool, from_iterate: bool = True,
    ) -> np.ndarray:
        """Compute the total aperture of each cell on a grid

        Parameters
        ----------
        g : pp.Grid
            grid
        scaled : bool
            whether to scale the aperture, which has units [m]
        from_iterate : bool
            whether to compute mechanical aperture from iterate.


        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        a_init = self.compute_initial_aperture(g, scaled=scaled)
        a_mech = self.mechanical_aperture(g, scaled=scaled, from_iterate=from_iterate)

        a_total = a_init + a_mech
        return a_total

    def mechanical_aperture(
        self, g: pp.Grid, scaled: bool, from_iterate: bool
    ) -> np.ndarray:
        """Compute aperture contribution from mechanical opening of fractures

        For 3d-matrix: zeros (aperture isn't really defined in 3d)
        For 2d-fracture: compute from the local displacement jump
        For 1d-intersection: Get the max from the two adjacent fractures

        Parameters
        ----------
        g : pp.Grid
            a grid
        scaled : bool
            whether to scale the displacement jump, which has units [m]
        from_iterate : bool
            whether to fetch displacement jump from from iterate or from pp.STATE.

        Returns
        -------
        aperture : np.ndarray
            Apertures for each cell in the grid.

        Notes
        -----
        This method relies on a current (pp.STATE) or previous ('previous_iterate')
        being set for the mortar displacement variable.
        In case pp.STATE doesn't exist, uniformly 0 is returned.
        Otherwise, we require values for the mortar displacement variable.
        One way to ensure this in practice, is to set initial conditions
        before discretizations
        """
        nd = self.Nd
        gb = self.gb
        nd_grid = self._nd_grid()

        def _aperture_from_edge(data: Dict):
            """Compute the mechanical contribution to the aperture
            for a given fracture. data is the edge data.
            """
            # Get normal displacement component from solution
            u_mortar_local = self.reconstruct_local_displacement_jump(
                data,
                from_iterate=from_iterate,
            )
            # Jump distances in each cell (index, -1, extracts normal component)
            aperture_contribution = np.abs(u_mortar_local[-1, :])

            # Mechanical aperture is scaled by default. "Upscale" if necessary.
            if not scaled:
                aperture_contribution *= self.params.length_scale / pp.METER

            return aperture_contribution

        # -- Computations --

        # No contribution in 3d-matrix
        if g.dim == nd:
            return np.zeros(g.num_cells)

        # For 2d & 1d, fetch the primary grids
        primary_grids = gb.node_neighbors(g, only_higher=True)
        de = [gb.edge_props((g, e)) for e in primary_grids]
        initialized = np.alltrue([pp.STATE in d for d in de])
        if not initialized:
            return np.zeros(g.num_cells)

        # In 2d-fractures
        elif g.dim == nd - 1:
            data_edge = gb.edge_props((g, nd_grid))
            aperture = _aperture_from_edge(data_edge)
            return aperture

        # TODO: Think about how to accurately do aperture computation (and specific volume)
        #  for fracture intersections where either side of fracture has different aperture.
        # In fracture intersections
        elif g.dim == nd - 2:
            # (g is the secondary grid.)
            # Fetch edges of g that points to a higher-dimensional grid
            intx_edges = [(g, g_h) for g_h in primary_grids]
            frac_edges = [(g_h, nd_grid) for g_h in primary_grids]
            frac_cell_faces = [g_h.cell_faces for g_h in primary_grids]

            # get apertures on the adjacent fractures
            data_edges = [gb.edge_props(edge) for edge in frac_edges]
            frac_apertures = [
                _aperture_from_edge(data_edge) for data_edge in data_edges
            ]

            # Map fracture apertures to internal faces ..
            frac_face_apertures = [
                np.abs(cell_face) * parent_aperture
                for cell_face, parent_aperture in zip(frac_cell_faces, frac_apertures)
            ]

            # .. then project face apertures to the interfaces and take maximum
            # Note: for matching grids, avg and int mortar projections are identical.
            mortar_grids = [gb.edge_props(edge)["mortar_grid"] for edge in intx_edges]
            mortar_apertures = [
                mg.primary_to_mortar_int() * ap
                for mg, ap in zip(mortar_grids, frac_face_apertures)
            ]
            # The reshape and max operations implicitly project the aperture to
            # the intersection grid (assuming a conforming mesh).
            intx_max_apertures = np.max(
                np.vstack(
                    [mortar_ap.reshape((2, -1)) for mortar_ap in mortar_apertures]
                ),
                axis=0,
            )

            return intx_max_apertures

        else:
            raise ValueError("Not implemented 1d intersection points")

    # --- Flow parameter related methods ---

    def permeability(self, g, scaled, from_iterate=True) -> np.ndarray:
        """ Set (uniform) permeability in a subdomain

        Modify parent method by passing from_iterate argument. This argument is
        needed by self.aperture().
        """
        return super().permeability(g, scaled, from_iterate=from_iterate)

    def intersection_volume_iterate(self, g):
        if g.dim == self.Nd - 2:
            V_k = self.specific_volume(g, scaled=True, from_iterate=True)
            V_n = self.specific_volume(g, scaled=True, from_iterate=False)
            return (V_k - V_n) * g.cell_volumes
        else:
            return np.zeros(g.num_cells)

    def bc_values_scalar(self, g) -> np.array:
        """ Hydrostatic pressure on the boundaries if gravity is set"""
        params = self.params
        # DIRICHLET
        all_bf, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)

        bf_centers = g.face_centers[:, all_bf]
        # Hydrostatic
        # Note: for scaled input height, unscaled depth is returned.
        if params.gravity:
            # Set to hydrostatic for depth of each boundary face
            depth = self.depth(bf_centers)
        else:
            # Set to uniform hydrostatic pressure at reference depth of 480 m.
            depth = self.depth(np.zeros_like(bf_centers))
        bc_values[all_bf] = (
            params.fluid.hydrostatic_pressure(depth) / params.scalar_scale
        )
        return bc_values

    # --- Constant pressure in tunnel cells ---

    def assemble_matrix_rhs(self):
        """ Modify global assembly to get constant pressure in tunnels"""
        A, b = self.assembler.assemble_matrix_rhs(matrix_format="csr")

        # Start of tunnel equilibration
        if self.time > -self.params.tunnel_equilibrium_time:
            pressure = self.params.tunnel_pressure * (
                pp.PASCAL / self.params.scalar_scale
            )
            rows_to_zero = self.inds_tunnel_scalar()
            # Modify global matrix and rhs
            self.csr_zero_rows(A, rows_to_zero)
            for glob_ind in rows_to_zero:
                A[glob_ind, glob_ind] = 1
                b[glob_ind] = pressure

        return A, b

    def inds_tunnel_scalar(self):
        """ Find scalar indices to zero out corresponding to tunnel cells"""
        glob_inds = []
        for g, d in self.gb:
            tunnels = g.tags["tunnel_cells"]  # type: np.ndarray[bool]
            if np.any(tunnels):
                dof_ind = self.assembler.dof_ind(g, self.scalar_variable)
                glob_ind = dof_ind[tunnels]
                glob_inds.append(glob_ind)
        rows_to_zero = (
            np.hstack(glob_inds) if len(glob_inds) > 0 else np.empty(0, dtype=np.bool)
        )
        return rows_to_zero

    def csr_zero_rows(self, csr, rows_to_zero):
        """Efficient way to set csr sparse matrix row to zero.

        https://stackoverflow.com/questions/19784868/
        what-is-most-efficient-way-of-setting-row-to-zeros-for-a-sparse-scipy-matrix/
        19800305#19800305"""
        rows, cols = csr.shape
        mask = np.ones((rows,), dtype=np.bool)
        mask[rows_to_zero] = False
        nnz_per_row = np.diff(csr.indptr)

        mask = np.repeat(mask, nnz_per_row)
        nnz_per_row[rows_to_zero] = 0
        csr.data = csr.data[mask]
        csr.indices = csr.indices[mask]
        csr.indptr[1:] = np.cumsum(nnz_per_row)

    def tag_tunnel_cells(self):
        """Tag tunnel-shearzone intersections

        Compute the nearest cell for each shear zone that intersects each of the
        tunnels.

        tunnel_sz is the following table:

            borehole   x_gts    y_gts   z_gts shearzone
        591       AU  72.625  125.321  33.436      S1_1
        592       VE   9.735   88.360  35.419      S1_1
        593       AU  74.565  135.311  33.858      S1_2
        594       VE  10.917   95.617  34.431      S1_2
        595       AU  74.839  143.317  33.611      S1_3
        596       VE  18.560  107.674  34.916      S1_3
        597       AU  72.094  106.617  32.813      S3_1
        598       VE  22.420  113.208  33.608      S3_1
        599       AU  72.185  110.025  33.639      S3_2
        600       VE  25.125  118.238  33.762      S3_2
        """
        tunnel_cells_key = "tunnel_cells"
        shearzones = self.params.shearzone_names
        tunnels = ["AU", "VE"]
        # Fetch tunnel-shearzone intersections
        isc_data = self.params.isc_data
        df = isc_data.structures
        _mask = df["borehole"].isin(tunnels) & df["shearzone"].isin(shearzones)
        t_sz = df[_mask]
        keepcols = ["borehole", "x_gts", "y_gts", "z_gts", "shearzone"]
        tunnel_sz = t_sz[keepcols].reset_index(drop=True)

        gb = self.gb
        for g, d in gb:
            g.tags[tunnel_cells_key] = np.zeros(g.num_cells, dtype=bool)
            pp.set_state(d, {tunnel_cells_key: np.zeros(g.num_cells, dtype=bool)})

        def _tag_intersection_cell(row):
            _coord = (
                row[["x_gts", "y_gts", "z_gts"]].to_numpy(dtype=float).reshape((3, -1))
            )
            coord = _coord * (pp.METER / self.params.length_scale)
            shearzone = row["shearzone"]
            grid: pp.Grid = self.grids_by_name(shearzone)[0]
            data = gb.node_props(grid)
            tags = np.zeros(grid.num_cells, dtype=bool)
            ids, dsts = grid.closest_cell(coord, return_distance=True)
            tags[ids] = True
            grid.tags[tunnel_cells_key] |= tags
            data[pp.STATE][tunnel_cells_key] |= tags

        tunnel_sz.apply(_tag_intersection_cell, axis=1)

    # --- Set flow parameters ---

    def set_scalar_parameters(self):
        """ See parent method

        Overwrite mass_weight and source.
        * For mass_weight, we set a more complex storage term that depends on the Biot coefficient
        * For source, we add a source term for the impact of expansion and contraction
            of 1d fracture intersections (see intersection_volume_iterate).
        """
        super().set_scalar_parameters()
        for g, d in self.gb:
            params: pp.Parameters = d[pp.PARAMETERS]
            scalar_params: dict = params[self.scalar_parameter_key]

            # Set mass weight for the biot problem
            #   mw = porosity * c + (alpha - porosity) / K
            #   Note: alpha and porosity are 1.0 in fractures,
            #         then mass_weight reduces to 'phi * c'.
            c = self.params.fluid.COMPRESSIBILITY
            porosity = self.porosity(g)
            alpha = self.biot_alpha(g)
            bulk = self.params.rock.BULK_MODULUS
            mass_weight = porosity * c + (alpha - porosity) / bulk
            mass_weight *= self.specific_volume(g, scaled=True)
            scalar_params["mass_weight"] = mass_weight

            # Set intersection transient source term
            scalar_params["source"] += self.intersection_volume_iterate(g)

    # --- Other flow related methods ---

    def assign_scalar_discretizations(self) -> None:
        """Assign k inverse scaling to the coupling discretization

        From IvaR:
        For long time steps, scaling the diffusive interface fluxes in the non-default
        way turns out to actually be beneficial for the condition number.
        """
        super().assign_scalar_discretizations()

        for e, d in self.gb.edges():
            d[pp.COUPLING_DISCRETIZATION][self.scalar_coupling_term][e][
                1
            ].kinv_scaling = True

    # --- MECHANICS ---

    def faces_to_fix(self, g: pp.Grid):
        """Fix some boundary faces to dirichlet to ensure unique solution to problem.

        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.

        Credits: Keilegavlen et al (2019) - Source code.
        """

        all_bf, *_ = self.domain_boundary_sides(g)
        point = np.array(
            [
                [(self.bounding_box["xmin"] + self.bounding_box["xmax"]) / 2],
                [(self.bounding_box["ymin"] + self.bounding_box["ymax"]) / 2],
                [self.bounding_box["zmin"]],
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

    def bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """We set Neumann values on all but a few boundary faces.
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

    def bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """Mechanical stress values as ISC

        All faces are Neumann, except 3 faces fixed
        by self.faces_to_fix(g), which are Dirichlet.

        If gravity is activated, the stress becomes lithostatic. The provided stress then
        corresponds to a true depth (here set to domain top, zmax). All stress components,
        including off-diagonal, are "scaled" by the depth, relative to the true depth.
        I.e. larger (compressive) stress below the true depth, and smaller above.
        """
        ss, ls = self.params.scalar_scale, self.params.length_scale
        # Retrieve the domain boundary
        all_bf, *_ = self.domain_boundary_sides(g)

        # Boundary values
        bc_values = np.zeros((g.dim, g.num_faces))

        # --- mechanical state ---
        # Get outward facing normal vectors for domain boundary, weighted for face area

        # 1. Get normal vectors on the faces. These are already weighed by face area.
        bf_normals: np.ndarray = g.face_normals
        # 2. Adjust direction so they face outwards
        flip_normal_to_outwards = np.where(g.cell_face_as_dense()[0, :] >= 0, 1, -1)
        outward_normals: np.ndarray = bf_normals * flip_normal_to_outwards
        bf_stress = np.dot(self.params.stress, outward_normals[:, all_bf])
        # Mechanical stress
        bc_values[:, all_bf] += bf_stress * (pp.PASCAL / ss)

        # --- gravitational forces ---
        # Boundary stresses are assumed to increase linearly with depth.
        # We assume all components of the tensor increase linearly, with
        # a factor relative to the pure vertical stress component.
        if self.params.gravity:
            # Set (unscaled) depth in the local coordinate system of the domain
            # where we consider the measured stress exact ..
            true_stress_depth = (
                (self.bounding_box["zmax"] + self.bounding_box["zmin"]) / 2 * ls
            )
            # .. and compute the relative (unscaled) depths in terms this reference depth.
            relative_depths: np.ndarray = (g.face_centers[2] * ls) - true_stress_depth

            # If the vertical stress is zero, raise.
            if np.abs(self.params.stress[2, 2]) < 1e-12:
                raise ValueError("Cannot set gravity if vertical stress is zero")
            # Each stress component scales relative to the vertical stress.
            stress_scaler = self.params.stress / self.params.stress[2, 2]

            # Lithostatic pressure
            gravity: np.ndarray = self.params.rock.lithostatic_pressure(relative_depths)
            lithostatic_stress = stress_scaler.dot(
                np.multiply(outward_normals, gravity)
            )
            lithostatic_bc = lithostatic_stress[:, all_bf]

            bc_values[:, all_bf] += lithostatic_bc * (pp.PASCAL / ss)

        # DIRICHLET
        faces = self.faces_to_fix(g)
        bc_values[:, faces] = 0  # / self.length scale

        return bc_values.ravel("F")

    def source_mechanics(self, g: pp.Grid) -> np.ndarray:
        """Gravity term.

        Gravity points downward, but we give the term
        on the RHS of the equation, thus we take the
        negative (i.e. the vector given will be
        pointing upwards)
        """
        values = np.zeros((self.Nd, g.num_cells))
        if self.params.gravity:
            # Gravity term
            scaling = self.params.length_scale / self.params.scalar_scale
            rho = self.params.rock.DENSITY
            values[2] = rho * pp.GRAVITY_ACCELERATION * scaling * g.cell_volumes
        return values.ravel("F")

    # --- Set mechanics parameters ---

    def rock_friction_coefficient(self, g: pp.Grid) -> np.ndarray:  # noqa
        """The friction coefficient is uniform, and equal to 1.

        Assumes self.set_rock() is called
        """
        return np.ones(g.num_cells) * self.params.rock.FRICTION_COEFFICIENT

    def set_mechanics_parameters(self) -> None:
        """ Set the dilation angle for slip in fractures"""
        super().set_mechanics_parameters()
        gb = self.gb

        # Add a little cohesion for numerical stability
        cohesion = self.params.cohesion / self.params.scalar_scale

        for g, d in gb:
            if g.dim == self.Nd - 1:
                params: pp.Parameters = d[pp.PARAMETERS]
                mech_params = {
                    "dilation_angle": self.params.dilation_angle,
                    "cohesion": cohesion,
                }
                params.update_dictionaries(
                    [self.mechanics_parameter_key],
                    [mech_params],
                )

    # --- Simulation and solvers ---

    def _prepare_grid(self):
        """Tag well cells right after creation.
        Called by self.prepare_simulation()
        """
        if self.gb is None:
            super()._prepare_grid()
        self.well_cells()  # tag well cells
        self.tag_tunnel_cells()  # tag tunnel cells

    @timer(logger, level="INFO")
    def before_newton_iteration(self) -> None:
        # Note: All parameters are updated *after* each Newton iteration.

        # Re-discretize the nonlinear term and all terms depending on the aperture
        # Here,
        #   !grad_p & !mpsa avoids re-discretizing the (3d) mechanics equation of Biot
        #   !div_u & !stabilization avoids re-discretizing the displacement term
        #       of the 3d flow equation.
        # In other words: We want to update the ..
        #   .. MPFA terms (k depends on aperture).
        #   .. Mass terms (depends on specific volume / aperture).
        term_filter = pp.assembler_filters.ListFilter(
            term_list=["!grad_p", "!div_u", "!stabilization", "!mpsa"]
        )
        self.assembler.discretize(term_filter)
        # Report on cells sticking, sliding, etc.:
        msg = "(Open,Sticking,Gliding)/Total: "
        for g, d in self.gb:
            if g.dim != self.Nd - 1:
                continue
            sz = d["name"]
            iterate = d[pp.STATE][pp.ITERATE]
            penetration = iterate["penetration"]
            sliding = iterate["sliding"]
            nsliding = np.sum(np.logical_and(sliding, penetration))
            nsticking = np.sum(np.logical_and(np.logical_not(sliding), penetration))
            nopen = np.sum(np.logical_not(penetration))
            msg += f"{sz}: ({nopen}, {nsticking}, {nsliding})/{sliding.size}. "
        logger.info(msg)

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        super().after_newton_iteration(solution_vector)
        # Update Biot parameters using aperture from iterate
        # (i.e. displacements from iterate)
        self.set_biot_parameters()

    # --- Exporting and visualization ---

    def export_step(self, write_vtk=True):

        # Export the pressure perturbation
        for g, d in self.gb:
            state = d[pp.STATE]
            initial_pressure = self.hydrostatic_pressure(g, scaled=False)
            if self.scalar_variable in state:
                p = state[self.scalar_variable] * self.params.scalar_scale
                p_perturb = p - initial_pressure
                state[self.p_perturb] = p_perturb
            else:
                state[self.p_perturb] = np.zeros(g.num_cells)

        # Export all other data
        super().export_step(write_vtk=True)

    def set_viz(self):
        super().set_viz()

        # Store pressure perturbation
        self.p_perturb = "p_perturb"  # noqa

        # Store well cells and tunnel cells
        # "well" state field defined by well_cells()
        # "tunnel_cells" state field defined by tag_tunnel_cells()
        self.export_fields.extend(
            [
                "well",
                "tunnel_cells",
                self.p_perturb,
            ]
        )

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
            f"max: {np.max(p):.2e}. "
            f"Mean: {np.mean(p):.2e}. "
            f"Min: {np.min(p):.2e}\n"
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
            f"{np.mean(self.permeability(self._nd_grid(), scaled=False)):.2e}\n"
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
