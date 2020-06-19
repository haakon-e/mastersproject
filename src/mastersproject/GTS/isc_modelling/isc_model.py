import logging
import time

import numpy as np

import porepy as pp
from GTS.isc_modelling.ISCGrid import create_grid
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotBase
from GTS.isc_modelling.parameter import BiotParameters

from mastersproject.util.logging_util import trace

logger = logging.getLogger(__name__)


class ISCBiotContactMechanics(ContactMechanicsBiotBase):
    def __init__(self, params: BiotParameters):
        super().__init__(params)

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


        # -- GRAVITY OPTIONS

        # TODO: Add these options to the MechanicsParameters
        # Turn on/off mechanical gravity term
        self._gravity_src = params.dict().get("_gravity_src", False)

        # Turn on/off gravitational effects on (Neumann) mechanical boundary conditions
        self._gravity_bc = params.dict().get("_gravity_bc", False)

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

    # ---- FLOW -----

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

        return k * np.ones(g.num_cells)

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



    # --- MECHANICS ---


    def faces_to_fix(self, g: pp.Grid):
        """ Fix some boundary faces to dirichlet to ensure unique solution to problem.

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

    def bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
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
        bf_stress = np.dot(self.params.stress, outward_normals[:, all_bf])
        bc_values[:, all_bf] += bf_stress / self.params.scalar_scale  # Mechanical stress

        # --- gravitational forces ---
        # See init-method to turn on/off gravity effects (Default: OFF)
        if self._gravity_bc:
            lithostatic_bc = self._adjust_stress_for_depth(g, outward_normals)

            # NEUMANN
            bc_values[:, all_bf] += lithostatic_bc[:, all_bf] / self.params.scalar_scale

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
        true_stress_depth = self.bounding_box["zmax"] * self.params.length_scale

        # We assume the relative sizes of all stress components scale with sigma_zz.
        # Except if sigma_zz = 0, then we don't scale.
        if np.abs(self.params.stress[2, 2]) < 1e-12:
            logger.critical("The stress scaler is set to 0 since stress[2, 2] = 0")
            stress_scaler = np.zeros(self.params.stress.shape)
        else:
            stress_scaler = self.params.stress / self.params.stress[2, 2]

        # All depths are translated in terms of the assumed depth
        # of the given stress tensor.
        relative_depths = g.face_centers[2] * self.params.length_scale - true_stress_depth
        rho_g_h = self.params.rock.lithostatic_pressure(relative_depths)
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
        scaling = self.params.length_scale / self.params.scalar_scale
        values[2] = self.params.rock.lithostatic_pressure(g.cell_volumes) * scaling
        return values.ravel("F")

    # --- Set parameters ---

    def rock_friction_coefficient(self, g: pp.Grid) -> np.ndarray:  # noqa
        """ The friction coefficient is uniform, and equal to 1.

        Assumes self.set_rock() is called
        """
        return np.ones(g.num_cells) * self.params.rock.FRICTION_COEFFICIENT

