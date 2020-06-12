"""
This is a setup file to solve linear elasticity with contact between fractures. We do not consider any fluid,
and solve only for the linear elasticity coupled to the contact.

The domain consists of a 3d matrix and five fractures. two fractures are intersecting the three other fractures.
One of the fractures contains an injection well.
"""

import numpy as np
import porepy as pp
from porepy.models.contact_mechanics_model import ContactMechanics

from GTS.deprecated_files.prototype_fracture_network import PrototypeNetwork


class IsotropicSetup(ContactMechanics):
    """
    In this setup, we consider isotropic mechanical conditions.

    """

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 3D domain with five fractures.

        Suggested values for mesh_args:
        mesh_args = {'mesh_size_frac': 10, 'mesh_size_min': 10}
        """
        mesh_args = {"mesh_size_frac": 10, "mesh_size_min": 10}
        domain = {
            "xmin": -6,
            "xmax": 80,
            "ymin": 55,
            "ymax": 150,
            "zmin": 0,
            "zmax": 50,
        }
        gb = PrototypeNetwork.make_mesh(mesh_args, domain)
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.box = domain
        self.Nd = gb.dim_max()

        return gb

    def faces_to_fix(self, g):
        """
        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.
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
        indexes = np.argsort(distances)
        faces = all_bf[indexes[: self.Nd]]
        return faces

    def bc_type(self, g):
        """
        Define type of boundary conditions.
        Here, we consider Neumann on all sides, except at three points (for uniqueness).
        """
        all_bf = g.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(g, all_bf, "neu")

        # Set boundary condition on sub_faces to Dirichlet.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        # Set three faces to dirichlet for a unique solution:
        faces = self.faces_to_fix(g)
        bc.is_neu[:, faces] = False
        bc.is_dir[:, faces] = True

        return bc

    def bc_values(self, g):
        """
        Set boundary conditions on all boundary faces

        in x: sigma_1 = 13.1 MPa
        in y: sigma_2 = 9.2 MPa
        in z: sigma_3 = 8.7 MPa
        """
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        A = g.face_areas

        bc_values = np.zeros((self.Nd, g.num_faces))

        # Set values
        # x
        bc_values[0, west] = 13.1 * pp.MEGA * A[west]
        bc_values[0, east] = 13.1 * pp.MEGA * A[east]
        # y
        bc_values[1, south] = 9.2 * pp.MEGA * A[south]
        bc_values[1, north] = 9.2 * pp.MEGA * A[north]
        # z
        bc_values[2, top] = 8.7 * pp.MEGA * A[top]
        bc_values[2, bottom] = 8.7 * pp.MEGA * A[bottom]

        # Set the artifical Dirichlet faces:
        faces = self.faces_to_fix(g)
        bc_values[:, faces] = 0

        return bc_values.ravel("F")

    # TODO: Find out what this parameter should be.
    # def _set_friction_coefficient(self, g):
    #     """ The friction coefficient is uniform, and equal to 1.
    #     """
    #     return np.ones(g.num_cells)

    def set_parameters(self):
        """
        Set the parameters for the simulation.

        Lambda = 34.5 GPa
        Mu = 17.8 GPa
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = 34.5 * pp.GIGA * np.ones(g.num_cells)
                mu = 17.8 * pp.GIGA * np.ones(g.num_cells)
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self.bc_type(g)

                # BC and source values
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

    def init_viz(self, overwrite=True):
        """ Initialize visualization.
        Will only create a new object if none exists.
        Alternatively, the existing exporter can be overwritten using 'overwrite'.

        """
        if (self.viz is None) or overwrite:
            # g3 = self.gb.grids_of_dimension(self.gb.dim_max())[0]
            self.viz = pp.Exporter(
                self.gb, name="mechanics", folder=self.viz_folder_name
            )

    def export_step(self):
        """ Implementation of export step"""
        export_fields = [self.displacement_variable + "_"]  # self.scalar_variable
        self.viz.write_vtk(export_fields)  # , time_step=1)
        # self.viz.write_pvd([1])


def run_model():
    """
    Set up and run isotropic setup model.    
    """
    params = {"folder_name": "GTS/prototype_1/isotropic_setup_viz"}
    model = IsotropicSetup(params=params)
    model.prepare_simulation()
    model.init_viz()

    tol = 1e-10

    # Get fracture grid(s):
    # Set zero values there to facilitate export.
    for i in [1, 2]:
        g_list = model.gb.grids_of_dimension(i)
        for g in g_list:
            data = model.gb.node_props(g)
            data[pp.STATE]["u_"] = np.zeros((3, g.num_cells))

    # Get the 3D data.
    g3 = model.gb.grids_of_dimension(3)[0]
    d3 = model.gb.node_props(g3)
    # Solve problem
    x = model.assemble_and_solve_linear_system(tol)
    model.after_newton_convergence(x)  # Distribute solution

    # Get the state, transform it, and save to another state variable
    sol3 = d3[pp.STATE][model.displacement_variable]
    trsol3 = np.reshape(np.copy(sol3), newshape=(g3.dim, g3.num_cells), order="F")
    d3[pp.STATE][model.displacement_variable + "_"] = trsol3

    # Export solution
    model.export_step()

    return model
