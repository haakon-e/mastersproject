import porepy as pp
import numpy as np
import scipy.sparse as sps
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot


class FullBiotSetup(ContactMechanicsBiot):
    pass


class SingleFracSetupBiot(ContactMechanicsBiot):
    def __init__(self, time_step=1):
        """ Fix a 'bug' where 'time_step' is not assigned to self."""
        super().__init__()
        self.viz = None

        # Time
        self.time = 0
        self.time_step = 1 * self.length_scale ** 2
        self.end_time = self.time_step * 10
        num_steps = self.end_time / self.time_step + 1
        self.time_steps_array = np.linspace(start=0, stop=self.end_time, num=num_steps)
        self.step_count = np.arange(len(self.time_steps_array))
        self.current_step = self.step_count[0]

    def create_grid(self):
        """ Create a 3D grid with one fracture on
        a cube.

        The method assigns the following attributes to self:
            box (dict): Bounding box of the domain, defined by minimum and
                maximum values in each dimension.
            gb (pp.GridBucket): The produced grid bucket.
            Nd (int): Dimensions of the matrix.

        According to requirements by pp.models.ContactMechanics, the following method
        is called after gb is set:
         pp.contact_conditions.set_projections(self.gb)
        """
        domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3, "zmin": -3, "zmax": 3}
        self.box = domain

        # Fractures denoted by vertices
        f_1 = pp.Fracture(np.array([[0, 1, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))

        network = pp.FractureNetwork3d([f_1], domain=domain)
        mesh_args = {"mesh_size_frac": 0.2, "mesh_size_min": 0.2, "mesh_size_bound": 1}
        gb = network.mesh(mesh_args, ensure_matching_face_cell=False)
        self.gb = gb
        self.Nd = self.gb.dim_max
        pp.contact_conditions.set_projections(self.gb)
        print("Number of cells 3D grid ", self.gb.grids_of_dimension(3)[0].num_cells)
        breakpoint()

    def bc_values_scalar(self, g):
        """ Set boundary values to 1 (Neumann) on top face.
        0 (Dirichlet) on bottom face.
        0 (Neumann) otherwise.
        """
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        top_face = np.nonzero(top)[0]
        bc_val = np.zeros(g.num_faces)
        bc_val[top_face] = 1
        return bc_val

    def bc_type_scalar(self, g):
        """ Set boundary conditions dirichlet on bottom face.
        Neumann otherwise.
        """
        # Define boundary regions
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bottom_face = np.nonzero(bottom)[0]
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, bottom_face, "dir")

    def init_viz(self, overwrite=False):
        """ Initialize visualization.
        Will only create a new object if none exists.
        Alternatively, the existing exporter can be overwritten using 'overwrite'.

        """
        if (self.viz is None) or overwrite:
            # g3 = self.gb.grids_of_dimension(self.gb.dim_max())[0]
            self.viz = pp.Exporter(
                self.gb, name="test_biot", folder=self.viz_folder_name
            )

    def export_step(self):
        """ Implementation of export step"""
        export_fields = [self.displacement_variable + "_"]  # self.scalar_variable
        # Test out: Export a single grid.
        # g3 = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        # data = self.gb.node_props(g3)
        # export_data = data[pp.STATE][self.displacement_variable]
        # export_fields = {self.displacement_variable: export_data}
        self.viz.write_vtk(export_fields, time_step=self.current_step)

    def export_pvd(self):
        """ Implementation of export pvd"""
        num_steps = np.arange(len(self.time_steps_array))
        self.viz.write_pvd(num_steps)


def run_model():
    """ Set up and run the biot model.
    """
    model = SingleFracSetupBiot()
    model.prepare_simulation()
    model.init_viz()
    time_steps = model.time_steps_array

    # breakpoint()
    print("Starting simulation...")
    tol = 1e-10

    # Get fracture grid(s):
    # Set zero values there to facilitate export.
    g2_list = model.gb.grids_of_dimension(2)
    for g in g2_list:
        data = model.gb.node_props(g)
        data[pp.STATE]["u_"] = np.zeros((3, g.num_cells))

    # Get the 3D data.
    g3 = model.gb.grids_of_dimension(3)[0]
    d3 = model.gb.node_props(g3)
    # Solve problem
    for curr_step, step in enumerate(time_steps):
        model.time = step
        model.current_step = curr_step
        x = model.assemble_and_solve_linear_system(tol)  # Solve time step
        model.after_newton_convergence(x)  # Distribute solution

        # Get the state, transform it, and save to another state variable
        sol3 = d3[pp.STATE][model.displacement_variable]
        trsol3 = np.reshape(np.copy(sol3), newshape=(g3.dim, g3.num_cells), order="F")
        d3[pp.STATE][model.displacement_variable + "_"] = trsol3

        # breakpoint()

        model.export_step()

    print("Successful simulation.")
    model.export_pvd()

    return model


# NOTE: SINCE WE GET A FLUID PRESSURE IN BIOT, DON'T CHANGE BC TYPES YET.
# def bc_values(self, g):
#     """ Set boundary conditions for the mechanics problem.
#
#     This method overwrites the equivalent method in pp.models.ContactMechanics.
#     """
#
#     # Values for all Nd components, facewise
#     values = np.zeros((self.Nd, g.num_faces))
#
#     all_bf, east, west, north, south, top, bottom = domain_boundary_sides(g)
#     top_side = np.nonzero(top)[0]
#     # Push down on top boundary by a force-vector [0, 0, -1].
#     # Is associated with a neumann condition on top.
#     values[2, top_side] = -1
#     # Reshape according to PorePy convention
#     values = values.ravel("F")
#     return values
#
# def bc_type(self, g):
#     """ Set boundary type.
#     Dirichlet on all faces except top face, which is neumann.
#     Also set dirichlet on fracture faces.
#
#     Overwrites the same method in pp.models.ContactMechanics
#
#     """
#     # First call the standard method, which is dirichlet on
#     # all global boundaries. And Dirichlet on fracture faces.
#     bc = super().bc_type(g)
#
#     all_bf, east, west, north, south, top, bottom = domain_boundary_sides(g)
#     top_side = np.nonzero(top)[0]
#     bc.is_neu[:, top_side] = True
#     bc.is_dir[:, top_side] = False
