from typing import Optional, Dict, List

import numpy as np
import porepy as pp


class ModelHelperMethods:
    """ Common helper methods for any model implementation"""

    def __init__(self):
        # Grid
        self.gb: Optional[pp.GridBucket] = None
        self.Nd: Optional[int] = None
        self.bounding_box: Optional[Dict[str, int]] = None
        self.assembler: Optional[pp.Assembler] = None

        # Viz
        self.viz: Optional[pp.Exporter] = None
        self.export_fields: List = []

    def _nd_grid(self):
        """ Get the grid of the highest dimension. Assumes self.gb is set.
        """
        return self.gb.grids_of_dimension(self.Nd)[0]

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
