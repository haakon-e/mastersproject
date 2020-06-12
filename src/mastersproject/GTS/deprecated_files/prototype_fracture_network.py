""" This file is a prototype for a fracture network created by manually picking vertices of
the fractures given in the matlab visualization tool.

This is done manually to get a simple network, as well as quickly prototype a mesh for
numerical experiments.

The file contains the following classes and methods:
class PrototypeNetwork:
    Extracts manually picked shear zone data (data found in

"""
# Python-native packages
from pathlib import Path

# External packages
import pandas as pd
import numpy as np
import porepy as pp

# Internal packages
from GTS.fit_plane import convex_hull


class PrototypeNetwork:
    """ Extract a fracture network from prototype data

    Methods:
        read_data(): Return read pre-determined data from file system
        get_convex_hulls(data: dict): Return convex hull of dictionary of plane data points
        fracture_network(
            fracs: dict,
            name: str = None,
            export: bool = False): Return pp.FractureNetwork3d from dictionary of convex point sets.
        make_network(cls): Return pp.FractureNetwork3d from data from file system (using read_data()).
        """

    def __init__(self):
        self.name = "prototype"

    @staticmethod
    def read_data(path=None):
        """ The data below was manually picked from the Matlab visualization.

            Only for prototyping.

            Returns dict with keys 'S1_1', 'S1_2', 'S1_3', 'S3_1', 'S3_2'
            and values in each as np.array.shape == (3, n)
            i.e. rows = x, y, z
                columns = different coordinates
            """
        if path is None:
            path = Path.cwd() / "GTS/ISC_SZ_manual_picking.txt"
        skip = lambda x: x in range(13)
        df = pd.read_csv(path, sep="\s+", skiprows=skip)

        keys = np.unique(df[["Shearzone"]].values)
        data = {
            key: df[df["Shearzone"] == key][["x", "y", "z"]].to_numpy().T
            for key in keys
        }
        return data

    @staticmethod
    def get_convex_hulls(data: dict):
        """ Get convex hulls from the manually picked data.

        Parameters:
            data (dict): Dictionary of plane point clouds.
        """
        keys = list(data.keys())
        sz_vertices = {key: {} for key in keys}
        for key in keys:
            vert = convex_hull(data[key])
            sz_vertices[key] = vert.copy()

        return sz_vertices

    @staticmethod
    def fracture_network(
        fracs: dict, name: str = None, export: bool = False, **network_kwargs
    ):
        """ Export fracture network for visualization

        Parameters:
            fracs (dict): Dictionary of fracture vertices.
            name (str): Filename of exported .vtu file.
            export (bool):
        """
        fractures = [pp.Fracture(fracs[key]) for key in list(fracs.keys())]
        network = pp.FractureNetwork3d(fractures)

        domain = network_kwargs.get("domain", None)
        if domain is not None:
            network.impose_external_boundary(domain=domain)

        if export:
            if name is None:
                name = "fracture_network.vtu"
            if name[-4:] != ".vtu":
                name = name + ".vtu"
            network.to_vtk(name)

        return network

    @classmethod
    def make_network(cls, domain: dict = None, path=None):
        """ Import data and make network"""
        data = cls.read_data(path=path)
        sz_vertices = cls.get_convex_hulls(data)
        network = cls.fracture_network(sz_vertices, domain=domain)
        return network

    @classmethod
    def make_mesh(cls, mesh_args: dict, domain: dict = None, **mesh_kwargs):
        """ Import data and make meshed region.

        Parameters:
            mesh_args (dict): Should contain fields 'mesh_size_frac', 'mesh_size_min',
                which represent the ideal mesh size at the fracture, and the
                minimum mesh size passed to gmsh. Can also contain
                'mesh_size_bound', wihch gives the far-field (boundary) mesh size.
            domain (dict, Optional): boundaries of 3D domain.
                should have total 6 keys 'dmin', 'dmax' where d=x,y,z.
                If not specified, domain is determined from fracture extents.

        Returns:
            pp.GridBucket: Mixed-dimensional mesh

        Example:
            domain = {'xmin':-6, 'xmax':80, 'ymin': 55, 'ymax': 150, 'zmin':0, 'zmax':50}
            mesh_args = {'mesh_size_frac': 10, 'mesh_size_min':10}
            gb = PrototypeNetwork.make_mesh(mesh_args, domain)
        """
        root = Path(
            r"C:\Users\Haakon\OneDrive\Dokumenter\FORSKNING\mastersproject\src\mastersproject\GTS"
        )
        path = root / "ISC_SZ_manual_picking.txt"
        network = cls.make_network(domain, path=path)
        root_gmsh = Path(
            "C:\\Users\\Haakon\\OneDrive\\Dokumenter\\FORSKNING\\mastersproject\\src\\mastersproject\\"
        )
        gmsh_path = str(
            root_gmsh / "gmsh_frac_file"
        )  # TEMPORARY: Resolve python "bug".
        gb = network.mesh(mesh_args, file_name=gmsh_path, **mesh_kwargs)
        return gb
