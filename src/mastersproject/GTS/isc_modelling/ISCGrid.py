from pathlib import Path
from typing import Dict, List

import porepy as pp
import numpy as np

from GTS.ISC_data.fracture import fracture_network


def create_grid(
        mesh_args: Dict[str, float],
        length_scale: float,
        box: Dict[str, float],
        shearzone_names: List[str],
        viz_folder_name: str,
):
    """ Create a GridBucket of a 3D domain with fractures defined by the ISC data set.

    shearzone_names are used to give names to each fracture grid. We assume that the
    order of names appearing in shearzone_names is preserved as fracture grids are constructed.


    Parameters
    ----------
    mesh_args : Dict[float]
        Mesh arguments (unscaled)
    length_scale : float
        length scale coefficient
    box : Dict[str, float]
        bounding box of domain (unscaled)
    shearzone_names : List[str]
        names of ISC shearzones to include or None
    viz_folder_name : str
        Path to store grid files

    Returns
    -------
        gb : pp.GridBucket
            The produced grid bucket
        network : pp.FractureNetwork3d
            fracture network

    """
    # Scale mesh args by length_scale:
    mesh_args = {k: v / length_scale for k, v in mesh_args.items()}
    # Scale bounding box by length_scale:
    box = {k: v / length_scale for k, v in box.items()}

    network = fracture_network(
        shearzone_names=shearzone_names,
        export_vtk=True,
        domain=box,
        length_scale=length_scale,
        network_path=f"{viz_folder_name}/fracture_network.vtu",
    )
    path = f"{viz_folder_name}/gmsh_frac_file"
    gb = network.mesh(mesh_args=mesh_args, file_name=path)

    pp.contact_conditions.set_projections(gb)

    # --- Set fracture grid names: ---
    # The 3D grid is tagged by 'None'
    # 2D fractures are tagged by their shearzone name (S1_1, S1_2, etc.)
    # 1D (and 0D) fracture intersections are tagged by 'None'.
    gb.add_node_props(
        keys=["name"]
    )  # Add 'name' as node prop to all grids. (value is 'None' by default)
    fracture_grids = gb.get_grids(lambda _g: _g.dim == gb.dim_max() - 1)

    # Set node property 'name' to each fracture with value being name of the shear zone.
    if fracture_grids.size > 0:
        for i, sz_name in enumerate(shearzone_names):
            gb.set_node_prop(fracture_grids[i], key="name", val=sz_name)
            # Note: Use self.gb.node_props(g, 'name') to get value.

    return gb, network


def create_structured_grid(length_scale: float, ):
    """ Create a structured 3d grid

    length_scale : float
        Length scale of physical dimension.
    """
    nx = np.array([20, 20, 20])
    physdims = np.array([300, 300, 300])
    gb = pp.meshing.cart_grid([], nx=nx, physdims=physdims / length_scale, )
    return gb


def structured_grid_1_frac(length_scale: float):
    nx = np.array([20, 20, 20])
    physdims = np.array([300, 300, 300])

    frac_pts = np.array(
        [[150, 150, 150, 150],
         [0, 300, 300, 0],
         [0, 0, 150, 150]])
    gb = pp.meshing.cart_grid([frac_pts], nx=nx, physdims=physdims / length_scale,)
    return gb


def structured_grid_1_frac_horizontal(length_scale: float):
    nx = np.array([20, 20, 20])
    physdims = np.array([300, 300, 300])

    frac_pts = np.array(
        [[0, 300, 300, 0],
         [150, 150, 150, 150],
         [0, 0, 150, 150]])
    gb = pp.meshing.cart_grid([frac_pts], nx=nx, physdims=physdims / length_scale,)
    return gb


def optimize_grid(in_file, out_file=None, method="", force=False, dim_tags=[]):
    """ Optimize a grid using an optimizer

    See: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py#L1444

    Parameters
    ----------
    in_file : str
        path to .msh file to be optimized
    out_file : str
        output file. By default, in_file+"optimized"
    method : str
        name of optimizer.
        Default: '' (gmsh default tetrahedral optimizer)
        Other options:
            'Netgen': Netgen optimizer
            'HighOrder': direct high-order mesh optimizer
            'HighOrderElastic': high-order elastic smoother
            'HighOrderFastCurving': fast curving algorithm
            'Laplace2D': Laplace smoothing
            'Relocate2D': Node relocation, 2d
            'Relocate3D': Node relocation, 3d
    force : bool
        If set, apply the optimization also to discrete entities
    dim_tags : List
        If supplied, only apply the optimizer to the given entities

    """
    assert Path(in_file).is_file()

    import gmsh

    gmsh.initialize()
    gmsh.open(in_file)

    gmsh.model.mesh.optimize(method=method, force=force, dimTags=dim_tags)


def create_unstructured_grid_fully_blocking_fracture(folder_name) -> pp.GridBucket:
    """ Domain with fully blocking fracture """
    domain = {
        'xmin': 0, 'ymin': 0, 'zmin': 0,
        'xmax': 300, 'ymax': 300, 'zmax': 300
    }

    frac_pts = np.array(
        [[50, 50, 250, 250],
         [0, 300, 300, 0],
         [0, 0, 300, 300]])
    frac = pp.Fracture(frac_pts)

    frac_network = pp.FractureNetwork3d(frac, domain)
    mesh_args = {"mesh_size_frac": 10, "mesh_size_min": 4.0, "mesh_size_bound": 40}

    gb = frac_network.mesh(mesh_args, file_name=folder_name + "/gmsh_frac_file")
    return gb


def two_intersecting_blocking_fractures(folder_name) -> pp.GridBucket:
    """ Domain with fully blocking fracture """
    domain = {
        'xmin': 0, 'ymin': 0, 'zmin': 0,
        'xmax': 300, 'ymax': 300, 'zmax': 300
    }

    frac_pts1 = np.array(
        [[50, 50, 250, 250],
         [0, 300, 300, 0],
         [0, 0, 300, 300]])
    frac1 = pp.Fracture(frac_pts1)
    frac_pts2 = np.array(
        [[300, 300, 50, 50],
         [0, 300, 300, 0],
         [50, 50, 300, 300]])
    frac2 = pp.Fracture(frac_pts2)

    frac_network = pp.FractureNetwork3d([frac1, frac2], domain)
    mesh_args = {"mesh_size_frac": 30, "mesh_size_min": 20, "mesh_size_bound": 60}

    gb = frac_network.mesh(mesh_args, file_name=folder_name + "/gmsh_frac_file")
    return gb
