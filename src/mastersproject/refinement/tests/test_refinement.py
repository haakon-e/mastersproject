from pathlib import Path
import os

import porepy as pp
import numpy as np

from refinement.grid_refinement import refine_mesh


def setup(tail_folders):
    """ Setup method.

    Set up a fracture network and mesh it.
    """
    pth = Path(os.path.abspath(__file__))
    root = pth.parent / tail_folders
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
    domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
    network = pp.FractureNetwork3d([f_1], domain=domain)
    # mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 2, "mesh_size_min": 0.1}
    mesh_args = {"mesh_size_bound": 10, "mesh_size_frac": 10, "mesh_size_min": 10}
    file_name = str(root / 'test')
    gb = network.mesh(mesh_args=mesh_args, file_name=file_name)
    return network, file_name, gb


# Test script
def test_refine():
    network, file_name, gb = setup(tail_folders='test_refine')

    # Refine the mesh
    gb_list = refine_mesh(
        in_file=f'{file_name}.geo', out_file=f"{file_name}.msh",
        dim=3, network=network, num_refinements=1)

    gb_ref = gb_list[-1]

    return gb, gb_ref


def test_refine_n_times():
    """ Test that a given grid is refined n times"""
    network, file_name, gb = setup(tail_folders='test_refine_n_times')

    n = 3

    # Refine the mesh
    gb_list = refine_mesh(
        in_file=f'{file_name}.geo', out_file=f"{file_name}.msh",
        dim=3, network=network, num_refinements=n)

    assert(len(gb_list) == 1 + n), "We should have the original grid, and 3 refinements."
