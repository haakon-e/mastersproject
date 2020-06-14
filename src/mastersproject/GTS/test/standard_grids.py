from pathlib import Path

import numpy as np

import porepy as pp


def two_intersecting_blocking_fractures(folder_name: Path) -> pp.GridBucket:
    """ Domain with fully blocking fracture """
    folder_name = Path(folder_name)
    # fmt: off
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
    # fmt: on
    frac2 = pp.Fracture(frac_pts2)

    frac_network = pp.FractureNetwork3d([frac1, frac2], domain)
    mesh_args = {"mesh_size_frac": 30, "mesh_size_min": 20, "mesh_size_bound": 60}

    file_name = str(folder_name / "gmsh_frac_file")
    gb = frac_network.mesh(mesh_args, file_name=file_name)
    return gb