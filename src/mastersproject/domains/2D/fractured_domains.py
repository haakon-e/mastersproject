import numpy as np

import porepy as pp


def fractured_domain_2d(domain, mesh_args, fracs_coords, fracs):
    """ Create a fractured 2d domain.

    Parameters:
    domain (dict): Dictionary specifying domain boundaries.
        Assumes containing 'xmin', 'xmax', 'ymin', 'ymax'.
    mesh_args (dict): Dictorionary specifying meshing arguments
        Must contain keys:
            'mesh_size_frac': Mesh size in fractures.
            'mesh_size_min': Minimum mesh size.
        Optional keys:
            'mesh_size_bound': Mesh size at boundaries.
    fracs_coords (np.ndarray 2 x n): Coordinates of fractures.
    fracs (np.ndarray 2 x num_fracs): Endpoints of fractures.
        Defines a mapping to fracs_coords.

    """
    network_2d = pp.FractureNetwork2d(fracs_coords, fracs, domain)
    gb = network_2d.mesh(mesh_args)
    return gb


def two_fractures(mesh_args=None):
    """ Create a domain with two fractures,
        one diagonal to the other.
    """
    if not mesh_args:
        mesh_args = {
            "mesh_size_frac": 0.2,
            "mesh_size_min": 0.1,
            "mesh_size_bound": 0.4,
        }

    domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3}
    fracs_coords = np.array([[-1, 2.5, 0, 2.5], [0, 0, 0, 2]])
    fracs = np.array([[0, 2], [1, 3]])

    gb = fractured_domain_2d(domain, mesh_args, fracs_coords, fracs)
    return gb
