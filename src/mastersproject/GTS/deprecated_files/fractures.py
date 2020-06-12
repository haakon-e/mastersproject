import porepy as pp

from GTS.fit_plane import convex_hull
from GTS.deprecated_files.old_fit_plane import get_shearzone_planes
from GTS.deprecated_files.temp_data_extraction import ShearzoneInterception


def get_fractures():
    """ Method to get fractures from the provided ISC data"""

    shearzones = get_shearzone_planes()
    fracs = {}

    for s in list(shearzones.keys()):
        fracs[s] = convex_hull(shearzones[s]["proj"])

    return fracs


def get_fractures_manual():
    """ Method to get fractures from manually loaded ISC shearzone intercept data"""
    planes = ShearzoneInterception().interpolate_shearzones()
    fracs = {}

    for s in list(planes.keys()):
        fracs[s] = convex_hull(planes[s]["proj"])

    return fracs


def export_network(fracs, name):
    """ Export fracture network for visualization

    Parameters:
        fracs (dict): Dictionary of fracture vertices.
        name (str): Filename of exported .vtu file.
    """
    if name[-4:] != ".vtu":
        name = name + ".vtu"
    fractures = [pp.Fracture(fracs[key]) for key in list(fracs.keys())]
    network = pp.FractureNetwork3d(fractures)
    network.to_vtk(name)
    return network
