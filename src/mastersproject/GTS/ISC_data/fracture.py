"""
Methods for assembling fractures from isc data and construct a discrete fracture network.

Public methods:
convex_plane(shearzone_names, coord_system='gts', path=None) -> pd.DataFrame:
    - Wrapper to construct convex hulls for the different shear-zones in isc.
        Gets data by calling gts.ISCData() directly.
fracture_network(shearzone_names, export: bool = False, path=None, **network_kwargs) -> pp.FractureNetwork3d:
    - Construct a 3D fracture network from the isc data.

"""

import logging
import os
from pathlib import Path

import pandas as pd
import porepy as pp

from GTS.ISC_data import ISCData
from GTS.fit_plane import plane_from_points, convex_hull

logger = logging.getLogger(__name__)


# TODO: Move convex_plane to the isc class.
def convex_plane(shearzone_names, coord_system="gts", path=None) -> pd.DataFrame:
    """ Compute vertices for the convex polygon of the projected point cloud
    to the plane of best fit for each shear-zone is shearzone_names.

    Data imported from gts.ISCData()

    Parameters:
    shearzone_names : str or list
        Names of shear-zones to construct convex planes of
        Input values, e.g.: 'S1_1', 'S1_2', 'S1_2', or a list of them.
    coord_system : str, Default: 'gts'
        Name of coordinate system to use
        Input values: Either 'gts' or 'swiss'
    path : pathlib.Path
        Path/to/01BasicInputData/

    Returns:
    convex_shearzones : pd.DataFrame
        Convex polygon of projected points to best fit plane

    """
    isc = ISCData(path=path)

    if isinstance(shearzone_names, str):
        shearzone_names = [shearzone_names]
    elif shearzone_names is None:
        shearzone_names = isc.shearzones
    assert isinstance(shearzone_names, list)

    results = []
    for sz in shearzone_names:
        logger.info(f"Interpolating shearzone {sz} ...")
        point_cloud = isc.get_shearzone(sz=sz, coords=coord_system)
        proj = plane_from_points(point_cloud)  # projection

        convex_vertices = convex_hull(proj)

        frame = pd.DataFrame(
            data=convex_vertices.T, columns=("x_proj", "y_proj", "z_proj")
        )
        frame["shearzone"] = sz
        results.append(frame)
    df = pd.concat(results, ignore_index=True)
    return df


def fracture_network(
    shearzone_names, export_vtk: bool = False, path=None, **network_kwargs
) -> pp.FractureNetwork3d:
    """ Make a fracture network from a selection of shear-zones.

    Parameters:
        shearzone_names : str or list
            Shearzones to make fracture network of.
            if 'None': Mesh only the 3d domain.
        export_vtk : bool
            Export network to vtk.
        path : pathlib.Path or str
            Path/to/01BasicInputData/
        network_kwargs : kwargs
            domain : dict
                keys 'xmin', 'xmax', etc. of domain boundaries.
            length_scale : float
                Domain scaling.
                Divide all nodes of shear zones by 'length_scale'.
            network_path : str
                path/to/fracture_network.vtu

    """
    if path is None:
        path = Path(os.path.abspath(__file__))
        _root = path.parents[2]
        data_path = _root / "GTS/01BasicInputData"
    else:
        data_path = Path(path)

    if isinstance(shearzone_names, str):
        shearzone_names = [shearzone_names]

    if shearzone_names is None:
        # This will mesh only a 3d domain.
        fractures = None
    else:
        convex = convex_plane(shearzone_names, coord_system="gts", path=data_path)

        # Domain scaling
        length_scale = network_kwargs.get("length_scale", 1)
        convex.loc[:, ("x_proj", "y_proj", "z_proj")] /= length_scale

        # Create each fracture
        # Saved to list of same order as the input list shearzone_names (ensures correct naming of shearzones later on)
        fractures = [
            pp.Fracture(
                convex.loc[convex.shearzone == sz, ("x_proj", "y_proj", "z_proj")]
                .to_numpy()
                .T
            )
            for sz in shearzone_names
        ]

    network = pp.FractureNetwork3d(fractures)

    # domain_default = {'xmin': -6, 'xmax': 80, 'ymin': 55, 'ymax': 150, 'zmin': 0, 'zmax': 50}
    domain = network_kwargs.get("domain", None)
    if domain is not None:
        network.impose_external_boundary(domain=domain)

    network_path = network_kwargs.get("network_path", None)
    if export_vtk:
        if network_path is None:
            network_path = "fracture_network.vtu"
        if network_path[-4:] != ".vtu":
            network_path = network_path + ".vtu"
        network.to_vtk(network_path)
        logger.info("Saving vtk file of fracture network in 3D.")

    return network
