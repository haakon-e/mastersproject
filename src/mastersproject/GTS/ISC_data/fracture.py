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
from pathlib import Path

import pandas as pd
import porepy as pp

import GTS as gts


# TODO: Move convex_plane to the isc class.
def convex_plane(shearzone_names, coord_system='gts', path=None) -> pd.DataFrame:
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
    isc = gts.ISCData(path=path)

    if isinstance(shearzone_names, str):
        shearzone_names = [shearzone_names]
    elif shearzone_names is None:
        shearzone_names = isc.shearzones
    assert (isinstance(shearzone_names, list))

    results = []
    for sz in shearzone_names:
        logging.info(f"Interpolating shearzone {sz} ...")
        point_cloud = isc.get_shearzone(sz=sz, coords=coord_system)
        proj = gts.plane_from_points(point_cloud)  # projection

        convex_vertices = gts.convex_hull(proj)

        frame = pd.DataFrame(data=convex_vertices.T, columns=('x_proj', 'y_proj', 'z_proj'))
        frame['shearzone'] = sz
        results.append(frame)
    df = pd.concat(results, ignore_index=True)
    return df


def fracture_network(shearzone_names, export: bool = False, path=None, **network_kwargs) -> pp.FractureNetwork3d:
    """ Make a fracture network from a selection of shear-zones.

    Parameters:
        shearzone_names : str or list
            Shearzones to make fracture network of.
            if 'None': use all known shearzones.
        export : bool
            Export network to vtk.
        path : pathlib.Path or str
            Path/to/01BasicInputData/
            Alternatively, provide 'windows' or 'linux' default paths for each system.
        network_kwargs : kwargs
            domain : dict
                keys 'xmin', 'xmax', etc. of domain boundaries.

    """
    if isinstance(path, str):
        if path == 'linux':
            _root = Path.cwd()  # should be path/to/mastersproject/src/mastersproject
        elif path == 'windows':
            _root = Path('C:/Users/Haakon/OneDrive/Dokumenter/FORSKNING/mastersproject/src/mastersproject')
        else:
            raise ValueError("Invalid input path type")
        path = _root / 'GTS/01BasicInputData'

    if isinstance(shearzone_names, str):
        shearzone_names = [shearzone_names]
    elif shearzone_names is None:
        shearzone_names = ['S1_1', 'S1_2', 'S1_3', 'S3_1', 'S3_2']
    assert (isinstance(shearzone_names, list))

    convex = convex_plane(shearzone_names, coord_system='gts', path=path)
    fractures = [pp.Fracture(
        convex.loc[convex.shearzone == sz,
                   ('x_proj', 'y_proj', 'z_proj')].to_numpy().T
    ) for sz in shearzone_names]

    network = pp.FractureNetwork3d(fractures)

    # domain_default = {'xmin': -6, 'xmax': 80, 'ymin': 55, 'ymax': 150, 'zmin': 0, 'zmax': 50}
    domain = network_kwargs.get('domain', None)
    if domain is not None:
        network.impose_external_boundary(domain=domain)

    name = network_kwargs.get('name', None)
    if export:
        if name is None:
            name = 'RIGHT_fracture_network.vtu'
        if name[-4:] != '.vtu':
            name = name + '.vtu'
        network.to_vtk(name)
        logging.info("Saving vtk file of fracture network in 3D.")

    return network