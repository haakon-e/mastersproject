""" This is a python implementation of the matlab ISC data manipulation script.

Citation:
"Comprehensive geological dataset for a fractured crystalline rock
volume at the Grimsel Test Site" Krietsch et al. 2018
DOI: 10.3929/ethz-b-000243199

link: https://doi.org/10.3929/ethz-b-000243199

"""
from pathlib import Path
import os
import logging

import numpy as np
import pandas as pd

from GTS.fit_plane import fit_normal_to_points

logger = logging.getLogger(__name__)

class ISCData:
    def __init__(self, path=None):
        """ Initialize the class managing data from the ISC project

        Parameters:
            path : str, pathlib.Path : Optional
                Path/to/01BasicInputData/
                By default, finds the data path,
                 which is known relative to this class.

        """
        # Verify path to data set.
        if path is None:
            path = Path(os.path.abspath(__file__))
            _root = path.parents[2]
            self.data_path = _root / "GTS/01BasicInputData"
        else:
            self.data_path = Path(path)

        logger.info(f"GTS-ISC data located at: {self.data_path}.")
        assert self.data_path.is_dir()

        # ========= CONSTANTS ==========================================================================================
        # Swiss to gts coordinates.
        self.gts_coordinates = np.array((667400, 158800, 1700))

        # Name of boreholes.
        self.borehole_types = {
            "FBS": np.array([1, 2, 3]),
            "SBH": np.array([1, 3, 4]),  # Note the skip of numbering for SBH
            "INJ": np.array([1, 2]),
            "PRP": np.array([1, 2, 3]),
            "GEO": np.array([1, 2, 3, 4]),
        }

        self.boreholes = [
            bh_set + str(bh_num)
            for bh_set in self.borehole_types
            for bh_num in self.borehole_types[bh_set]
        ]

        # Name of shearzones
        self.shearzone_types = {"S1": np.array([1, 2, 3]), "S3": np.array([1, 2])}

        self.shearzones = [
            sz_set + "_" + str(sz_num)
            for sz_set in self.shearzone_types
            for sz_num in self.shearzone_types[sz_set]
        ]

        # ============ LOAD DATA =======================================================================================

        # 1. Step: Load all available data. ============================================================================
        # Load borehole data
        self.borehole_geometry = self._borehole_data()

        # Load borehole structure data
        self.borehole_structures = self._borehole_structure_data()

        # Load tunnel structures (only shear-zones and fractures)
        self.tunnel_structures = self._tunnel_shearzone_data()

        # Load interpolation-ready shear-zone - borehole intersections
        # i.e. 1-1 (-0) mapping between shear-zones and boreholes.
        self.shearzone_borehole_geometry = self._shearzone_borehole_data()

        # 2. Step: All characterized structures ========================================================================
        self.structures = self._full_structure_geometry()

    # ========= PUBLIC CLASS METHODS ===================================================================================

    def get_shearzone(self, sz: str, coords: str = "gts"):
        """ Extract shear-zone coordinates for a given shear-zone

        Coordinates extracted will either be 'swiss' or 'gts'.

        Parameters:
        sz (str): Name of shear-zone (S1_1, S1_2, S1_3, S3_1, S3_2)
        coords (str, Default: 'gts'):
            Get coordinates in 'gts' or 'swiss'.

        Returns
        np.ndarray (3, n): Coordinates of shearzone intersections.
        """
        df = self.structures
        assert sz in self.shearzones, f"unknown shear-zone {sz}."
        assert coords in ["swiss", "gts"], f"unknown coordinate system {coords}."
        sz = df.loc[df.shearzone == sz, (f"x_{coords}", f"y_{coords}", f"z_{coords}")]
        return sz.to_numpy().T

    def structures_depth(
        self,
        borehole: str,
        depth: np.ndarray,
        structure=None,
        shearzone=None,
        coords="gts",
    ):
        """ Get structures in a borehole at depth

        For a given borehole, and a given depth (or depth interval),
        get all structures - or a subset of structures, or specific shearzones.

        Parameters:
        borehole (str): name of borehole (INJ1, INJ2, ...)
        depth (np.array): Depth interval in borehole.
        structure (str or list, Optiona): Filter by certain structures
            (Fracture, Minor ductile Shear-zone, S1 Shear-zone, Quartz, ...)
        shearzone (str or list, Optional): Filter by certain shear-zones
            (S1_1, S1_2, ...)
        coords (str, optional): which coordinate system to return

        Returns:
        pd.DataFrame: Filtered dataframe

        """
        assert (depth.shape[0] == 2) and (
            depth[0] <= depth[1]
        ), "Depth must be given as an interval."
        assert coords in ["swiss", "gts"], f"unknown coordinate system {coords}."
        df = self.structures

        # Structure mask
        if isinstance(structure, str):
            structure = [structure]
        if structure is not None:
            _mask_struc = df.type.isin(structure)
        else:
            _mask_struc = np.ones(df.shape[0], dtype=bool)

        # Shear-zone mask
        if isinstance(shearzone, str):
            shearzone = [shearzone]
        if shearzone is not None:
            _mask_sz = df.shearzone.isin(shearzone)
        else:
            _mask_sz = np.ones(df.shape[0], dtype=bool)

        # Borehole mask
        _mask_bh = df.borehole == borehole

        # Depth mask
        _mask_depth = (depth[0] <= df.depth) & (df.depth <= depth[1])

        # Full mask #
        _mask = _mask_bh & _mask_depth & _mask_struc & _mask_sz

        # Filter DataFrame #
        _bh = df.loc[
            _mask,
            (
                "depth",
                "azimuth_struc",
                "dip",
                "aperture",
                "type",
                "borehole",
                "shearzone",
                f"x_{coords}",
                f"y_{coords}",
                f"z_{coords}",
            ),
        ]
        return _bh

    def borehole_plane_intersection(self):
        """ Compute new intersections of boreholes and shear-zones.

        There will be new intersections due to regression over old
        intersections to produce shear-zone planes.

        Returns
        df : pd.DataFrame
            DataFrame of shear-zone -- borehole intersections.
            Only intersections found in self.shearzone_borehole_geometry are considered.
            Each row represents an intersection and contains the following data:
                * plane normal and centroid.
                * Original depth of intersection ('old_depth')
                * borehole root coordinates and direction vector
                * depth of intersection ('depth')
                * gts coordinates of intersection ('x_sz', 'y_sz', 'z_sz')

        """

        # 1. Step: Compute direction vectors to each borehole ==========================================================
        borehole_data = self.borehole_geometry.copy()
        borehole_data["depth"] = 0
        borehole_to_global_coords(
            data=borehole_data,
            x="x",
            y="y",
            z="z",
            depth="depth",
            upward_gradient="upward_gradient",
            azimuth="azimuth",
        )

        # Extract relevant columns from borehole data
        _mask = ["borehole", "x_gts", "y_gts", "z_gts", "_trig_x", "_trig_y", "_trig_z"]
        bh_data = borehole_data[_mask]

        mapper = {
            "x_gts": "x_bh",
            "y_gts": "y_bh",
            "z_gts": "z_bh",
            "_trig_x": "r_x",
            "_trig_y": "r_y",
            "_trig_z": "r_z",
        }
        bh_data = bh_data.rename(columns=mapper)

        # 2. Step: Calculate shear-zone unit normals and centroids =====================================================
        sz = self.planes()

        # 3. Step: Extract shear-zone borehole geometry ================================================================
        # i.e. only the shear-zones used for computing shear-zone planes.
        sz_bh = self.shearzone_borehole_geometry.copy()
        sz_bh = sz_bh[sz_bh.depth.notna()]
        sz_bh = sz_bh.rename(columns={"depth": "old_depth"})

        # 4. Step: Merge the collected data ============================================================================
        df = sz.merge(sz_bh, on="shearzone").merge(bh_data, on="borehole")

        # 5. Step: Calculate new shear-zone borehole intersections. ====================================================
        # Quantities
        n_vec = ["n_x", "n_y", "n_z"]
        r_vec = ["r_x", "r_y", "r_z"]
        bh_coords = ["x_bh", "y_bh", "z_bh"]
        sz_coords = ["x_c", "y_c", "z_c"]

        # Depth calculation
        df["depth"] = (
            (df[sz_coords].values - df[bh_coords].values) * df[n_vec].values
        ).sum(axis=1) / (df[n_vec].values * df[r_vec].values).sum(axis=1)

        # Calculate global coordinates
        df.loc[:, "x_sz"] = df.x_bh + (df.depth * df.r_x)
        df.loc[:, "y_sz"] = df.y_bh + (df.depth * df.r_y)
        df.loc[:, "z_sz"] = df.z_bh + (df.depth * df.r_z)

        return df

    def planes(self):
        """ Compute plane of best fit from point cloud of each shear-zone.

        Returns
        df : pd.DataFrame
            Normal vector and centroid of each shear-zone.
        """

        results = []
        for sz in self.shearzones:
            point_cloud = self.get_shearzone(sz=sz, coords="gts")
            n_pts = point_cloud.shape[1]
            centroid = np.sum(point_cloud, axis=1) / n_pts
            normal = fit_normal_to_points(point_cloud)

            data = np.atleast_2d(np.hstack((centroid, normal)))
            columns = ("x_c", "y_c", "z_c", "n_x", "n_y", "n_z")
            frame = pd.DataFrame(data=data, columns=columns)
            frame["shearzone"] = sz
            results.append(frame)

        df = pd.concat(results, ignore_index=True)
        return df

    # ======= PRIVATE CLASS UTILITY METHODS ============================================================================

    def _borehole_data(self):
        """ Fetch data with borehole coordinates

        Assumes existence of the following attributes:
            gts_coordinates
            borehole_types

        Returns
        pd.DataFrame: data on location and orientation of each borehole INJ1, FBS1, etc.
            Columns:
        """
        file_loc = self.data_path / "02_Boreholes"
        columns = ["x", "y", "z", "length", "diameter", "azimuth", "upward_gradient"]

        data = []
        for parent in self.borehole_types:
            path = file_loc / (parent + ".txt")
            frame = pd.read_csv(path, sep=None, names=columns, engine="python")
            borehole_name = np.array(
                [parent + str(id) for id in self.borehole_types[parent]]
            )
            frame["borehole"] = borehole_name
            data.append(frame)

        df = pd.concat(data, ignore_index=True)
        return df

    def _borehole_structure_data(self):
        """ Data on geological structures' intersections to boreholes.

        Returns
        pd.DataFrame: data on depth, orientation, and thickness of intersecting structures to each borehole.

        """
        file_loc = self.data_path / "03_GeologicalMapping" / "02_BoreholeIntersections"
        columns = ["depth", "azimuth", "dip", "aperture", "type"]

        data = []
        for borehole in self.boreholes:
            path = file_loc / (borehole + "_structures.txt")
            frame = pd.read_csv(
                path, sep=None, names=columns, skiprows=2, engine="python"
            )
            frame["borehole"] = borehole
            data.append(frame)
        df = pd.concat(data, ignore_index=True)
        return df

    def _tunnel_shearzone_data(self):
        """Data on tunnel intersections with shearzones

        """
        file_loc = self.data_path / "03_GeologicalMapping" / "01_TunnelIntersections"
        columns = ["x", "y", "z", "true_dip_direction", "dip", "tunnel", "shearzone"]

        path = file_loc / "Tunnel_intersections.txt"
        df = pd.read_csv(path, sep=None, names=columns, engine="python")
        df["shearzone"] = df["shearzone"].apply(rename_sz)
        df = df.rename(
            columns={"true_dip_direction": "azimuth_struc", "tunnel": "borehole",}
        )
        return df

    def _shearzone_borehole_data(self):
        """ Import data on shearzone intersections with boreholes

        This data is 'opposite' of the data from _borehole_structure_data(self).
        i.e. We extract for every shearzone (S1_1, S1_2, ...) their intersections with boreholes.

        Note: The borehole_shearzone_data contains multiple intersections of shearzones to boreholes,
        whereas the _shearzone_borehole_data only contains one (a subset of the former).
        Why this is so, I do not know.
        TODO: Find out why we have single intersections with this data, but not with the other data.
            (see borehole_shearzone_data(self)).

        """
        file_loc = self.data_path / "06_ShearzoneInterpolation"
        columns = ["borehole", "depth"]

        data = []
        for parent in self.shearzone_types:
            for sz_num in self.shearzone_types[parent]:
                sz_name = parent + "_" + str(sz_num)  # e.g. 'S1_1'
                path = file_loc / (sz_name + ".txt")
                frame = pd.read_csv(
                    path, sep=None, names=columns, skiprows=1, engine="python"
                )
                frame["shearzone"] = sz_name
                data.append(frame)
        df = pd.concat(data, ignore_index=True)
        return df

    def _characterize_shearzones(self):
        """ Classify all structures as specific shear-zones (e.g. S1_1) or none

        Helper method to self._full_structure_geometry()

        Load all borehole structures, and classify each as not a shear-zone,
        or as Sx_y. This method combines structure-borehole data with
        shear-zone -- borehole data.

        Note:
            - S1_2 intersects GEO3 is of type 'Minor ductile Shear-zone'
            - One shearzone is inconsistent (to +- 0.01) across the two datasets.
        These two notes has significantly complicated the code.

        Returns:
        pd.DataFrame: Characterized structures.

        """

        # 1. Step: Import shear-zone -- borehole data.
        shearzone_borehole = self.shearzone_borehole_geometry

        # Remove rows with no intersection.
        shearzone_borehole = shearzone_borehole[shearzone_borehole["depth"].notna()]

        # 2. Step: Import borehole - structure data.
        structures = self.borehole_structures.merge(
            self.borehole_geometry,
            how="outer",
            on="borehole",
            suffixes=("_struc", "_bh"),
            validate="m:1",
        )

        # 3. Step: Left merge data structures with simulation-shearzones.
        # Notes:
        # - Absolute tolerance = 0.01
        # - exact match on column = 'borehole'
        _merge = pd.merge_asof(
            structures.sort_values("depth"),
            shearzone_borehole.sort_values("depth"),
            by="borehole",
            on="depth",
            tolerance=0.01,
            direction="nearest",
        )

        # The above merge includes some nearby 'fractures', etc.
        # All shearzones are classified as 'S1 Shear-zone' or 'S3 Shear-zone'
        # except once: S1_2 intersects GEO3 is of type 'Minor ductile Shear-zone'.
        # With this set of 'type' filters, no spurious shearzone intersections are located.

        # 4. Step: Complete classification of shear-zones.
        _mask_nna = _merge.shearzone.notna()
        _mask_szset = _merge.type.isin(
            ["S1 Shear-zone", "S3 Shear-zone", "Minor ductile Shear-zone"]
        )
        _shearzones = _merge[_mask_nna & _mask_szset]

        # 5. Step: Get indices of non-shearzones:
        _index_other = set(_merge.index) - set(_shearzones.index)

        # 6. Set non-shearzones to nan
        _merge.loc[np.array(list(_index_other)), "shearzone"] = np.nan

        return _merge

    def _full_structure_geometry(self):
        """ Compute geometry of all structures in ISC.

        Geometry of all structures in boreholes and shear-zones in tunnels are located,
        and global coordinates computed.

        Returns:
            pd.DataFrame: Full data set - with all characterized structures in local coordinates.

        """
        # Characterized borehole structures
        borehole_structures = self._characterize_shearzones()

        # Tunnel shearzone data
        tunnel_structures = self.tunnel_structures

        structures = pd.concat(
            [borehole_structures, tunnel_structures], ignore_index=True, sort=False
        )

        # Fill NaN-values in all columns to 0 except in column 'shearzone', for which we do nothing.
        structures = structures.fillna(
            value={**{s: 0 for s in borehole_structures}, **{"shearzone": np.nan}}
        )

        mapping = {
            "x": "x",
            "y": "y",
            "z": "z",
            "depth": "depth",
            "upward_gradient": "upward_gradient",
            "azimuth": "azimuth_bh",
        }
        borehole_to_global_coords(structures, **mapping)

        return structures


# UTILITY METHODS ======================================================================================================


def borehole_to_global_coords(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    z: str,
    depth: str,
    upward_gradient: str,
    azimuth: str,
):
    """ Convert coordinates in a borehole to global coordinates

    For all rows in a DataFrame, convert some (x,y,z) coordinates to global
    coordinates, localized to Swiss and/or GTS.

    Parameters:
    data (pd.DataFrame)
    x, y, z, depth, upward_gradient, azimuth (str):
        Column names for the respective quantities
    """

    # Compute angle scalers
    rad = np.pi / 180
    data.loc[:, "_trig_x"] = (data[upward_gradient] * rad).apply(np.cos) * (
        data[azimuth] * rad
    ).apply(np.sin)

    data.loc[:, "_trig_y"] = (data[upward_gradient] * rad).apply(np.cos) * (
        data[azimuth] * rad
    ).apply(np.cos)

    data.loc[:, "_trig_z"] = (data[upward_gradient] * rad).apply(np.sin)

    # Swiss coordinates
    data.loc[:, "x_swiss"] = data[x] + (data[depth] * data["_trig_x"])
    data.loc[:, "y_swiss"] = data[y] + (data[depth] * data["_trig_y"])
    data.loc[:, "z_swiss"] = data[z] + (data[depth] * data["_trig_z"])

    # TODO: Use attribute self.gts_coordinates instead.
    #   Also, remove _swiss coordinates, as they are not used.
    # GTS coordinates
    data[["x_gts", "y_gts", "z_gts"]] = data[["x_swiss", "y_swiss", "z_swiss"]].apply(
        swiss_to_gts, axis=1, raw=True, result_type="expand"
    )


def swiss_to_gts(v):
    """ Convert from swiss coordinates to gts coordinates

    GTS coordinates are: (x,y,z) = (667400, 158800, 1700)

    Parameters:
    v (np.array (3,)): Coordinate array

    """
    return v - np.array([667400, 158800, 1700])


def rename_sz(sz):
    """ Rename shearzone on form '12' to 'S1_2'. """
    sz = str(sz)
    sz_set = sz[0]
    num = sz[1]
    return f"S{sz_set}_{num}"
