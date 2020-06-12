""" This is a replicate of the geological_model_visualization.m file from"""
from pathlib import Path
import os

import numpy as np


class GeologicalModel:
    def __init__(self):
        """ Define the global constants for the geological dataset.

        These include:
        GTS_coordinates (np.array, (3)):
            Origin of coordinate system. This system will be our new origin when coordinates are imported.

        """
        # Get correct path (we guess):
        wd = Path.cwd()
        if (wd / "01BasicInputData").is_dir():
            self.data_path = wd / "01BasicInputData"
        else:
            self.data_path = wd / "GTS/01BasicInputData"

        self.GTS_coordinates = np.array((667400, 158800, 1700))

        # 1. Step: Define Tunnel =======================================================================================
        # TODO: Consider importing tunnels at a late time.

        # 2. Step: Drill Borehole ======================================================================================
        # Import all boreholes in order to use their coordinates for further calculations.

        # Name of boreholes.
        self.BH_import = ["FBS", "SBH", "INJ", "PRP", "GEO"]

        # Import Borehole Data
        self.BH_coordinates = self.drill_boreholes()

        # 3. Step: Import structures from optv logs ====================================================================
        # Import geological structures from OPTV logs.
        # The output contains data for each borehole, and in addition,
        # filtered data for each of fractures, S1 shear zones and S3 shear zones.
        self.optv_logs = self.import_optv_structures()

        # 4. Step: Import data from geodetic mapping along tunnel walls ================================================
        # Import structures intersecting with the tunnel walls.
        # Information provides includes: (x,y,z)-coordinates, dip-direction/dip, tunnel name.
        self.sz_tunnel = self.tunnel_intersections()

        # 5. Step: Calculate locations of geological features ==========================================================
        # TODO: Implement step 5.
        # Calculate true locations of the geological features fracture, S1 shear zones, S3 shear zones.
        # By default:
        #   Each fracture is represented by a disc of radius 1 m.
        #   Each shear zone is represented by a disc of radius 2 m.

        # 6. Step: First linear interpolation between geological observations ==========================================
        # Define a linear interpolation between mapped shear zone coordinates that belong to the same set.
        # Note: True local orientations are disregarded.
        self.sz = self.shearzones_patches()

    def drill_boreholes(self):
        """ Fetch the borehole coordinates.

        Assumes the following attributes exists:
            GTS_coordinates: To adjust coordinate system origin.
            BH_import: For file names

        """
        bh_coordinates = {}
        path = self.data_path / "02_Boreholes"
        for bh in self.BH_import:
            name = path / (bh + ".txt")
            bh_coordinates[bh] = np.genfromtxt(name)  # Default delimiter is spaces.
            if np.all(np.isnan(bh_coordinates[bh])):  # If delimiter isn't spaces ...
                bh_coordinates[bh] = np.genfromtxt(
                    name, delimiter=","
                )  # ... then it is commas.

            bh_coordinates[bh][:, :3] = np.subtract(
                bh_coordinates[bh][:, :3], self.GTS_coordinates
            )

        return bh_coordinates

    def import_optv_log(self, bh):
        """ Import OPTV logs.

        Mirrors the 'importOPTVlog.m' function in the
        matlab script.

        """

        path = self.data_path / "03_GeologicalMapping/02_BoreholeIntersections"
        name = path / (bh + "_structures.txt")
        dtype = [float, float, float, float, "U24"]
        names = ["Depth", "Azimuth", "Dip", "Aperture", "Type"]
        structures = np.genfromtxt(
            name, dtype=dtype, names=names, delimiter="\t", skip_header=2
        )
        return structures

    def import_optv_structures(self):
        """ Import location of structures recorded in OTPV logs.

        Mirrors 'importOPTVstructures.m' in the matlab script.

        """
        boreholenames = [
            "FBS1",
            "FBS2",
            "FBS3",
            "SBH1",
            "SBH3",
            "SBH4",
            "INJ1",
            "INJ2",
            "PRP1",
            "PRP2",
            "PRP3",
            "GEO1",
            "GEO2",
            "GEO3",
            "GEO4",
        ]
        optv_logs = {}
        structure_types = ["Fracture", "S1 Shear-zone", "S3 Shear-zone"]

        for bh in boreholenames:
            structures = self.import_optv_log(bh)
            optv_logs[bh] = structures

            # Filter by structure type
            for st in structure_types:
                lst = [l for l in structures if l[4] == st]
                if optv_logs.get(st, None) is None:
                    optv_logs[st] = lst
                else:
                    optv_logs[st].extend(lst)

        return optv_logs

    def tunnel_intersections(self):
        """ Compute tunnel intersections.

        Mirrors 'Tunnel_intersections.m' in the matlab script.

        """
        rel_path = (
            "03_GeologicalMapping/01_TunnelIntersections/Tunnel_intersections.txt"
        )
        path = self.data_path / rel_path
        delimiter = "\t"
        dtype = [float, float, float, float, float, "O", int]
        names = ["x", "y", "z", "dip-direction", "dip", "tunnel", "shear zone set"]
        structures = np.genfromtxt(path, dtype=dtype, names=names, delimiter=delimiter)

        # Convert to local coordinates
        for st in structures:
            for i in range(0, 3):
                st[i] = st[i] - self.GTS_coordinates[i]

        return structures

    def shearzones_patches(self):
        """ Compute linear interpolation of shear zones in both sets.

        Use data stored in attributes: 'BH_coordinates' and 'sz_tunnel' to calculate
        shear zone interpolations.

        Mirrors 'S1_shearzones_patches.m' and 'S3_shearzones_patches.m' in the matlab script.

        """
        path = self.data_path / "06_ShearzoneInterpolation"
        shear_zones = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]  # 'S3_1', 'S3_2'

        sz = {}  # Dictionary for shear zone coordinates.
        """ When done, sz will have the following structure:
        sz = {
            'S1_1': {
                    'Borehole': ['INJ1', 'INJ2', ..., "TUNNEL_NAME"]  <-- Note last entry
                    'Depth': np.array([3.4, 2.3, ..., np.nan]) <--- Last entry is tunnel depth (np.nan)
                    'x': np.array([5.4, 2.0, ..., 56.7]), <--- Last entry is tunnel coordinate.
                    'y': np.array([...]),
                    'z': np.array([...]),
                    
                     }
        }
        """

        delimiter = "\t"
        dtype = ["U4", float]
        for shear_zone in shear_zones:
            fname = path / (shear_zone + ".txt")
            _result = np.genfromtxt(fname, dtype=dtype, delimiter=delimiter, names=True)

            # Get Borehole names and depth (in borehole) the shear zone 'file'.
            _bh = np.array([r[0] for r in _result])
            _dp = np.array([r[1] for r in _result])
            sz[shear_zone] = {"Borehole": _bh, "Depth": _dp}

            # Aliases
            bh_coords = self.BH_coordinates

            # Initialize coordinates
            xyz = ["x", "y", "z"]
            for d in xyz:
                sz[shear_zone][d] = np.zeros(len(_bh))

            # Calculate absolute coordinates for borehole-shearzone intersections.
            for bh_idx, bh in enumerate(sz[shear_zone]["Borehole"]):

                # If bh doesn't intersect with the shear zone, set coords to np.nan.
                if np.isnan(sz[shear_zone]["Depth"][bh_idx]):
                    for d_i, d in enumerate(
                        xyz
                    ):  # Assumes 'x, y, z' are first 3 (ordered) entries.
                        sz[shear_zone][d][bh_idx] = np.nan
                    continue

                # Otherwise, compute the global coordinate of the shear zone.
                bh_name = bh[:3]  # Get the 3 letters from e.g. 'INJ1'
                num = int(bh[-1])  # Get the 1-digit number from e.g. 'INJ1'

                # Fetch the borehole coordinates (and more) for a given borehole
                _idx = num - 1  # - 1 because of 0-indexing.
                # TODO: Fix Hard coded case for SBH3 and SBH4, whose well number
                #   doesn't correspond with borehole index in 02_Boreholes > SBH.txt.
                # ==============================
                if bh == "SBH3" or bh == "SBH4":
                    _idx = _idx - 1
                # ==============================
                borehole = bh_coords[bh_name][_idx]
                rad = np.pi / 180

                # Get coordinates ============
                # (e.g.: borehole[0] for x-coordinate of e.g. INJ1.
                # Columns in borehole:
                # x | y | z | length | diameter | azimuth | upward gradient
                for d_i, d in enumerate(
                    xyz
                ):  # Assumes 'x, y, z' are first 3 (ordered) entries.
                    sz[shear_zone][d][bh_idx] = borehole[d_i] + np.sin(
                        borehole[5] * rad
                    ) * np.cos(borehole[6] * rad)

        # Assign tunnel intersections to Shear zones
        # Columns in sz_tunnel:
        # 'x', 'y', 'z', 'dip-direction', 'dip', 'tunnel', 'shear zone set'
        for j, tunnel in enumerate(self.sz_tunnel):
            sz_type = str(tunnel[6])[0]  # Get first integer in e.g. 12
            sz_num = str(tunnel[6])[1]  # Get second integer in e.g. 12
            sz_id = f"S{sz_type}_{sz_num}"  # e.g. 'S1_2'

            sz[sz_id]["Borehole"] = np.append(
                sz[sz_id]["Borehole"], tunnel[5]
            )  # Tunnel name
            sz[sz_id]["Depth"] = np.append(sz[sz_id]["Depth"], np.nan)  # Tunnel name

            xyz = ["x", "y", "z"]
            for d_i, d in enumerate(xyz):
                sz[sz_id][d] = np.append(sz[sz_id][d], tunnel[d_i])

        return sz

    def export_intersections(self):
        """ Export intersection points for each shear zone"""
        sz = self.sz
        intxs = {}

        for key in list(sz.keys()):
            x = np.vstack((sz[key]["x"], sz[key]["y"], sz[key]["z"]))

            # Assert that one coordinate dimension is nan exactly when all dimensions are.
            cond = np.isnan(x)
            assert np.all(np.equal(np.all(cond, 0), np.any(cond, 0)))

            x = x[:, np.logical_not(cond[0])]  # fetch not nan points.
            intxs[key] = np.copy(x)

        return intxs
