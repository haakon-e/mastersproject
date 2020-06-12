from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GTS.deprecated_files.old_fit_plane import FitPlane


class ShearzoneInterception:
    def __init__(self):
        """ Load all shearzone interceptions"""
        self.sz = get_shearzones()

    def interpolate_shearzones(self):
        sz = self.sz

        planes = {}
        for key in list(sz.keys()):
            s = sz[key]
            pts = s[["x", "y", "z"]].to_numpy().T  # Order dim x n

            fp = FitPlane(pts)
            planes[key] = {}
            planes[key]["normal"] = fp.n
            planes[key]["proj"] = fp.proj

        return planes


def get_shearzones():
    """ The shearzone intercepts manually copied from computed results in the matlab code

    Returns:
        dictionary with keys S1_1, S1_2, etc. each containing:
            pd.DataFrame:
                Columns: Borehole, Depth, x, y, z
                Note: x, y, z are in GTS coordinates.

    """

    s11 = pd.DataFrame(
        [
            [
                "INJ1",
                34.9200000000000,
                43.7896710260604,
                106.944298581165,
                13.5331970965852,
            ],
            [
                "INJ2",
                30.9900000000000,
                56.3567185441727,
                110.066920092377,
                11.4901140651349,
            ],
            [
                "FBS1",
                31.0900000000000,
                48.4437351192416,
                108.197239759454,
                16.3436397456107,
            ],
            ["FBS2", np.nan, np.nan, np.nan, np.nan],
            [
                "FBS3",
                19.4200000000000,
                61.4076444299798,
                114.369946943679,
                20.4866182767554,
            ],
            [
                "PRP1",
                33.0400000000000,
                51.2328752347784,
                107.516568323472,
                9.68848744003483,
            ],
            [
                "PRP2",
                30.1000000000000,
                50.7892481188025,
                109.424407636357,
                17.1738855601783,
            ],
            [
                "PRP3",
                25.5800000000000,
                58.5959588772716,
                113.672963230578,
                22.2894248646728,
            ],
            ["GEO1", np.nan, np.nan, np.nan, np.nan],
            ["GEO2", np.nan, np.nan, np.nan, np.nan],
            [
                "GEO3",
                18.6300000000000,
                58.4088484704877,
                111.999263472691,
                18.6148103711778,
            ],
            [
                "GEO4",
                19.5200000000000,
                54.6352202904385,
                112.008451293265,
                21.6404975998854,
            ],
            ["SBH1", np.nan, np.nan, np.nan, np.nan],
            ["SBH3", np.nan, np.nan, np.nan, np.nan],
            ["SBH4", np.nan, np.nan, np.nan, np.nan],
            ["AU", np.nan, 72.6250000000000, 125.320999999996, 33.4359999999999],
            ["VE", np.nan, 9.73499999998603, 88.3599999999860, 35.4190000000001],
        ],
        columns=["Borehole", "Depth", "x", "y", "z"],
    )

    s12 = pd.DataFrame(
        [
            [
                "INJ1",
                39.1100000000000,
                41.0969686090808,
                109.169521979076,
                11.2193616393885,
            ],
            [
                "INJ2",
                34.3500000000000,
                55.2258458209840,
                112.219089550601,
                9.17086989794717,
            ],
            [
                "FBS1",
                36.0500000000000,
                45.5752206834566,
                111.278731853595,
                13.7211158838619,
            ],
            ["FBS2", np.nan, np.nan, np.nan, np.nan],
            ["FBS3", np.nan, np.nan, np.nan, np.nan],
            [
                "PRP1",
                43.0400000000000,
                46.6090481266577,
                113.027030891109,
                2.74190373544486,
            ],
            [
                "PRP2",
                34.4300000000000,
                48.5277578980176,
                112.317942023914,
                14.8799760743170,
            ],
            [
                "PRP3",
                31.1400000000000,
                56.4671524408976,
                118.240284401884,
                19.9396673293945,
            ],
            ["GEO1", np.nan, np.nan, np.nan, np.nan],
            ["GEO2", np.nan, np.nan, np.nan, np.nan],
            [
                "GEO3",
                26.4900000000000,
                53.1291259250457,
                111.995577530412,
                12.7920862443639,
            ],
            [
                "GEO4",
                28.0500000000000,
                47.5818980095895,
                112.012144404513,
                16.8434640203271,
            ],
            ["SBH1", np.nan, np.nan, np.nan, np.nan],
            ["SBH3", np.nan, np.nan, np.nan, np.nan],
            ["SBH4", np.nan, np.nan, np.nan, np.nan],
            ["AU", np.nan, 74.5649999999441, 135.310999999987, 33.8580000000000],
            ["VE", np.nan, 10.9170000000158, 95.6169999999984, 34.4310000000000],
        ],
        columns=["Borehole", "Depth", "x", "y", "z"],
    )

    s13 = pd.DataFrame(
        [
            [
                "INJ1",
                43.2000000000000,
                38.4685311662535,
                111.341637419997,
                8.96074898546618,
            ],
            [
                "INJ2",
                38.6800000000000,
                53.7685009128271,
                114.992569834563,
                6.18208202773205,
            ],
            [
                "FBS1",
                42.4400000000000,
                41.8796950292899,
                115.248637999070,
                10.3425014732621,
            ],
            ["FBS2", np.nan, np.nan, np.nan, np.nan],
            ["FBS3", np.nan, np.nan, np.nan, np.nan],
            [
                "PRP1",
                46.3400000000000,
                45.0831851809778,
                114.845483538429,
                0.449531112930167,
            ],
            [
                "PRP2",
                41.8500000000000,
                44.6524097598597,
                117.276377394737,
                10.9490734449656,
            ],
            ["PRP3", np.nan, np.nan, np.nan, np.nan],
            ["GEO1", np.nan, np.nan, np.nan, np.nan],
            ["GEO2", np.nan, np.nan, np.nan, np.nan],
            ["GEO3", np.nan, np.nan, np.nan, np.nan],
            [
                "GEO4",
                35.6200000000000,
                41.3223845669135,
                112.015421878387,
                12.5863076079876,
            ],
            ["SBH1", np.nan, np.nan, np.nan, np.nan],
            ["SBH3", np.nan, np.nan, np.nan, np.nan],
            ["SBH4", np.nan, np.nan, np.nan, np.nan],
            ["AU", np.nan, 74.8390000000363, 143.317000000010, 33.6110000000001],
            ["VE", np.nan, 18.5600000000559, 107.673999999999, 34.9159999999999],
        ],
        columns=["Borehole", "Depth", "x", "y", "z"],
    )

    s31 = pd.DataFrame(
        [
            [
                "INJ1",
                28.2000000000000,
                48.1082772890921,
                103.375443871388,
                17.2441694766238,
            ],
            [
                "INJ2",
                20.2900000000000,
                59.9580096567081,
                103.213285210531,
                18.8758023356434,
            ],
            [
                "FBS1",
                23.3400000000000,
                52.9257889251557,
                103.382408362359,
                20.4413332795932,
            ],
            ["FBS2", np.nan, np.nan, np.nan, np.nan],
            [
                "FBS3",
                42.0500000000000,
                49.8603366776398,
                100.569314056744,
                6.76367706166663,
            ],
            [
                "PRP1",
                23.7100000000000,
                55.5469059266550,
                102.375306747866,
                16.1696500364173,
            ],
            [
                "PRP2",
                22.1700000000000,
                54.9309611559444,
                104.125163365383,
                21.3749715238921,
            ],
            [
                "PRP3",
                15.5700000000000,
                62.4285762204558,
                105.450142200944,
                26.5198336646972,
            ],
            ["GEO1", np.nan, np.nan, np.nan, np.nan],
            ["GEO2", np.nan, np.nan, np.nan, np.nan],
            ["GEO3", np.nan, np.nan, np.nan, np.nan],
            ["GEO4", np.nan, np.nan, np.nan, np.nan],
            ["SBH1", np.nan, np.nan, np.nan, np.nan],
            ["SBH3", np.nan, np.nan, np.nan, np.nan],
            [
                "SBH4",
                20.3700000000000,
                55.6852414294299,
                108.318946137473,
                35.6853624797699,
            ],
            ["AU", np.nan, 72.0940000000410, 106.616999999998, 32.8130000000001],
            ["VE", np.nan, 22.4200000000419, 113.208000000013, 33.6080000000000],
        ],
        columns=["Borehole", "Depth", "x", "y", "z"],
    )

    s32 = pd.DataFrame(
        [
            [
                "INJ1",
                32.1600000000000,
                45.5633843126627,
                105.478518968221,
                15.0573464669582,
            ],
            [
                "INJ2",
                26.6800000000000,
                57.8073320670724,
                107.306250340905,
                14.4650969105453,
            ],
            [
                "FBS1",
                27.8300000000000,
                50.3290893653423,
                106.171904229837,
                18.0673147031311,
            ],
            ["FBS2", np.nan, np.nan, np.nan, np.nan],
            [
                "FBS3",
                33.3900000000000,
                54.2792357114562,
                105.850510733759,
                12.0151428558632,
            ],
            [
                "PRP1",
                29.3100000000000,
                52.9575627461074,
                105.461165785743,
                12.2795631618469,
            ],
            [
                "PRP2",
                27.4300000000000,
                52.1837467076006,
                107.640172806155,
                18.5883747812522,
            ],
            [
                "PRP3",
                19.3400000000000,
                60.9851229353604,
                108.547048822494,
                24.9265628179348,
            ],
            ["GEO1", np.nan, np.nan, np.nan, np.nan],
            ["GEO2", np.nan, np.nan, np.nan, np.nan],
            ["GEO3", np.nan, np.nan, np.nan, np.nan],
            ["GEO4", np.nan, np.nan, np.nan, np.nan],
            ["SBH1", np.nan, np.nan, np.nan, np.nan],
            ["SBH3", np.nan, np.nan, np.nan, np.nan],
            [
                "SBH4",
                23.7900000000000,
                53.4952731274406,
                110.928848729036,
                35.9834351199669,
            ],
            ["AU", np.nan, 72.1850000000559, 110.024999999994, 33.6389999999999],
            ["VE", np.nan, 25.1250000000000, 118.238000000012, 33.7619999999999],
        ],
        columns=["Borehole", "Depth", "x", "y", "z"],
    )

    # Drop rows with nan-value in all of columns 'x', 'y', 'z'.
    s11 = s11.dropna(axis="index", how="all", subset=["x", "y", "z"])
    s12 = s12.dropna(axis="index", how="all", subset=["x", "y", "z"])
    s13 = s13.dropna(axis="index", how="all", subset=["x", "y", "z"])
    s31 = s31.dropna(axis="index", how="all", subset=["x", "y", "z"])
    s32 = s32.dropna(axis="index", how="all", subset=["x", "y", "z"])

    sz = {"S1_1": s11, "S1_2": s12, "S1_3": s13, "S3_1": s31, "S3_2": s32}
    return sz


### THIS METHOD FAILS BECAUSE THE POINTS IN E.G. S11_INTERP_GRID ARE NOT A LINEAR INTERPOLATION ========================
def get_data_interp_grid():
    """ Get the coordinates of matlab-computed linear shearzones

    This method looks up the shearzones in '06_ShearzoneInterpolation'
    and gets the files 'Sxx_interp_grid', which contains the shearzone planes
    computed by the matlab script.

    The target files are named 'S11_interp_grid', etc.

    Returns:
    dictionary
        level 1: name of shearzone: 'S11', 'S12', 'S13', 'S31', 'S32'.
            level 2: 'x', 'y', 'z'
                level 3: np.ndarray of floats.
    """

    # Get path to files
    wd = Path.cwd()
    if (wd / "01BasicInputData").is_dir():
        root_path = wd / "01BasicInputData" / "06_ShearzoneInterpolation"
    else:
        root_path = wd / "GTS" / "01BasicInputData" / "06_ShearzoneInterpolation"

    # Shearzones
    sz_names = ["S11", "S12", "S13", "S31", "S32"]
    suffix = "_interp_grid.txt"
    sz_interp_grid = {s: {} for s in sz_names}

    for sz_name in sz_names:
        path = root_path / (sz_name + suffix)
        data = np.genfromtxt(path)
        sz_interp_grid[sz_name]["x"] = data[:20].flatten(order="C")
        sz_interp_grid[sz_name]["y"] = data[20:40].flatten(order="C")
        sz_interp_grid[sz_name]["z"] = data[40:].flatten(order="C")

    return sz_interp_grid


def plot_points3d(*p, labels=None):
    """ Scatter plot of arbitary many point cloud sets (1 cloud per argument).
        ndarray.shape == (3,n_i)
    """
    if labels in None:
        labels = list(range(len(p[0])))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    t = p[0]
    for i in range(len(t)):
        s = t[i]
        # print(f'hi {i} of {len(p)}')
        ax.scatter(s[0], s[1], s[2], label=labels[i])
    plt.legend()
    plt.show()


def unpack_sz(sz: dict):
    """ Wrapper to unpack sz dicts to plot 3d.

    I.e. from coordinates as columns x, y, z
    to np.ndarray.shape == (3, n)
    """
    arrays = []

    keys = list(sz.keys())
    for key in keys:
        pts = sz[key][["x", "y", "z"]].to_numpy().T
        arrays.append(pts.copy())

    return arrays, keys
