""" Parameter setup for Grimsel Test Site"""
from __future__ import annotations  # forward reference to not-yet-constructed model

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pendulum
from pydantic import BaseModel, validator

import porepy as pp
from GTS import ISCData

logger = logging.getLogger(__name__)


# --- Parameters ---
def stress_tensor() -> np.ndarray:
    """ Stress at ISC test site

    Values from Krietsch et al 2019
    """

    # Note: Negative side due to compressive stresses
    stress_value = -np.array([13.1, 9.2, 8.7]) * pp.MEGA * pp.PASCAL
    dip_direction = np.array([104.48, 259.05, 3.72])
    dip = np.array([39.21, 47.90, 12.89])

    def r(th, gm):
        """ Compute direction vector of a dip (th) and dip direction (gm)."""
        rad = np.pi / 180
        x = np.cos(th * rad) * np.sin(gm * rad)
        y = np.cos(th * rad) * np.cos(gm * rad)
        z = -np.sin(th * rad)
        return np.array([x, y, z])

    rot = r(th=dip, gm=dip_direction)

    # Orthogonalize the rotation matrix (which is already close to orthogonal)
    rot, _ = np.linalg.qr(rot)

    # Stress tensor in principal coordinate system
    stress = np.diag(stress_value)

    # Stress tensor in euclidean coordinate system
    stress_eucl: np.ndarray = np.dot(np.dot(rot, stress), rot.T)
    return stress_eucl


# Define the rock type at Grimsel Test Site
class GrimselGranodiorite(pp.UnitRock):
    def __init__(self):
        super().__init__()
        from porepy.params import rock as pp_rock

        self.PERMEABILITY = 1.8e-20
        self.THERMAL_EXPANSION = 1
        self.DENSITY = 2700 * pp.KILOGRAM / (pp.METER ** 3)

        # Lamé parameters
        self.YOUNG_MODULUS = (
            40 * pp.GIGA * pp.PASCAL
        )  # Selvadurai (2019): Biot aritcle --> Table 5., on Pahl et. al (1989)
        self.POISSON_RATIO = (
            0.25  # Selvadurai (2019): Biot aritcle --> Table 5., on Pahl et. al (1989)
        )

        self.LAMBDA, self.MU = pp_rock.lame_from_young_poisson(
            self.YOUNG_MODULUS, self.POISSON_RATIO
        )

        self.FRICTION_COEFFICIENT = 0.8  # TEMPORARY: FRICTION COEFFICIENT TO 0.2
        self.POROSITY = 0.7 / 100

    def lithostatic_pressure(self, depth):
        """ Lithostatic pressure.

        NOTE: Returns positive values for positive depths.
        Use the negative value when working with compressive
        boundary conditions.
        """
        rho = self.DENSITY
        return rho * depth * pp.GRAVITY_ACCELERATION


# --- Models ---


class BaseParameters(BaseModel):
    """ Common parameters for any model implementation

    length_scale, scalar_scale : float
        scaling coefficients for variables and geometry
    head, folder_name : str, Path
        Determine the folder to store all results
    viz_file_name : Path
        base file name of all visualization files (.vtu, .pvd)
    solver : str
        name of linear solver
    time, time_step, end_time : float
        time stepping
    """

    # Scaling
    length_scale: float = 1  # > 0
    scalar_scale: float = 1  # > 0

    # Directories
    base: Path = Path("C:/Users/haako/mastersproject-data")
    head: Optional[str] = None
    folder_name: Optional[Path] = None

    viz_file_name: Path = "simulation_run"

    # Linear solver
    linear_solver: str = "direct"

    # Time-stepping
    time: float = 0
    time_step: float = 1
    end_time: float = 1

    # Fluid and temperature. Default is ISC temp (11 C).
    fluid: pp.UnitFluid = pp.Water(theta_ref=11)

    # Rock parameters
    rock: pp.UnitRock = pp.Granite(theta_ref=11)

    # --- Validators ---

    @validator("length_scale", "scalar_scale")
    def validate_scaling(cls, v):  # noqa
        assert v > 0
        return v

    @validator("folder_name", always=True)
    def construct_absolute_path(cls, p: Optional[Path], values):  # noqa
        """ Construct a valid path, either from 'folder_name' or 'head'."""
        head: str = values["head"]
        base: Path = values["base"]
        if not bool(head) ^ bool(p):  # XOR operator
            raise ValueError("Exactly one of head and folder_name should be set")

        if head:
            # Creates a folder on the form
            #   <base>/results/<YYMMDD>/<head>
            root = base / "results"
            date = pendulum.now().format("YYMMDD")
            p = root / date / head
        else:
            p = p.resolve()

        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        # Needed to allow fluid initiation
        arbitrary_types_allowed = True


class GeometryParameters(BaseParameters):
    """ Parameters for geometry"""

    # Define identifying names for shear zones and the intact 3d matrix.
    shearzone_names: Optional[List[str]] = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]
    intact_name: str = "intact"

    bounding_box: Optional[Dict[str, float]] = {
        "xmin": -100,
        "xmax": 200,
        "ymin": 0,
        "ymax": 300,
        "zmin": -100,
        "zmax": 200,
    }

    _sz = 20
    mesh_args: Optional[Dict[str, float]] = {
        "mesh_size_frac": _sz,
        "mesh_size_min": 0.2 * _sz,
        "mesh_size_bound": 3 * _sz,
    }

    @property
    def n_frac(self):
        return len(self.shearzone_names) if self.shearzone_names else 0


class MechanicsParameters(GeometryParameters):
    """ Parameters for a mechanics model"""

    stress: np.ndarray = stress_tensor()
    # See "numerics > contact_mechanics > contact_conditions.py > ColoumbContact"
    # for details on dilation angle
    dilation_angle: float = 0

    # Parameters for Newton solver
    newton_options = {
        "max_iterations": 40,
        "nl_convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
    }

    # Gravity for mechanics
    gravity_src: bool = False
    gravity_bc: bool = False

    class Config:
        arbitrary_types_allowed = True


class FlowParameters(GeometryParameters):
    """ Parameters for the flow model

    source_scalar_borehole_shearzone : Dict[str, str]
        In conjunction with method shearzone_injection_cell,
        define the borehole and shearzone to inject into
    well_cells : method(FlowParameters, pp.GridBucket) -> None
        A method to tag non-zero injection cells
    injection_rate
    """

    source_scalar_borehole_shearzone: Optional[Dict[str, str]] = {
        "shearzone": "S1_2",
        "borehole": "INJ1",
    }

    well_cells: Callable[["FlowParameters", pp.GridBucket], None] = None
    injection_rate: float = 0

    # Set transmissivity in fractures
    frac_transmissivity: Union[float, List[float]]

    # See 'methods to estimate permeability of shear zones at Grimsel Test Site'
    # in 'Presentasjon til underveismøte' for details on relations between
    # aperture, transmissivity, permeability and hydraulic conductivity

    @property
    def initial_fracture_aperture(self) -> Union[Dict[str, float], None]:
        """ Initial aperture, computed from initial transmissivity

        Assumes uniform transmissivity in each shear zone
        """

        transmissivity: Union[float, List[float]] = self.frac_transmissivity
        shear_zones: List = self.shearzone_names
        if shear_zones:
            n_sz = len(shear_zones)
            if isinstance(transmissivity, (float, int)):
                transmissivity = [transmissivity] * n_sz
            assert len(transmissivity) == len(shear_zones)
            initial_aperture = {
                s: self.b_from_T(t) for s, t in zip(shear_zones, transmissivity)
            }
            return initial_aperture
        else:
            return None

    @property
    def rho_g_over_mu(self):
        mu = self.fluid.dynamic_viscosity()
        rho = self.fluid.density()
        g = pp.GRAVITY_ACCELERATION
        return rho * g / mu

    @property
    def mu_over_rho_g(self):
        return 1 / self.rho_g_over_mu

    def T_from_K_b(self, K, b):
        return K * b

    def K_from_k(self, k):
        return k * self.rho_g_over_mu

    def cubic_law(self, a):
        # k from a
        return a ** 2 / 12

    def T_from_b(self, b):
        return self.T_from_K_b(self.K_from_k(self.cubic_law(b)), b)

    def b_from_T(self, T):
        return np.cbrt(T * 12 * self.mu_over_rho_g)

    # Validators
    @validator("source_scalar_borehole_shearzone")
    def validate_source_scalar_borehole_shearzone(cls, v, values):  # noqa
        if v:
            assert "shearzone" in v
            assert "borehole" in v
            assert v["shearzone"] in values["shearzone_names"]
        return v


class BiotParameters(FlowParameters, MechanicsParameters):
    """ Parameters for the Biot problem with contact mechanics"""

    # Selvadurai (2019): Biot aritcle --> Table 9., on Pahl et. al (1989), mean of aL, aU.
    alpha: float = 0.57


# --- Flow injection cell taggers ---


def nd_injection_cell_center(params: FlowParameters, gb: pp.GridBucket) -> None:
    """ Tag the center cell of the nd-grid with 1 (injection)

    Parameters
    ----------
    params : FlowParameters
    gb : pp.GridBucket

    """

    # Get the center of the domain.
    box = gb.bounding_box()
    pts = (box[1] + box[0]) / 2  # center of domain
    pts = np.atleast_2d(pts).T

    # Get the Nd-grid
    nd_grid = gb.grids_of_dimension(gb.dim_max())[0]

    # Tag highest dim grid with 1 in the cell closest to the grid center
    _tag_injection_cell(gb, nd_grid, pts, params.length_scale)


def shearzone_injection_cell(params: FlowParameters, gb: pp.GridBucket) -> None:
    """ Tag the borehole - shearzone intersection cell with 1 (injection)

    Parameters
    ----------
    params : FlowParameters
    gb : pp.GridBucket
    """
    # Shorthand
    shearzone = params.source_scalar_borehole_shearzone.get("shearzone")

    # Get intersection point
    pts = shearzone_borehole_intersection(params)

    # Get the grid to inject to
    injection_grid = gb.get_grids(lambda g: gb.node_props(g, "name") == shearzone)[0]
    assert (
        injection_grid.dim == gb.dim_max() - 1
    ), "Injection grid should be a Nd-1 fracture"

    # Tag injection grid with 1 in the injection cell
    _tag_injection_cell(gb, injection_grid, pts, params.length_scale)


def nd_sides_shearzone_injection_cell(
    params: FlowParameters, gb: pp.GridBucket, reset_frac_tags: bool = True,
) -> None:
    """ Tag the Nd cells surrounding a shear zone injection point

    Parameters
    ----------
    params : FlowParameters
        parameters that contain "source_scalar_borehole_shearzone"
        (with "shearzone", and "borehole") and "length_scale".
    gb : pp.GridBucket
        grid bucket
    reset_frac_tags : bool [Default: True]
        if set to False, keep injection tag in the shear zone.
    """
    # Shorthand
    shearzone = params.source_scalar_borehole_shearzone.get("shearzone")

    # First, tag the fracture cell, and get the tag
    shearzone_injection_cell(params, gb)
    fracture = gb.get_grids(lambda g: gb.node_props(g, "name") == shearzone)[0]
    tags = fracture.tags["well_cells"]
    # Second, map the cell to the Nd grid
    nd_grid: pp.Grid = gb.grids_of_dimension(gb.dim_max())[0]
    data_edge = gb.edge_props((fracture, nd_grid))
    mg: pp.MortarGrid = data_edge["mortar_grid"]

    slave_to_master_face = mg.mortar_to_master_int() * mg.slave_to_mortar_int()
    face_to_cell = nd_grid.cell_faces.T
    slave_to_master_cell = face_to_cell * slave_to_master_face
    nd_tags = np.abs(slave_to_master_cell) * tags

    # Set tags on the nd-grid
    nd_grid.tags["well_cells"] = nd_tags
    gb.set_node_prop(nd_grid, "well", nd_tags)

    if reset_frac_tags:
        # reset tags on the fracture
        zeros = np.zeros(fracture.num_cells)
        fracture.tags["well_cells"] = zeros
        gb.set_node_prop(fracture, "well", zeros)


def nd_and_shearzone_injection_cell(params: FlowParameters, gb: pp.GridBucket) -> None:
    """ Wrapper of above method to toggle keep shear zone injection tag"""
    reset_frac_tags = False
    nd_sides_shearzone_injection_cell(params, gb, reset_frac_tags)


def center_of_shearzone_injection_cell(
    params: FlowParameters, gb: pp.GridBucket
) -> None:
    """ Tag the center cell of the given shear zone with 1 (injection)

    Parameters
    ----------
    params : FlowParameters
    gb : pp.GridBucket
    """

    # Shorthand
    shearzone = params.source_scalar_borehole_shearzone.get("shearzone")

    # Get the grid to inject to
    frac: pp.Grid = gb.get_grids(lambda g: gb.node_props(g, "name") == shearzone)[0]
    centers: np.ndarray = frac.cell_centers
    pts = np.atleast_2d(np.mean(centers, axis=1)).T

    # Tag injection grid with 1 in the injection cell
    _tag_injection_cell(gb, frac, pts, params.length_scale)


def _tag_injection_cell(
    gb: pp.GridBucket, g: pp.Grid, pts: np.ndarray, length_scale
) -> None:
    """ Helper method to tag find closest point on g to pts

    The tag is set locally to g and to node props on gb.
    length_scale is used to log the unscaled distance to
    the injection cell from pts.

    Parameters
    ----------
    gb : pp.GridBucket
    g : pp.Grid
    pts : np.ndarray, shape: (3,1)
    length_scale : float

    """
    assert pts.shape == (3, 1), "We only consider one point; array needs shape 3x1"
    tags = np.zeros(g.num_cells)
    ids, dsts = g.closest_cell(pts, return_distance=True)
    tags[ids] = 1
    g.tags["well_cells"] = tags
    gb.set_node_prop(g, "well", tags)

    # Log information on the injection point
    logger.info(
        f"Closest cell found has (unscaled) distance: {dsts[0] * length_scale:4f}\n"
        f"ideal (scaled) point coordinate: {pts.T}\n"
        f"nearest (scaled) cell center coordinate: {g.cell_centers[:, ids].T}\n"
    )


def _shearzone_borehole_intersection(
    borehole: str, shearzone: str, length_scale: float
):
    """ Find the cell which is the intersection of a borehole and a shear zone"""
    # Compute the intersections between boreholes and shear zones
    df = ISCData().borehole_plane_intersection()

    # Get the UNSCALED coordinates of the borehole - shearzone intersection.
    _mask = (df.shearzone == shearzone) & (df.borehole == borehole)
    result = df.loc[_mask, ("x_sz", "y_sz", "z_sz")]
    if result.empty:
        raise ValueError("No intersection found.")

    # Scale the intersection coordinates by length_scale. (scaled)
    pts = result.to_numpy().T / length_scale
    assert pts.shape == (3, 1), "There should only be one intersection"
    return pts


def shearzone_borehole_intersection(params: FlowParameters):
    """ Wrapper to get intersection using FlowParameters class"""
    borehole = params.source_scalar_borehole_shearzone.get("borehole")
    shearzone = params.source_scalar_borehole_shearzone.get("shearzone")
    length_scale = params.length_scale
    return _shearzone_borehole_intersection(borehole, shearzone, length_scale)


# --- other models ---


class GTSModel(BaseModel):
    """ Parameters used at Grimsel Test Site.

    This class defines parameter values for the Biot equations
    which are set up to model the dynamics at the
    In-Situ Stimulation and Circulation (ISC) experiment at
    Grimsel Test Site (GTS).

    This model assumes uniform values on the grid

    The Biot equation according to Berge et al (2019):
        Finite volume discretization for poroelastic media with fractures
        modeled by contact mechanics
    See Equation (1):
        - div sigma = fu                            in interior
        C : sym(grad u) - alpha p I = sigma         in interior
        c dp/dt + alpha div du/dt + div q = fp      in interior
        q = - K grad p                              in interior
        u = guD                                     on part of boundary
        sigma n = guN                               on part of boundary
        p = gpD                                     on part of boundary
        q n = gpN                                   on part of boundary

    We shall define:
    fu              body force [Pa / m]
    C(mu, lambda)   constitutive relation, defined by Lame parameters [Pa]
    c               mass term [1/Pa]
    alpha           biot coefficient [-]
    fp              scalar source
    K               conductivity (permeability / viscosity) [m2 / (Pa s)]
    guD             Dirichlet boundary condition, mechanics
    guN             Neumann boundary condition, mechanics
    gpD             Dirichlet boundary condition, scalar
    gpN             Neumann boundary condition, scalar
    """
