""" Parameter setup for Grimsel Test Site"""
from __future__ import annotations  # forward reference to not-yet-constructed model

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
# from scipy import spatial

import numpy as np

import pendulum
import porepy as pp
from GTS import ISCData
from GTS.time_protocols import InjectionRateProtocol
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


# --- Parameters ---
def stress_tensor() -> np.ndarray:
    """Stress at ISC test site

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
class UnitRock(BaseModel):
    """ Default rock model"""

    PERMEABILITY = 1.0 * (pp.METER ** 2)
    DENSITY = 1.0 * (pp.KILOGRAM / pp.METER ** 3)
    POROSITY = 1.0
    YOUNG_MODULUS = 1.0 * pp.PASCAL
    POISSON_RATIO = 1.0 * pp.PASCAL
    FRICTION_COEFFICIENT = 1.0

    @property
    def LAMBDA(self):
        """ Compute Lame parameters from Young's modulus and Poisson's ratio."""
        e, nu = self.YOUNG_MODULUS, self.POISSON_RATIO
        return e * nu / ((1 + nu) * (1 - 2 * nu))

    @property
    def MU(self):
        """ Compute Lame parameters from Young's modulus and Poisson's ratio."""
        e, nu = self.YOUNG_MODULUS, self.POISSON_RATIO
        return e / (2 * (1 + nu))

    @property
    def BULK_MODULUS(self):
        """ Compute bulk modulus from Young's modulus and Poisson's ratio."""
        e, nu = self.YOUNG_MODULUS, self.POISSON_RATIO
        return e / (3 * (1 - 2 * nu))

    def lithostatic_pressure(self, depth):
        """Lithostatic pressure.

        NOTE: Returns positive values for positive depths.
        Use the negative value when working with compressive
        boundary conditions.
        """
        rho = self.DENSITY
        return rho * depth * pp.GRAVITY_ACCELERATION


class GrimselGranodiorite(UnitRock):
    """ Grimsel Granodiorite parameters"""

    PERMEABILITY = 5e-21
    DENSITY = 2700 * pp.KILOGRAM / (pp.METER ** 3)
    POROSITY = 0.7 / 100

    # Lamé parameters
    YOUNG_MODULUS = (
        40 * pp.GIGA * pp.PASCAL
    )  # Selvadurai (2019): Biot article --> Table 5., on Pahl et. al (1989)
    POISSON_RATIO = (
        0.25  # Selvadurai (2019): Biot article --> Table 5., on Pahl et. al (1989)
    )

    FRICTION_COEFFICIENT = 0.8  # TEMPORARY: FRICTION COEFFICIENT TO 0.2


# Fluid
class UnitFluid(BaseModel):
    """ Unit fluid at constant temperature"""

    COMPRESSIBILITY: float = 1 / pp.PASCAL
    theta_ref: float = 11 * pp.CELSIUS

    @property
    def BULK(self) -> float:
        return 1 / self.COMPRESSIBILITY

    @property
    def density(self) -> float:
        """ Units: kg / m^3 """
        return 1.0

    @property
    def dynamic_viscosity(self) -> float:
        """Units: Pa s"""
        return 1.0

    def hydrostatic_pressure(self, depth) -> Union[float, np.ndarray]:
        rho = self.density
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Water(UnitFluid):
    """ Water at constant temperature"""

    COMPRESSIBILITY = 4e-10 / pp.PASCAL  # Moderate dependency on theta

    @property
    def density(self):
        """ Units: kg / m^3 """
        # at 11 C: 999.622
        theta = self.theta_ref
        theta_0 = 10 * pp.CELSIUS
        rho_0 = 999.8349 * (pp.KILOGRAM / pp.METER ** 3)
        return rho_0 / (1.0 + self.thermal_expansion(theta - theta_0))

    @property
    def dynamic_viscosity(self):
        """Units: Pa s"""
        # at 11 C: 0.001264
        theta = self.theta_ref
        theta = pp.CELSIUS_to_KELVIN(theta)
        mu_0 = 2.414 * 1e-5 * (pp.PASCAL * pp.SECOND)
        return mu_0 * np.power(10, 247.8 / (theta - 140))

    @staticmethod
    def thermal_expansion(delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return (
            0.0002115
            + 1.32 * 1e-6 * delta_theta
            + 1.09 * 1e-8 * np.power(delta_theta, 2)
        )


# --- Models ---


class BaseParameters(BaseModel):
    """Common parameters for any model implementation

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
    use_temp_path: bool = False  # Create a temporary path
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
    fluid: UnitFluid = Water(theta_ref=11)

    # Rock parameters
    rock: UnitRock = GrimselGranodiorite()

    # Gravity
    gravity: bool = False
    # Either use constant density model (i.e. rho = 1000 kg/m3)
    # or variable density, rho = rho0 * exp( c * (p - p0) )
    constant_density: float = True
    # Depth is needed because hydrostatic pressure depends on the depth.
    # We center the domain at 480m below the surface (see Krietsch et al, 2018a).
    depth: float = 480 * pp.METER

    # --- Validators ---

    @validator("length_scale", "scalar_scale")
    def validate_scaling(cls, v):  # noqa
        assert v > 0
        return v

    @validator("folder_name", always=True)
    def construct_absolute_path(cls, p: Optional[Path], values):  # noqa
        """ Construct a valid path, either from 'folder_name' or 'head'."""
        use_temp_path: bool = values["use_temp_path"]
        if use_temp_path:
            return None
            # path = TemporaryDirectory()
            # # Use path.cleanup() to remove directory and contents
            # return path

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
    fractures: Optional[List[str]] = ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]
    intact_name: str = "intact"

    bounding_box: Optional[Dict[str, float]] = {
        "xmin": -100,
        "xmax": 200,
        "ymin": 0,
        "ymax": 300,
        "zmin": -100,
        "zmax": 200,
    }
    # Set bounding box for fractured zone (only applicable for isc_box_model).
    fraczone_bounding_box: Optional[Dict[str, float]] = None

    _sz = 20
    mesh_args: Optional[Dict[str, float]] = {
        "mesh_size_frac": _sz,
        "mesh_size_min": 0.2 * _sz,
        "mesh_size_bound": 3 * _sz,
    }

    @property
    def n_frac(self):
        return len(self.fractures) if self.fractures else 0


class MechanicsParameters(GeometryParameters):
    """ Parameters for a mechanics model"""

    stress: np.ndarray  # e.g.: stress_tensor()
    # See "numerics > contact_mechanics > contact_conditions.py > ColoumbContact"
    # for details on dilation angle
    dilation_angle: float = 0
    # Cohesion (for numerical stability)
    cohesion: float = 0.0

    # Parameters for Newton solver
    newton_options = {
        "max_iterations": 40,
        "convergence_tol": 1e-10,
        "nl_divergence_tol": 1e5,
    }

    class Config:
        # For numpy arrays (currently this doesn't work as expected).
        arbitrary_types_allowed = True


class FlowParameters(GeometryParameters):
    """Parameters for the flow model

    source_scalar_borehole_shearzone : Dict[str, str]
        In conjunction with method shearzone_injection_cell,
        define the borehole and shearzone to inject into
    well_cells : method(FlowParameters, pp.GridBucket) -> None
        A method to tag non-zero injection cells
    injection_rate
    """

    # Injection location, method, and protocol
    source_scalar_borehole_shearzone: Optional[Dict[str, str]] = {
        "shearzone": "S1_2",
        "borehole": "INJ1",
    }
    isc_data: Optional[ISCData] = ISCData()
    well_cells: Callable[["FlowParameters", pp.GridBucket], None] = None
    injection_protocol: InjectionRateProtocol = InjectionRateProtocol.create_protocol(
        [0.0, 1.0], [0.0]
    )

    # Set constant pressure value in tunnel - shear zone intersections
    # See e.g. assemble_matrix_rhs() in isc_model.py
    tunnel_pressure: float = pp.ATMOSPHERIC_PRESSURE
    # Set time for tunnel equilibration
    # NOTE: To "turn off" this effect, set value to a negative value larger than end_time.
    tunnel_equilibrium_time: float = 30 * pp.YEAR

    # Set transmissivity in fractures. List in same order as fractures
    # frac_transmissivity: Union[float, List[float]] = 1

    # Set initial fault thickness `b` and fracture aperture `a`
    fault_thickness: List[float]
    frac_aperture: List[float]

    def initial_fault_thickness(self, g: pp.Grid, fault: str) -> np.ndarray:
        """Compute initial fault thickness in each grid cell"""
        fault_thickness = self.fault_thickness[self.fractures.index(fault)]
        return fault_thickness * np.ones(g.num_cells)

    def initial_aperture(self, g: pp.Grid, fault: str) -> np.ndarray:
        """Compute initial fault aperture in each grid cell"""
        aperture = self.frac_aperture[self.fractures.index(fault)]
        return aperture * np.ones(g.num_cells)

    # Different transmissivity near injection point
    # If radius is set to 0, this is not activated.
    # near_injection_transmissivity: float = 1
    # near_injection_t_radius: float = 0

    # See 'methods to estimate permeability of shear zones at Grimsel Test Site'
    # in 'Presentasjon til underveismøte' for details on relations between
    # aperture, transmissivity, permeability and hydraulic conductivity

    # def compute_initial_aperture(self, g: pp.Grid, shear_zone: str) -> np.ndarray:
    #     """ Compute initial aperture"""
    #
    #     # First, get the background aperture.
    #     aperture = np.ones(g.num_cells)
    #
    #     # Set background aperture
    #     background_aperture = self.initial_background_aperture(shear_zone)
    #     aperture *= background_aperture
    #
    #     # Then, adjust for a heterogeneous permeability near the injection point
    #     injection_shearzone = self.source_scalar_borehole_shearzone.get("shearzone")
    #     if self.near_injection_t_radius > 0 and shear_zone == injection_shearzone:
    #         radius = self.near_injection_t_radius / self.length_scale
    #         pts = shearzone_borehole_intersection(self)  # already scaled
    #         cells = g.cell_centers.T
    #         tree = spatial.cKDTree(cells)
    #         inside_idx = tree.query_ball_point(pts.T, radius)[0]
    #         aperture[inside_idx] = self.b_from_T(self.near_injection_transmissivity)
    #
    #     return aperture
    #
    # def initial_background_aperture(self, shear_zone):
    #     """ Compute initial fracture aperture for a given shear zone"""
    #     frac_T: Union[float, List[float]] = self.frac_transmissivity
    #     if isinstance(frac_T, float):
    #         ft = frac_T
    #     else:
    #         # get the transmissivity corresponding to the shear zone name
    #         # position in the fractures list.
    #         ft = frac_T[self.fractures.index(shear_zone)]
    #     return self.b_from_T(ft)

    @property
    def rho_g_over_mu(self):
        mu = self.fluid.dynamic_viscosity
        rho = self.fluid.density
        g = pp.GRAVITY_ACCELERATION
        return rho * g / mu

    @property
    def mu_over_rho_g(self):
        return 1 / self.rho_g_over_mu

    def T_from_K_b(self, K, b):
        """Compute `transmissivity` from `hydraulic conductivity` and `fault thickness`"""
        return K * b

    def K_from_k(self, k):
        """Compute `hydraulic conductivity` from `permeability`"""
        return k * self.rho_g_over_mu

    def cubic_law(self, a):
        """Compute `permeability` from `hydraulic aperture`"""
        return np.power(a, 2) / 12

    def K_from_a(self, a):
        """Compute `hydraulic conductivity` from `hydraulic aperture`"""
        return self.K_from_k(self.cubic_law(a))

    def T_from_a_b(self, a, b):
        """Compute `transmissivity` from `hydraulic aperture` and `fault thickness` using the cubic law"""
        return self.T_from_K_b(self.K_from_a(a), b)

    # inverse relations
    def a_from_k(self, k):
        """Compute `hydraulic aperture` from `permeability`"""
        return np.sqrt(12 * k)

    def k_from_K(self, K):
        """Compute `permeability` from `hydraulic conductivity`"""
        return K / self.rho_g_over_mu

    # Validators
    @validator("source_scalar_borehole_shearzone")
    def validate_source_scalar_borehole_shearzone(cls, v, values):  # noqa
        if v:
            assert "shearzone" in v
            assert "borehole" in v
            assert v["shearzone"] in values["fractures"]
        return v


class BiotParameters(FlowParameters, MechanicsParameters):
    """ Parameters for the Biot problem with contact mechanics"""

    # Selvadurai (2019): Biot aritcle --> Table 9., on Pahl et. al (1989), mean of aL, aU.
    alpha: float = 0.54


# --- Flow injection cell taggers ---


def nd_injection_cell_center(params: FlowParameters, gb: pp.GridBucket) -> None:
    """Tag the center cell of the nd-grid with 1 (injection)

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
    """Tag the borehole - shearzone intersection cell with 1 (injection)

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
    params: FlowParameters,
    gb: pp.GridBucket,
    reset_frac_tags: bool = True,
) -> None:
    """Tag the Nd cells surrounding a shear zone injection point

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

    slave_to_master_face = mg.mortar_to_primary_int() * mg.secondary_to_mortar_int()
    face_to_cell = nd_grid.cell_faces.T
    slave_to_master_cell = face_to_cell * slave_to_master_face
    nd_tags = np.abs(slave_to_master_cell) * tags

    # Set tags on the nd-grid
    nd_grid.tags["well_cells"] = nd_tags
    ndd = gb.node_props(nd_grid)
    pp.set_state(ndd, {"well": tags})

    if reset_frac_tags:
        # reset tags on the fracture
        zeros = np.zeros(fracture.num_cells)
        fracture.tags["well_cells"] = zeros
        d = gb.node_props(fracture)
        pp.set_state(d, {"well": zeros})


def nd_and_shearzone_injection_cell(params: FlowParameters, gb: pp.GridBucket) -> None:
    """ Wrapper of above method to toggle keep shear zone injection tag"""
    reset_frac_tags = False
    nd_sides_shearzone_injection_cell(params, gb, reset_frac_tags)


def center_of_shearzone_injection_cell(
    params: FlowParameters, gb: pp.GridBucket
) -> None:
    """Tag the center cell of the given shear zone with 1 (injection)

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
    """Helper method to tag find closest point on g to pts

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
    d = gb.node_props(g)
    pp.set_state(d, {"well": tags})

    # Log information on the injection point
    logger.info(
        f"Closest cell found has (unscaled) distance: {dsts[0] * length_scale:4f}\n"
        f"ideal (scaled) point coordinate: {pts.T}\n"
        f"nearest (scaled) cell center coordinate: {g.cell_centers[:, ids].T}\n"
    )


def _shearzone_borehole_intersection(
    borehole: str,
    shearzone: str,
    length_scale: float,
    isc_data=None,
) -> np.ndarray:
    """ Find the cell which is the intersection of a borehole and a shear zone"""
    # Compute the intersections between boreholes and shear zones
    if isc_data is None:
        df = ISCData().borehole_plane_intersection()
    else:
        df = isc_data.borehole_plane_intersection()

    # Get the UNSCALED coordinates of the borehole - shearzone intersection.
    _mask = (df.shearzone == shearzone) & (df.borehole == borehole)
    result = df.loc[_mask, ("x_sz", "y_sz", "z_sz")]
    if result.empty:
        raise ValueError("No intersection found.")

    # Scale the intersection coordinates by length_scale. (scaled)
    pts = result.to_numpy(dtype=float).T / length_scale
    assert pts.shape == (3, 1), "There should only be one intersection"
    return pts


def shearzone_borehole_intersection(params: FlowParameters) -> np.ndarray:
    """ Wrapper to get intersection using FlowParameters class. Return shape: (3,1)."""
    borehole = params.source_scalar_borehole_shearzone.get("borehole")
    shearzone = params.source_scalar_borehole_shearzone.get("shearzone")
    length_scale = params.length_scale
    isc_data = params.isc_data
    return _shearzone_borehole_intersection(borehole, shearzone, length_scale, isc_data)
