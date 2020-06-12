""" Parameter setup for Grimsel Test Site"""
import numpy as np
import porepy as pp
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)


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
