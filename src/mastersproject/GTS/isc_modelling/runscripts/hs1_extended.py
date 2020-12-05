""" Simulation script for HS1 stimulation experiment at Grimsel Test Site
Author: Haakon Ludvig Langeland Ervik

Instructions:
To run a simulation with configuration same as in the thesis, run
the method "box_runscript" with the parameter 'case' as "A1", "A2", "B1" or "B2".

"""

import logging
from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np

from GTS import BiotParameters, stress_tensor
from GTS.isc_modelling.isc_box_model import ISCBoxModel
from GTS.isc_modelling.parameter import shearzone_injection_cell, GrimselGranodiorite
from GTS.time_machine import NewtonParameters, TimeMachinePhasesConstantDt
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logger = logging.getLogger(__name__)


def prepare_B1() -> Tuple[BiotParameters, NewtonParameters, TimeStepProtocol]:
    """Hydro shearing experiment HS1

    HS1 targeted S1.3 through INJ2 at 39.75 - 40.75 m.
    """
    tunnel_equilibration_time = 30 * pp.YEAR
    injection_protocol, time_params = isc_dt_and_injection_protocol(
        tunnel_equilibration_time
    )

    newton_params = NewtonParameters(
        convergence_tol=1e-4,
        max_iterations=300,
    )
    base = Path.home() / "research/mastersproject-data/hs1-radT"
    head = f"B1/test_S1-S5_longdt"

    fraczone_bounding_box = {
        "xmin": -1,
        "ymin": 80,
        "zmin": -5 - 5,
        "xmax": 86 + 10,
        "ymax": 151 + 5,
        "zmax": 41 + 5,
    }
    fbb = fraczone_bounding_box
    radius = 150
    bounding_box = {
        "xmin": (fbb["xmin"] + fbb["xmax"]) / 2 - radius,
        "ymin": (fbb["ymin"] + fbb["ymax"]) / 2 - radius,
        "zmin": (fbb["zmin"] + fbb["zmax"]) / 2 - radius,
        "xmax": (fbb["xmin"] + fbb["xmax"]) / 2 + radius,
        "ymax": (fbb["ymin"] + fbb["ymax"]) / 2 + radius,
        "zmax": (fbb["zmin"] + fbb["zmax"]) / 2 + radius,
    }  # Make global bounding box 150m in every direction from center of the fractured zone.

    biot_params = BiotParameters(
        # BaseParameters
        length_scale=1.0,
        scalar_scale=1e11,
        base=base,
        head=head,
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        rock=GrimselGranodiorite(
            PERMEABILITY=1e-21,
        ),  # low k: 5e-21 -- high k: 2 * 5e-21
        gravity=False,
        # GeometryParameters
        fractures=[
            # "S1_1",
            # "S1_2",
            "S1_3",
            "S3_1",
        ],  # "S3_2"],  # ["S1_2", "S3_1"],
        bounding_box=bounding_box,
        fraczone_bounding_box=fraczone_bounding_box,
        # MechanicsParameters
        stress=stress_tensor(),
        dilation_angle=np.radians(3),
        newton_options=newton_params.dict(),
        # FlowParameters
        source_scalar_borehole_shearzone={
            "shearzone": "S1_3",
            "borehole": "INJ2",
        },
        well_cells=shearzone_injection_cell,
        tunnel_equilibrium_time=tunnel_equilibration_time,
        injection_protocol=injection_protocol,
        frac_transmissivity=[
            # 5e-8,
            # 1e-9,
            8.3e-11,  # Pre-stimulation T during HS1, observed in S1.3-INJ2
            3.7e-7,
            # 1e-9,
        ],
        # Initial transmissivites for
        # ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]:
        # [5e-8,   1e-9,   5e-10,  3.7e-7, 1e-9]
        # Note background hydraulic conductivity: 1e-14 m/s
        near_injection_transmissivity=5e-9,  # Transmissivity near well rough log-average of init and resulting T 
        near_injection_t_radius=10,
    )

    return biot_params, newton_params, time_params


def box_runscript(run=True):
    biot_params, newton_params, time_params = prepare_B1()
    # *2 gives ~25kc on 4fracs.
    # l=0.3, lcin 5*5*l, lcout 50*10*l
    # lcin = 5*10 lcout = 50*20

    # For 4frac setups:
    # lcin=5*1.4, lcout=50*1.4 --> 44k*3d + 5k*2d + 50*1d
    # lcin=5*10.3, lcout=50*10.4 --> 3k*3d + 200*2d + 9*1d
    # lcin=5*5.4, lcout=50*5.4 --> 6k*3d + 500*2d + 15*1d
    # lcin=5*3, lcout=50*3 --> 12k*3d + 1.2k*2d + 27*1d
    # lcin=5*2, lcout=50*2 --> 22k*3d, 2.5k*2d + 39*1d

    # For 2frac setups (S1.3, S3.1):
    # lcin=8, lcout=20 --> 60k*3d + 2.4k*2d + 19*1d
    setup = ISCBoxModel(biot_params, lcin=12, lcout=25)
    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)

    if run:
        time_machine.run_simulation()
    return time_machine


def isc_dt_and_injection_protocol(tunnel_time: float):
    """Stimulation protocol for the rate-controlled phase of the ISC experiment

    Here, we consider Doetsch et al (2018) [see e.g. p. 78/79 or App. J]
            Hydro Shearing Protocol:
            * Injection Cycle 3:
                - Four injection steps of 10, 15, 20 and 25 l/min
                - Each step lasts 10 minutes.
                - Then, the interval is shut-in and monitored for 40 minutes.
                - Venting was foreseen at 20 minutes

            For this setup, we only consider Injection Cycle 3.

    Parameters
    ----------
        tunnel_time : float
            AU, VE tunnels were constructed 30 years ago.
    """
    assert tunnel_time > 0
    _1min = pp.MINUTE
    _5min = 5 * _1min
    _10min = 10 * _1min
    initialization_time = 30e3 * pp.YEAR
    phase_limits = [
        -initialization_time,
        -tunnel_time,
        0,
        # S1,       5 min
        1 * _5min,
        # # S2,       5 min
        2 * _5min,
        # S3,       5 min
        3 * _5min,
        # # S4,       5 min
        4 * _5min,
        # # S5,       15 min
        7 * _5min,
        # # shut-in,  46 min
        # 81 * _1min,
        # # Venting,  29 min
        # 11 * _10min,
    ]
    rates = [
        0,  # initialization
        0,  # tunnel calibration
        15,  # C3, step 1
        20,  # C3, step 2
        25,  # C3, step 3
        30,  # C3, step 4
        35,  # C3, step 5
        # 0,  # shut-in
        # 0,  # venting (currently modelled as: shut-in)
    ]
    rates = [r / 60 for r in rates]  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    time_steps = [
        initialization_time / 2,
        tunnel_time / 2,
        5.0 * _1min,  # S1, 5min
        2.5 * _1min,  # S2, 5min
        1.0 * _1min,  # S3, 5min
        2.5 * _1min,  # S4, 5min
        2.5 * _1min,  # S5, 15min
        # 16 * _1min,  # shut-in, 46min
        # 15 * _1min,  # venting
    ]
    time_step_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)

    return injection_protocol, time_step_protocol
