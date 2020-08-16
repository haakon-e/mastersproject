import logging
from pathlib import Path

import porepy as pp
import numpy as np

from GTS import ISCBiotContactMechanics, BiotParameters
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import NewtonParameters, TimeMachine
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logging.basicConfig(level=logging.INFO)


def validation():
    """ Validation on the ISC grid"""
    time_params = TimeStepProtocol.create_protocol(
        phase_limits=[0, 5*pp.MINUTE, 8.5*pp.MINUTE, 10*pp.MINUTE],
        time_steps=[1*pp.MINUTE] * 3,
    )
    injection_protocol = stimulation_protocol_isc()

    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=30
    )
    sz = 6
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=12.8,
        scalar_scale=10*pp.GIGA,
        base=Path("/home/haakonervik/mastersproject-data"),
        head="validation_example",
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        gravity=True,
        # GeometryParameters
        shearzone_names=["S1_2", "S3_1"],
        mesh_args={
            "mesh_size_frac": sz,
            "mesh_size_min": 0.2 * sz,
            "mesh_size_bound": 3 * sz,
        },
        # MechanicsParameters
        dilation_angle=np.radians(10),
        newton_options=newton_params.dict(),
        # FlowParameters
        well_cells=shearzone_injection_cell,
        injection_protocol=injection_protocol,
        frac_transmissivity=[1e-9, 3.7e-7],
    )
    setup = ISCBiotContactMechanics(biot_params)

    time_machine = TimeMachine(setup, newton_params, time_params, max_newton_failure_retries=1)

    time_machine.run_simulation()
    return time_machine


def stimulation_protocol_isc():
    """ Stimulation protocol for the rate-controlled phase of the ISC experiment

    Here, we consider Doetsch et al (2018) [see e.g. p. 78/79 or App. J]
            Hydro Shearing Protocol:
            * Injection Cycle 3:
                - Four injection steps of 10, 15, 20 and 25 l/min
                - Each step lasts 10 minutes.
                - Then, the interval is shut-in and monitored for 40 minutes.
                - Venting was foreseen at 20 minutes

            For this setup, we only consider Injection Cycle 3.
    """
    _10min = 10 * pp.MINUTE
    phase_limits = [
        -1e2 * pp.YEAR, 0, _10min, 2 * _10min, 3 * _10min, 4 * _10min, 7 * _10min
    ]
    rates = [0, 10, 15, 20, 25, 0]
    rates /= 60  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)
    return injection_protocol
