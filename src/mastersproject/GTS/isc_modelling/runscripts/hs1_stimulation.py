import logging
from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np

from GTS import BiotParameters
from GTS.isc_modelling.isc_box_model import ISCBoxModel
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import NewtonParameters, TimeMachinePhasesConstantDt
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logger = logging.getLogger(__name__)


def prepare_params(
    length_scale, scalar_scale,
) -> Tuple[BiotParameters, NewtonParameters, TimeStepProtocol]:
    """ Hydro shearing experiment HS1

    HS1 targeted S1.3 through INJ2 at 39.75 - 40.75 m.
    """
    injection_protocol, time_params = isc_dt_and_injection_protocol()

    newton_params = NewtonParameters(convergence_tol=1e-6, max_iterations=200,)
    base = Path.home() / "mastersproject-data/hs1"
    #head = "test-biot-effects/4frac-dt10min-nograv-nodil-coarse-k_1e-20"
    head = "4frac-dt10min-nograv-3degdil"
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=length_scale,
        scalar_scale=scalar_scale,
        base=base,
        head=head,
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        gravity=False,
        # GeometryParameters
        shearzone_names=["S1_1", "S1_2", "S1_3", "S3_1"],  # "S3_2"],  # ["S1_2", "S3_1"],
        # MechanicsParameters
        dilation_angle=np.radians(3),
        newton_options=newton_params.dict(),
        # FlowParameters
        source_scalar_borehole_shearzone={"shearzone": "S1_3", "borehole": "INJ2",},
        well_cells=shearzone_injection_cell,
        injection_protocol=injection_protocol,
        frac_transmissivity= [5e-8, 1e-9, 5e-12, 3.7e-7, 1e-9],  # set background s13 T artificially low. 
        # Initial transmissivites for
        # ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]:
        # [5e-8,   1e-9,   5e-10,  3.7e-7, 1e-9]
        # Note background hydraulic conductivity: 1e-14 m/s
        near_injection_transmissivity=8.3e-11,
        near_injection_t_radius=3,
    )

    return biot_params, newton_params, time_params


def box_runscript(run=True):
    biot_params, newton_params, time_params = prepare_params(
        length_scale=0.04, scalar_scale=1e8,
    )
    # *2 gives ~25kc on 4fracs.
    # l=0.3, lcin 5*5*l, lcout 50*10*l
    # lcin = 5*10 lcout = 50*20
    
    # For 4frac setups:
    # lcin=5*1.4, lcout=50*1.4 --> 44k*3d + 5k*2d + 50*1d
    setup = ISCBoxModel(biot_params, lcin=5*1.4, lcout=50*1.4)
    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)
    
    if run:
        time_machine.run_simulation()
    return time_machine


def isc_dt_and_injection_protocol():
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
    _1min = pp.MINUTE
    _10min = 10 * _1min
    initialization_time = 30e3 * pp.YEAR
    phase_limits = [
        -initialization_time,
        0,
        _10min,
        2 * _10min,
        3 * _10min,
        4 * _10min,
        8 * _10min,
    ]
    rates = [
        0,
        #17.5,
        10,
        15,
        20,
        25,
        0,
    ]
    rates = [r / 60 for r in rates]  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    time_steps = [
        initialization_time / 2,
        10 * _1min,
        10 * _1min,
        10 * _1min,
        10 * _1min,
        40 * _1min,
    ]
    time_step_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)

    return injection_protocol, time_step_protocol
