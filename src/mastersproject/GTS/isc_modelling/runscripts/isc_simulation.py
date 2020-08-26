import logging
from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np

from GTS import ISCBiotContactMechanics, BiotParameters
from GTS.isc_modelling.optimal_scaling import best_cond_numb
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import NewtonParameters, TimeMachinePhasesConstantDt
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logging.basicConfig(level=logging.INFO)


def prepare_setup(
    length_scale, scalar_scale,
) -> Tuple[ISCBiotContactMechanics, NewtonParameters, TimeStepProtocol]:
    """ Validation on the ISC grid"""
    injection_protocol, time_params = isc_dt_and_injection_protocol()

    newton_params = NewtonParameters(convergence_tol=1e-6, max_iterations=30,)
    sz = 5
    path = Path(__file__).parent / "isc_simulation/scaling-delete-me"
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=length_scale,
        scalar_scale=scalar_scale,
        # base=Path("/home/haakonervik/mastersproject-data"),
        # head="validation_example",
        folder_name=path,
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
    return setup, newton_params, time_params


def validation():
    setup, newton_params, time_params = prepare_setup(
        length_scale=12.8, scalar_scale=10 * pp.GIGA,
    )
    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)

    time_machine.run_simulation()
    return time_machine


def optimal_scaling():
    def assemble_matrix(values):
        ls, log_ss = values
        ss = np.float_power(10, log_ss)
        setup, newton, time = prepare_setup(length_scale=ls, scalar_scale=ss,)
        setup.prepare_simulation()
        A, b = setup.assembler.assemble_matrix_rhs()
        return A

    initial_guess = np.array([0.05, 6])
    results = best_cond_numb(assemble_matrix, initial_guess)
    return results


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
    phase_limits = [
        -1e2 * pp.YEAR,
        0,
        _10min,
        2 * _10min,
        3 * _10min,
        4 * _10min,
        7 * _10min,
    ]
    rates = [0, 10, 15, 20, 25, 0]
    rates = [r / 60 for r in rates]  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    time_steps = [1e2 * pp.YEAR / 2, _1min, _1min, _1min, _1min, 3 * _1min]
    time_step_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)

    return injection_protocol, time_step_protocol
