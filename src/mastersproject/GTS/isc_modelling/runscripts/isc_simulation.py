import logging
from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np

from GTS import ISCBiotContactMechanics, BiotParameters
from GTS.isc_modelling.isc_box_model import ISCBoxModel
from GTS.isc_modelling.optimal_scaling import best_cond_numb, condition_number_porepy
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import NewtonParameters, TimeMachinePhasesConstantDt
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def prepare_params(
    length_scale, scalar_scale,
) -> Tuple[BiotParameters, NewtonParameters, TimeStepProtocol]:
    """ Validation on the ISC grid"""
    injection_protocol, time_params = isc_dt_and_injection_protocol()

    newton_params = NewtonParameters(convergence_tol=1e-6, max_iterations=200,)
    sz = 10
    # path = Path(__file__).parent / "isc_simulation/200830/box-test/t1-gravity"
    root = Path.home()
    path = root / "mastersproject-data/results/200830/5f-50kc/t1"
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
        shearzone_names=["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"],
        mesh_args={
            "mesh_size_frac": sz,
            "mesh_size_min": 0.1 * sz,
            "mesh_size_bound": 3 * sz,
        },
        bounding_box=None,
        # MechanicsParameters
        dilation_angle=np.radians(3),
        newton_options=newton_params.dict(),
        # FlowParameters
        well_cells=shearzone_injection_cell,
        injection_protocol=injection_protocol,
        frac_transmissivity=[5e-8, 1e-9, 5e-10, 3.7e-7, 1e-9],
        # Initial transmissivites for
        # ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]:
        # [5e-8,   1e-9,   5e-10,  3.7e-7, 1e-9]
    )

    return biot_params, newton_params, time_params


def validation():
    biot_params, newton_params, time_params = prepare_params(
        length_scale=0.01, scalar_scale=1e6,
    )
    setup = ISCBiotContactMechanics(biot_params)
    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)

    time_machine.run_simulation()
    return time_machine


def box_validation():
    biot_params, newton_params, time_params = prepare_params(
        length_scale=0.01, scalar_scale=1e6,
    )
    setup = ISCBoxModel(biot_params, lcin=5 * 1.5, lcout=50 * 1.5)
    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)

    time_machine.run_simulation()
    return time_machine


# --- CONDITION NUMBER ---


def cond_num_isc(ls, log_ss):
    values = np.array([ls, log_ss])
    A = assemble_matrix(values)
    cond = condition_number_porepy(A)
    print(f"cond pp: {cond:.2e}")


def assemble_matrix(values):
    ls, log_ss = values
    ss = np.float_power(10, log_ss)
    biot_params, newton, time = prepare_params(length_scale=ls, scalar_scale=ss,)
    setup = ISCBiotContactMechanics(biot_params)
    setup.prepare_simulation()
    A, b = setup.assembler.assemble_matrix_rhs()
    return A


def optimal_scaling():
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
    initialization_time = 30e3 * pp.YEAR
    phase_limits = [
        -initialization_time,
        0,
        _10min,
        # 2 * _10min,
        # 3 * _10min,
        # 4 * _10min,
        # 7 * _10min,
    ]
    rates = [
        0,
        10,
        # 15,
        # 20,
        # 25,
        # 0,
    ]
    rates = [r / 60 for r in rates]  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    time_steps = [
        initialization_time / 3,
        _1min,
        # _1min,
        # _1min,
        # _1min,
        # 3 * _1min,
    ]
    time_step_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)

    return injection_protocol, time_step_protocol
