import logging
from pathlib import Path

import porepy as pp
import numpy as np

from GTS import ISCBiotContactMechanics, BiotParameters
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import TimeParameters, NewtonParameters, TimeMachine

logging.basicConfig(level=logging.INFO)


def validation():

    time_params = TimeParameters(
        end_time=10*pp.MINUTE,
        time_step=1*pp.MINUTE,
        max_time_step=1*pp.MINUTE,
        must_hit_times=[5*pp.MINUTE, 8.5*pp.MINUTE, 10*pp.MINUTE]
    )
    newton_params = NewtonParameters(convergence_tol=1e-6)
    sz = 6
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=12.8,
        scalar_scale=10*pp.GIGA,
        base=Path("~/mastersproject-data"),
        head="validation_example",
        time_step=time_params.time_step,
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
        injection_rate=10/60,  # 10 l/min
        frac_transmissivity=[1e-9, 3.7e-7],
    )
    setup = ISCBiotContactMechanics(biot_params)

    time_machine = TimeMachine(setup, newton_params, time_params)
