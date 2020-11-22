from pathlib import Path

import porepy as pp
import numpy as np
from GTS import FlowISC
from GTS.isc_modelling.parameter import (
    GrimselGranodiorite,
    FlowParameters, center_of_shearzone_injection_cell,
)
from GTS.time_machine import TimeMachinePhasesConstantDt, NewtonParameters
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol


def simple_example():
    """ Simple flow example"""
    path = Path(__file__).parent / "results"
    # Grid
    mesh_size = 80
    gb, box, mesh_args = two_intersecting_blocking_fractures(path, mesh_size)

    # Injection phases and time configuration
    phase_limits = [0, 10 * pp.MINUTE, 4 * pp.HOUR]
    rates = [1/6, 0]  # Injection rates, [l/s]
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    phase_time_steps = [2.0 * pp.MINUTE, 1 * pp.HOUR]
    time_params = TimeStepProtocol.create_protocol(phase_limits, phase_time_steps)

    # Newton
    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=50,
    )

    rock = GrimselGranodiorite()
    # Model parameters
    flow_params = FlowParameters(
        # BaseParameters
        length_scale=15,
        scalar_scale=1e9,
        folder_name=path,
        time=time_params.start_time,
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        gravity=False,
        rock=rock,
        # GeometryParameters
        shearzone_names=["f1", "f2"],
        box=box,
        mesh_args=mesh_args,
        # FlowParameters
        source_scalar_borehole_shearzone={
            "shearzone": "f1",
            "borehole": 'NaN',
        },
        well_cells=center_of_shearzone_injection_cell,
        injection_protocol=injection_protocol,
        frac_transmissivity=5.17e-3,  # Gives a0=2e-3, which are Ivar's values.
    )
    setup = FlowISC(flow_params)
    setup.gb = gb

    time_machine = TimeMachinePhasesConstantDt(setup, newton_params, time_params)

    time_machine.run_simulation()
    return time_machine


def two_intersecting_blocking_fractures(folder_name, mesh_size):
    """ Domain with fully blocking fracture """
    # fmt: off
    domain = {
        'xmin': 0, 'ymin': 0, 'zmin': 0,
        'xmax': 300, 'ymax': 300, 'zmax': 300
    }

    frac_pts1 = np.array(
        [[50, 50, 250, 250],
         [0, 300, 300, 0],
         [0, 0, 300, 300]])
    frac1 = pp.Fracture(frac_pts1)
    frac_pts2 = np.array(
        [[300, 300, 50, 50],
         [0, 300, 300, 0],
         [50, 50, 300, 300]])
    # fmt: on
    frac2 = pp.Fracture(frac_pts2)

    frac_network = pp.FractureNetwork3d([frac1, frac2], domain)
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 2 / 3 * mesh_size,
        "mesh_size_bound": 2 * mesh_size,
    }

    folder_name.mkdir(parents=True, exist_ok=True)
    gb = frac_network.mesh(mesh_args, file_name=str(folder_name / "gmsh_frac_file"))
    return gb, domain, mesh_args
