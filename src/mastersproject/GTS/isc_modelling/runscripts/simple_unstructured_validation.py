from pathlib import Path

import porepy as pp
import numpy as np
from GTS import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import (
    nd_injection_cell_center,
    GrimselGranodiorite,
    BiotParameters,
)
from GTS.time_machine import TimeMachinePhasesConstantDt, NewtonParameters
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol


def simple_validation():
    """ Validation on easy setup"""
    path = Path(__file__).parent / "results"
    # Grid
    mesh_size = 20
    gb, box, mesh_args = two_intersecting_blocking_fractures(str(path), mesh_size)

    # Injection phases and time configuration
    start_time = -1e2 * pp.YEAR
    phase_limits = [start_time, 0, 12 * pp.HOUR]
    rates = [0, 0]
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    phase_time_steps = [-start_time, 2.0 * pp.MINUTE]
    time_params = TimeStepProtocol.create_protocol(phase_limits, phase_time_steps)

    # Newton
    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=50,
    )

    rock = GrimselGranodiorite()
    rock.FRICTION_COEFFICIENT = 0.2
    stress = np.diag(-np.array([6, 13.1, 6]) * pp.MEGA * pp.PASCAL)
    # Model parameters
    biot_params = BiotParameters(
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
        fractures=["f1", "f2"],
        box=box,
        mesh_args=mesh_args,
        # MechanicsParameters
        stress=stress,
        dilation_angle=np.radians(5.0),
        newton_options=newton_params.dict(),
        # FlowParameters
        well_cells=nd_injection_cell_center,
        injection_protocol=injection_protocol,
        frac_transmissivity=5.17e-3,  # Gives a0=2e-3, which are Ivar's values.
        # BiotParameters
        alpha=0.8,
    )
    setup = ISCBiotContactMechanics(biot_params)
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

    gb = frac_network.mesh(mesh_args, file_name=folder_name + "/gmsh_frac_file")
    return gb, domain, mesh_args
