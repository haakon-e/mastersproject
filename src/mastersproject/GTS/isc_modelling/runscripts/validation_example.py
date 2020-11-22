import logging
from pathlib import Path
from typing import Tuple

import porepy as pp
import numpy as np
from GTS import BiotParameters, ISCBiotContactMechanics, stress_tensor
from GTS.isc_modelling.parameter import UnitRock, GrimselGranodiorite
from GTS.time_machine import NewtonParameters, TimeMachinePhasesConstantDt
from GTS.time_protocols import InjectionRateProtocol, TimeStepProtocol

logging.basicConfig(level=logging.INFO)


def simple_validation():
    """ Validation on Ivar setup"""
    # Grid
    gb, box, mesh_args = create_grid()

    # Injection phases and time configuration
    start_time = -1e2 * pp.YEAR
    end_time = 15 * pp.YEAR
    phase_limits = [start_time, 0, 10 * 3600, end_time]
    rates = [0, 75, 20]
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    phase_time_steps = [-start_time, 2.0, 2 / 3 * pp.HOUR]
    time_params = TimeStepProtocol.create_protocol(phase_limits, phase_time_steps)

    # Newton
    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=15,
    )

    # Model parameters
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=15,
        scalar_scale=1e9,
        base=Path("/home/haakonervik/mastersproject-data"),
        head="validation_example",
        time=time_params.start_time,
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        gravity=True,
        rock=IvarGranite(),
        # GeometryParameters
        shearzone_names=["f1", "f2", "f3"],
        box=box,
        mesh_args=mesh_args,
        # MechanicsParameters
        stress=stress_tensor(),
        dilation_angle=np.radians(5.0),
        newton_options=newton_params.dict(),
        # FlowParameters
        well_cells=_tag_ivar_well_cells,
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


def create_grid() -> Tuple[pp.Grid, dict, dict]:
    """ Ivar ex3 grid"""
    # Define the three fractures
    n_points = 16

    # Injection
    f_1 = pp.EllipticFracture(
        np.array([10, 3.5, -3]),
        11,
        18,
        0.5,
        0,
        0,
        num_points=n_points,
    )
    f_2 = pp.EllipticFracture(
        np.array([1, 5, 1]),
        15,
        10,
        np.pi * 0,
        np.pi / 4.0,
        np.pi / 2.0,
        num_points=n_points,
    )
    # Production
    f_3 = pp.EllipticFracture(
        np.array([-13, 0, 0]),
        20,
        10,
        0.5,
        np.pi / 3,
        np.pi / 1.6,
        num_points=n_points,
    )
    fractures = [f_1, f_2, f_3]

    # Define the domain
    size = 50
    box = {
        "xmin": -size,
        "xmax": size,
        "ymin": -size,
        "ymax": size,
        "zmin": -size,
        "zmax": size,
    }
    mesh_size = 3.2
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.5 * mesh_size,
        "mesh_size_bound": 3 * mesh_size,
    }
    # Make a fracture network
    network = pp.FractureNetwork3d(fractures, domain=box)
    # Generate the mixed-dimensional mesh
    gb = network.mesh(mesh_args)
    return gb, box, mesh_args


def _tag_ivar_well_cells(_, gb: pp.GridBucket) -> None:
    """
    Tag well cells with unitary values, positive for injection cells and negative
    for production cells.
    """
    box = gb.bounding_box(as_dict=True)
    nd = gb.dim_max()
    for g, d in gb:
        tags = np.zeros(g.num_cells)
        if g.dim < nd:
            point = np.array(
                [
                    [(box["xmin"] + box["xmax"]) / 2],
                    [box["ymax"]],
                    [0],
                ]
            )
            distances = pp.distances.point_pointset(point, g.cell_centers)
            indexes = np.argsort(distances)
            if d["node_number"] == 1:
                tags[indexes[-1]] = 1  # injection
            elif d["node_number"] == 3:
                tags[indexes[-1]] = -1  # production
                # write_well_cell_to_csv(g, indexes[-1], self)
        g.tags["well_cells"] = tags
        pp.set_state(d, {"well": tags.copy()})


class IvarGranite(UnitRock):
    PERMEABILITY = 1e-15
    DENSITY = 2700.0 * pp.KILOGRAM / pp.METER ** 3
    POROSITY = 0.01

    YOUNG_MODULUS = 40.0 * pp.GIGA * pp.PASCAL
    POISSON_RATIO = 0.2
    FRICTION_COEFFICIENT = 0.5
