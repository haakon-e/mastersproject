import porepy as pp
import numpy as np
from GTS import BiotParameters
from GTS.isc_modelling.isc_box_model import ISCBoxModel
from GTS.time_machine import NewtonParameters
from GTS.time_protocols import InjectionRateProtocol, TimeStepProtocol
import pytest


def create_protocols():
    """ Create a test injection protocol"""
    _1min = pp.MINUTE
    phase_limits = [
        0,
        50 * pp.MINUTE,
        100 * pp.MINUTE,
    ]
    rates = [
        0,
        10 / 60,
    ]
    inj_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)
    time_steps = [
        10 * pp.MINUTE,
        20 * pp.MINUTE,
    ]
    dt_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)
    return inj_protocol, dt_protocol


@pytest.fixture(scope="session")
def params(tmpdir_factory):
    inj, dt = create_protocols()
    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=10,
    )
    path = tmpdir_factory.mktemp("results")
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=0.04,
        scalar_scale=1e8,
        folder_name=path,
        time_step=dt.initial_time_step,
        end_time=dt.end_time,
        # GeometryParameters
        shearzone_names=["S1_1", "S1_2", "S1_3", "S3_1"],
        # MechanicsParameters
        newton_options=newton_params,
        # FlowParameters
        frac_transmissivity=1e-8,
    )
    return biot_params, newton_params, dt


@pytest.fixture(scope="function")
def create_setup(params):
    biot, newton, dt = params
    setup = ISCBoxModel(biot, lcin=5 * 10, lcout=50 * 10)
    return setup


def test_tag_tunnel_cells(create_setup):
    # Prepare setup
    setup = create_setup
    setup.create_grid(n_optimize=0, use_logger=False)

    # Get data on tunnels
    strcts = setup.params.isc_data.structures
    _mask = strcts["borehole"].isin(["AU", "VE"]) & strcts["shearzone"].isin(
        setup.params.shearzone_names
    )
    data = strcts[_mask]
    data = data[["borehole", "x_gts", "y_gts", "z_gts", "shearzone"]]
    data = data.reset_index(drop=True)

    # Tag tunnel cells
    setup.tag_tunnel_cells()

    # Discover tags
    method_count = 0
    found_inds = []
    for g, d in setup.gb:
        ind = np.where(g.tags["tunnel_cells"])[0]
        found_inds.append(ind)

        method_count += np.sum(g.tags["tunnel_cells"])
    found_inds = np.hstack(found_inds)
    found_inds.sort()

    # Tag manually
    manual_inds = []
    for idx, row in data.iterrows():
        coord = (
            row[["x_gts", "y_gts", "z_gts"]].to_numpy(dtype=float).reshape((3, -1))
            / setup.params.length_scale
        )
        sz = row["shearzone"]
        g = setup.grids_by_name(sz)[0]
        ids, dsts = g.closest_cell(coord, return_distance=True)
        manual_inds.append(ids)
    # breakpoint()
    manual_inds = np.hstack(manual_inds)
    manual_inds.sort()

    # Assert equality
    tunnel_intersections_count = data.shape[0]
    assert method_count == tunnel_intersections_count
    assert np.allclose(manual_inds, found_inds)
