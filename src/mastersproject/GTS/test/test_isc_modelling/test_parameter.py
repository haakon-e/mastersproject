import pytest
from pathlib import Path

import numpy as np

from GTS.isc_modelling.isc_model import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import BiotParameters, stress_tensor, shearzone_injection_cell, GrimselGranodiorite, \
    nd_sides_shearzone_injection_cell, FlowParameters


class TestFlowParameters:
    def test_validate_source_scalar_borehole_shearzone(self):
        assert False

    def test_set_fracture_aperture_from_cubic_law(self, tmp_path):
        here = tmp_path / "simulations"
        _sz = 6
        params = FlowParameters(
            # BaseParameters
            folder_name=here,
            # GeometryParameters
            shearzone_names=["S1_1", "S1_2"],
            # FlowParameters
            injection_rate=1,
            frac_transmissivity=1,
        )
        assert params.initial_fracture_aperture is not None
        # Check that correct Transmissivity is calculated
        res = np.cbrt(params.mu_over_rho_g*12)
        aperture_list = np.array([*params.initial_fracture_aperture.values()])
        assert np.allclose(res, aperture_list)
        assert np.isclose(params.mu_over_rho_g, 1/params.rho_g_over_mu)

        # Another test
        params.frac_transmissivity = params.rho_g_over_mu/12
        aperture_list = np.array([*params.initial_fracture_aperture.values()])
        assert np.allclose(1, aperture_list)


def test_nd_sides_shearzone_injection_cell():
    """ Test that the 2 Nd-cells are tagged correctly"""
    # --- Paste biot_params_small
    here = Path(__file__).parent / "simulations"
    _sz = 6

    values = {
        "folder_name": here,
        "stress": stress_tensor(),
        "injection_rate": 1 / 6,
        "frac_permeability": 3,  # -> frac_aperture = 6.0
        "intact_permeability": 1e-12,
        "well_cells": shearzone_injection_cell,
        "rock": GrimselGranodiorite(),
        "mesh_args": {
            "mesh_size_frac": 10,
            "mesh_size_min": 10,
            "mesh_size_bound": 30,
        },
        "length_scale": 7,
        "scalar_scale": 11,
    }
    # ---
    biot_params_small = values

    frac_name = "S1_2"
    biot_params_small["shearzone_names"] = [frac_name]
    params = BiotParameters(**biot_params_small)
    setup = ISCBiotContactMechanics(params)
    setup.create_grid()
    gb = setup.gb

    # Tag the cells
    nd_sides_shearzone_injection_cell(params, gb)

    nd_grid = setup._nd_grid()
    frac = setup.grids_by_name(frac_name)

    for g, d in gb:
        tags = d["well"]
        if g.dim == setup.Nd:
            assert np.sum(np.abs(tags)) == np.sum(tags) == 2
        else:
            assert np.sum(tags) == np.sum(np.abs(tags)) == 0
