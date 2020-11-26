from pathlib import Path

import numpy as np
import porepy as pp
import pytest

from GTS.isc_modelling.isc_model import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import (
    BiotParameters,
    FlowParameters,
    GrimselGranodiorite,
    nd_sides_shearzone_injection_cell,
    shearzone_injection_cell,
    stress_tensor,
)


class TestFlowParameters:
    def test_validate_source_scalar_borehole_shearzone(self):
        assert False

    def test_set_fracture_aperture_from_cubic_law(self, tmpdir):
        shearzones = ["S1_1", "S1_2"]
        params = FlowParameters(
            # BaseParameters
            folder_name=tmpdir,
            # GeometryParameters
            shearzone_names=shearzones,
            # FlowParameters
            injection_rate=1,
            frac_transmissivity=1,
        )
        # Check that correct Transmissivity is calculated
        res = np.cbrt(params.mu_over_rho_g * 12)
        apertures = np.array(
            [params.initial_background_aperture(sz) for sz in shearzones]
        )
        assert np.allclose(res, apertures)
        assert np.isclose(params.mu_over_rho_g, 1 / params.rho_g_over_mu)

        # Another test
        params.frac_transmissivity = params.rho_g_over_mu / 12
        apertures = np.array(
            [params.initial_background_aperture(sz) for sz in shearzones]
        )
        assert np.allclose(1, apertures)

    def test_set_increased_aperture_near_injection_point(self, tmpdir, mocker):
        shearzones = ["S1_2"]
        near_inj_T = 1
        params = FlowParameters(
            # BaseParameters
            folder_name=tmpdir,
            # GeometryParameters
            shearzone_names=shearzones,
            # FlowParameters
            injection_rate=1,
            frac_transmissivity=0,
            near_injection_transmissivity=near_inj_T,
            near_injection_t_radius=1,
        )
        g = pp.CartGrid(nx=[3, 3])
        g.compute_geometry()

        # Mock the borehole shearzone intersection method
        method = mocker.patch(
            "GTS.isc_modelling.parameter.shearzone_borehole_intersection"
        )

        # Assert we get 4 cells of T=1 each
        method.return_value = np.array([1.5, 1.5, 0])
        aperture = params.compute_initial_aperture(g, shearzones[0])
        known_aperture = params.b_from_T(near_inj_T)
        assert np.isclose(np.sum(aperture), known_aperture * 5)

        # Assert the locations of these cells
        known_idx = [1, 3, 4, 5, 7]
        ap_idx = np.where(np.isclose(aperture, known_aperture))[0]
        assert np.allclose(np.sort(ap_idx), known_idx)

        # Assert no transmissivity added if no cells within the radius is found
        params.near_injection_t_radius = 0.4
        method.return_value = np.array([2, 2, 0])
        aperture = params.compute_initial_aperture(g, shearzones[0])
        assert np.sum(aperture) == 0


def test_nd_sides_shearzone_injection_cell():
    """ Test that the 2 Nd-cells are tagged correctly"""
    # --- Paste biot_params_small
    here = Path(__file__).parent / "simulations"

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
    biot_params_small["fractures"] = [frac_name]
    params = BiotParameters(**biot_params_small)
    setup = ISCBiotContactMechanics(params)
    setup.create_grid()
    gb = setup.gb

    # Tag the cells
    nd_sides_shearzone_injection_cell(params, gb)

    for g, d in gb:
        tags = d["well"]
        if g.dim == setup.Nd:
            assert np.sum(np.abs(tags)) == np.sum(tags) == 2
        else:
            assert np.sum(tags) == np.sum(np.abs(tags)) == 0
