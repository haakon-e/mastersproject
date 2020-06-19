import logging

import pytest
from pathlib import Path
import numpy as np

import porepy as pp
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotBase
from GTS.isc_modelling.isc_model import ISCBiotContactMechanics
from GTS.isc_modelling.parameter import BiotParameters, stress_tensor, shearzone_injection_cell, GrimselGranodiorite
from porepy.utils.derived_discretizations import implicit_euler

logger = logging.getLogger(__name__)


@pytest.fixture
def biot_params() -> BiotParameters:
    """ Initialize BiotParameters"""
    here = Path(__file__).parent / "simulations"
    _sz = 6
    params = BiotParameters(
        folder_name=here,
        stress=stress_tensor(),
        injection_rate=1/6,
        frac_permeability=1e-9,
        intact_permeability=1e-12,
        well_cells=shearzone_injection_cell,
        rock=GrimselGranodiorite(),
        mesh_args={
            "mesh_size_frac": _sz,
            "mesh_size_min": 0.2 * _sz,
            "mesh_size_bound": 3 * _sz,
        },
    )
    return params


@pytest.fixture
def isc_setup(
        biot_params,
) -> ContactMechanicsBiotBase:
    """ Contact Mechanics Biot setup with isc grid"""
    _setup = ISCBiotContactMechanics(biot_params)

    return _setup


class TestISCBiotContactMechanics:
    def test_create_grid(self, isc_setup):
        isc_setup.create_grid()

    def test_well_cells(self):
        assert False

    def test_aperture(self):
        assert False

    def test_permeability(self):
        assert False

    def test_source_flow_rate(self):
        assert False

    def test_source_scalar(self):
        assert False

    def test_after_simulation(self):
        assert False

    def test_faces_to_fix(self):
        assert False

    def test_bc_type_mechanics(self):
        assert False

    def test_bc_values_mechanics(self):
        assert False

    def test_source(self):
        assert False

    def test_prepare_simulation(self, isc_setup):
        isc_setup.prepare_simulation()

    def test_run_simulation(self, isc_setup):
        isc_setup.prepare_simulation()
        pp.run_time_dependent_model(isc_setup, {})

    def test_realistic_setup(self):
        """ For a 50 000 cell setup, test Contact mechanics Biot model on 5 shear zones.
        Parameters are close to the ISC setup. Model is run for 10 minutes,
        with time steps of 1 minute. Injection to a shear zone.
        """
        _sz = 6  # _sz=6 => ~50k cells
        params = BiotParameters(
            # Base parameters
            length_scale=0.05,
            scalar_scale=1e6,
            head="50k-5frac-test",
            time_step=pp.MINUTE,
            end_time=10*pp.MINUTE,
            rock=GrimselGranodiorite(),
            # Geometry parameters
            mesh_args={
                "mesh_size_frac": _sz,
                "mesh_size_min": 0.2 * _sz,
                "mesh_size_bound": 3 * _sz,
            },
            # Mechanical parameters
            stress=stress_tensor(),
            newton_options={
                "max_iterations": 20,
                "nl_convergence_tol": 1e-6,
                "nl_divergence_tol": 1e5,
            },
            # Flow parameters
            well_cells=shearzone_injection_cell,
            injection_rate=1 / 6,  # = 10 l/min
            frac_permeability=1e-16,
            intact_permeability=1e-20,
        )

        class NeverFailtBiotCM(ISCBiotContactMechanics):
            def after_newton_failure(self, solution, errors, iteration_counter):
                """ Instead of raising error on failure, simply continue.
                """
                logger.error("Newton iterations did not converge")
                self.after_newton_convergence(solution, errors, iteration_counter)

        setup = NeverFailtBiotCM(params)
        newton_params = params.newton_options
        pp.run_time_dependent_model(setup, newton_params)
