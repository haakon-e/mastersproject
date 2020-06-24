import pytest
from pathlib import Path
import numpy as np

import porepy as pp
from GTS.isc_modelling.contact_mechanics_biot import ContactMechanicsBiotBase
from GTS.isc_modelling.mechanics import Mechanics
from GTS.isc_modelling.flow import Flow
from GTS.isc_modelling.parameter import BaseParameters
from GTS.test.standard_grids import two_intersecting_blocking_fractures
from porepy.utils.derived_discretizations import implicit_euler


@pytest.fixture
def base_params() -> BaseParameters:
    """ Initialize BaseParameters with folder_name pointing here."""
    here = Path(__file__).parent
    params = BaseParameters(
        folder_name=here,
    )
    return params


@pytest.fixture
def setup(
        base_params,
) -> ContactMechanicsBiotBase:
    """ Contact Mechanics Biot Base setup with simple fractured grid"""
    _setup = ContactMechanicsBiotBase(base_params)

    # create grid
    _setup.gb = two_intersecting_blocking_fractures(_setup.params.folder_name)
    _setup.bounding_box = _setup.gb.bounding_box(as_dict=True)
    pp.contact_conditions.set_projections(_setup.gb)

    return _setup


class TestContactMechanicsBiotBase:
    def test_set_biot_parameters(self, setup):
        setup.set_biot_parameters()

        key_s = setup.scalar_parameter_key
        key_m = setup.mechanics_parameter_key
        gb = setup.gb

        for g, d in gb:
            params_s = d[pp.PARAMETERS][key_s]
            params_m = d[pp.PARAMETERS][key_m]

            # Check scalar parameters
            scalar_param_keys = [
                "second_order_tensor",
                "bc",
                "bc_values",
                "mass_weight",
                "source",
                "time_step",
                "biot_alpha"
            ]
            for p in scalar_param_keys:
                assert p in params_s

            # Check mechanics parameters
            assert "biot_alpha" in params_m
            if g.dim == setup.Nd:
                p_list = ["bc", "bc_values", "source", "fourth_order_tensor"]
                for p in p_list:
                    assert p in params_m
            elif g.dim == setup.Nd - 1:
                assert "friction_coefficient" in params_m

        for e, d in gb.edges():
            params = d[pp.PARAMETERS]
            assert key_m in params
            assert "normal_diffusivity" in params[key_s]

    def test_assign_biot_variables(self, setup):
        setup.assign_biot_variables()

        gb = setup.gb
        nd = setup.Nd
        primary_vars = pp.PRIMARY_VARIABLES
        var_s = setup.scalar_variable
        var_mortar_s = setup.mortar_scalar_variable
        var_m = setup.displacement_variable
        var_contact = setup.contact_traction_variable
        var_mortar_m = setup.mortar_displacement_variable

        for g, d in gb:
            assert primary_vars in d

            # flow
            assert var_s in d[primary_vars]
            assert d[primary_vars][var_s]["cells"] == 1

            # mechanics
            if g.dim == nd:
                assert var_m in d[primary_vars]
                assert d[primary_vars][var_m]["cells"] == nd
            elif g.dim == nd - 1:
                assert var_contact in d[primary_vars]
                assert d[primary_vars][var_contact]["cells"] == nd

        for e, d in gb.edges():
            assert primary_vars in d
            assert var_mortar_s in d[primary_vars]
            _, g_h = gb.nodes_of_edge(e)
            if g_h.dim == nd:
                assert var_mortar_m in d[primary_vars]

    def test_assign_biot_discretizations(self, setup):
        setup.assign_biot_discretizations()

        key_s, key_m = setup.scalar_parameter_key, setup.mechanics_parameter_key
        var_s, var_m = setup.scalar_variable, setup.displacement_variable
        var_contact = setup.contact_traction_variable
        discr_key, coupling_discr_key = pp.DISCRETIZATION, pp.COUPLING_DISCRETIZATION
        gb = setup.gb
        nd = setup.Nd

        for g, d in gb:
            discr = d[discr_key]
            # flow
            discr_s = discr[var_s]
            assert isinstance(discr_s["diffusion"], implicit_euler.ImplicitMpfa)
            assert isinstance(discr_s["mass"], implicit_euler.ImplicitMassMatrix)
            assert isinstance(discr_s["source"], pp.ScalarSource)

            # biot coupling
            if g.dim == nd:
                assert isinstance(discr_s["stabilization"], pp.BiotStabilization)
                assert isinstance(discr[var_m + "_" + var_s]["grad_p"], pp.GradP)
                assert isinstance(discr[var_s + "_" + var_m]["div_u"], pp.DivU)

            # mechanics
            if g.dim == nd:
                discr_m = discr[var_m]
                assert isinstance(discr_m["mpsa"], pp.Mpsa)
            elif g.dim == nd - 1:
                discr_contact = d[discr_key][var_contact]
                assert isinstance(discr_contact["empty"], pp.VoidDiscretization)

        # biot coupling for mixed-dimensional problems
        for e, d in gb.edges():
            cdiscr = d[coupling_discr_key]
            g_l, g_h = gb.nodes_of_edge(e)
            if g_h.dim == nd:
                div_u = cdiscr["div_u_coupling"]
                assert isinstance(div_u[e][1], pp.DivUCoupling)
                grad_p_matrix = cdiscr["matrix_scalar_to_force_balance"]
                assert isinstance(grad_p_matrix[e][1], pp.MatrixScalarToForceBalance)

                if setup.subtract_fracture_pressure:
                    grad_p_fracture = cdiscr["fracture_scalar_to_force_balance"]
                    assert isinstance(grad_p_fracture[e][1], pp.FractureScalarToForceBalance)

    def test_discretize(self):
        assert False

    def test_initial_biot_condition(self, setup):
        """ Test for zero initial conditions.
        Will fail if gravity is introduced
        """
        setup.set_biot_parameters()
        setup.initial_biot_condition()
        gb = setup.gb
        nd = setup.Nd
        var_s = setup.scalar_variable
        var_m = setup.displacement_variable
        var_contact = setup.contact_traction_variable
        var_mortar_m = setup.mortar_displacement_variable
        var_mortar_s = setup.mortar_scalar_variable

        for g, d in gb:
            # flow
            scalar = d[pp.STATE][var_s]
            assert np.allclose(scalar, np.zeros(g.num_cells))

            # mechanics
            if g.dim == nd:
                displacement = d[pp.STATE][var_m]
                assert np.allclose(displacement, np.zeros(g.num_cells * nd))

            elif g.dim == nd - 1:
                traction = d[pp.STATE][var_contact]
                prev_traction = d[pp.STATE]["previous_iterate"][var_contact]
                init_traction = np.vstack(
                    (np.zeros((g.dim, g.num_cells)), -1 * np.ones(g.num_cells))
                ).ravel(order="F")
                assert np.allclose(traction, init_traction)
                assert np.allclose(prev_traction, init_traction)

        for e, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            # flow
            scalar = d[pp.STATE][var_mortar_s]
            assert np.allclose(scalar, np.zeros(mg.num_cells))

            # mechanics
            if mg.dim == nd - 1:
                mortar_m = d[pp.STATE][var_mortar_m]
                prev_mortar_m = d[pp.STATE]["previous_iterate"][var_mortar_m]
                assert np.allclose(mortar_m, np.zeros(mg.num_cells * nd))
                assert np.allclose(prev_mortar_m, np.zeros(mg.num_cells * nd))

    def test_check_convergence_call_parents(self, setup, mocker):
        mocker.patch(
            "GTS.isc_modelling.flow.Flow.check_convergence",
            return_value=(0, True, False)
        )
        mocker.patch(
            "GTS.isc_modelling.mechanics.Mechanics.check_convergence",
            return_value=(0, True, False)
        )
        arr = np.zeros(setup.gb.num_cells())
        setup.check_convergence(arr, arr, arr, {})

        Flow.check_convergence.assert_called_once_with(arr, arr, arr, {})
        Mechanics.check_convergence.assert_called_once_with(arr, arr, arr, {})

    def test_before_newton_loop(self, ):
        assert False

    def test_before_newton_iteration(self):
        assert False

    def test_after_newton_iteration(self):
        assert False

    def test_set_viz(self, setup):
        setup.set_viz()

    def test_export_step(self):
        assert False

    def test_reconstruct_stress(self):
        assert False

    def test_prepare_simulation(self, setup):
        setup.prepare_simulation()

    def test_run_simulation(self, setup):
        setup.prepare_simulation()
        pp.run_time_dependent_model(setup, {})
