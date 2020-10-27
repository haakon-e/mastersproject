import logging

import numpy as np

import GTS as gts
import porepy as pp
from mastersproject.util.logging_util import trace

logger = logging.getLogger(__name__)


@trace(logger=logger)
def test_unit_convergence_study():
    """ Unit test for convergence study

    Simple prototype-like setup.
    """

    # 1. Prepare parameters
    stress = gts.stress_tensor()
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    params = {
        "mesh_args": {
            "mesh_size_frac": 10,
            "mesh_size_min": 0.1 * 10,
            "mesh_size_bound": 6 * 10,
        },
        "stress": stress,
        "shearzone_names": None,  # ['S1_1'],
    }

    n_refinements = 2

    # Run ContactMechanicsISC model
    gb_list, errors = gts.isc_modelling.setup.run_models_for_convergence_study(
        model=gts.ContactMechanicsISC,
        run_model_method=pp.run_stationary_model,
        params=params,
        n_refinements=n_refinements,
        variable=["u"],
        variable_dof=[3],
    )

    return gb_list, errors


@trace(logger=logger)
def test_unit_biot_convergence_study():
    """ Unit test for convergence study

    -- Key setup
    * Full biot equations
    *
    """

    # 1. Prepare parameters
    stress = gts.isc_modelling.stress_tensor()
    hydrostatic = np.mean(np.diag(stress)) * np.ones(stress.shape[0])
    stress = np.diag(hydrostatic)

    sz = 40
    params = {
        "mesh_args": {
            "mesh_size_frac": sz,
            "mesh_size_min": 0.1 * sz,
            "mesh_size_bound": 6 * sz,
        },
        "stress": stress,
        "shearzone_names": ["S1_1"],  # None,
        "scalar_scale": 1 * pp.GIGA,
    }

    n_refinements = 2

    # Run ContactMechanicsISC model
    gb_list, errors = gts.isc_modelling.setup.run_models_for_convergence_study(
        model=gts.ContactMechanicsBiotISC,
        run_model_method=pp.run_time_dependent_model,
        params=params,
        n_refinements=n_refinements,
        variable=["p", "p_exp", "u", "u_exp"],
        variable_dof=[1, 1, 3, 3],
    )

    return gb_list, errors
