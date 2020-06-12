import GTS as gts
from GTS.isc_modelling.setup import run_abstract_model, _prepare_params
import porepy as pp
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def test_conditioning(*, length_scales, scalar_scales, **kwargs):
    """ Run the 'test_condition_number' for various scaling parameters

    Parameters
    ----------
    length_scales, scalar_scales : list
        lists of length- and scalar scales to test.
    **kwargs :
        Recommended to adjust:
            sz : size of mesh elements (uniform)
            shearzone_names : which shearzones (if any) to include
    """

    # Initialize empty DataFrame for storage
    results = pd.DataFrame(columns=["ls", "ss", "max_elem", "max_A_sum", "min_A_sum"])

    # loop through all scaling coefficients
    for ls in length_scales:
        for ss in scalar_scales:
            max_elem, max_A_sum, min_A_sum = test_condition_number(
                ls=ls, ss=ss, **kwargs
            )
            v = {
                "ls": ls,
                "ss": ss,
                "max_elem": max_elem,
                "max_A_sum": max_A_sum,
                "min_A_sum": min_A_sum,
            }
            results = results.append(v, ignore_index=True)

    results["ratio"] = results["max_A_sum"] / results["min_A_sum"]
    return results


def test_condition_number(**kwargs):
    # TODO: Rename this method (and dependent notebooks usages) to something more suitable
    """ Method to create a mesh and discretize equations.

    Parameters
    ----------
    **kwargs :
        pass non-default arguments.

    Returns
    -------
    max_elem, max_A_sum, min_A_sum : float
        quantitative estimates for condition number
    """
    ls = kwargs.get("ls", 1)
    ss = kwargs.get("ss", 1)

    params = make_params_for_scaling(**kwargs)

    setup = gts.ContactMechanicsBiotISC(params)
    setup.prepare_simulation()

    logger.info(f"ls= {ls:.3e}, ss= {ss:.3e}")
    return report_condition_number(setup)


def report_condition_number(setup):
    """ Extract the estimated condition number for a given setup"""

    A, b = setup.assembler.assemble_matrix_rhs()
    logger.info("Max element in A {0:.2e}".format(np.max(np.abs(A))))
    logger.info(
        "Max {0:.2e} and min {1:.2e} A sum.".format(
            np.max(np.sum(np.abs(A), axis=1)), np.min(np.sum(np.abs(A), axis=1))
        )
    )
    max_elem = np.max(np.abs(A))
    max_A_sum = np.max(np.sum(np.abs(A), axis=1))
    min_A_sum = np.min(np.sum(np.abs(A), axis=1))

    return max_elem, max_A_sum, min_A_sum


def make_params_for_scaling(**kwargs):
    """ Wrapper to _prepare_params for typical setups these tests will use

    You should set:
    ls, ss : scaling
    sz : uniform characteristic size of mesh elements
    shearzone_names : which shearzones to include.
    """

    # Get some common args
    sz = kwargs.get("sz", 80)
    ls = kwargs.get("ls", 1)
    ss = kwargs.get("ss", 1)
    sz_names = kwargs.get(
        "shearzone_names", None
    )  # ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]

    params = {
        "length_scale": ls,
        "scalar_scale": ss,
        "shearzone_names": sz_names,
        "mesh_args": {
            "mesh_size_frac": sz,
            "mesh_size_min": sz,
            "mesh_size_bound": sz,
        },
        # turn off gravity
        "_gravity_bc_p": False,
        "_gravity_src": False,
        "_gravity_bc": False,
        "path_head": kwargs.get("path_head", "test_fracture_complexity/test_1"),
    }

    params = _prepare_params(params=params, setup_loggers=False)

    return params
