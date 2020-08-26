""" method to find optimal scaling of a problem """
from pathlib import Path

import numpy as np
import pandas as pd

from GTS import (
    BiotParameters,
    GrimselGranodiorite,
    ISCBiotContactMechanics,
    stress_tensor,
)


def best_cond_numb(assemble_A_method, initial_guess: np.array = None,) -> pd.DataFrame:
    """ Find best condition numbers

    assemble_A_method is a method that assembles a matrix

    initial_guess = (length_scale, log10(scalar_scale))
        if not set, then default is length_scale=0.05, scalar_scale=1e+9.
    """
    if initial_guess is None:
        initial_guess = np.array([0.05, 6])
    ls_0, log_ss_0 = initial_guess

    length_scales = np.array((1 / 4, 1 / 2, 1, 2, 4)) * ls_0
    log_scalar_scales = np.array((-2, -1, 0, 1, 2)) + log_ss_0

    results = pd.DataFrame(columns=["ls", "log_ss", "cond_pp", "cond_umfpack"])
    for ls in length_scales:
        for log_ss in log_scalar_scales:
            A = assemble_A_method(np.array([ls, log_ss]))
            try:
                cond_pp = condition_number_porepy(A)
                cond_umfpack = condition_number_umfpack(A)
            except ValueError as e:
                cond_pp = np.nan
                cond_umfpack = np.nan
                print(e)
            v = {
                "ls": ls,
                "log_ss": log_ss,
                "cond_pp": cond_pp,
                "cond_umfpack": cond_umfpack,
            }
            results = results.append(v, ignore_index=True)

    return results


def assemble_isc_matrix(values):
    """ Optimize the isc setup

    values = (length_scale, log10(scalar_scale))

    if return_both is set, return both estimates for condition number.
    """
    length_scale, log_scalar_scale = values  # Component values of input
    scalar_scale = np.float_power(10, log_scalar_scale)

    _sz = 40
    mesh_args = {
        "mesh_size_frac": _sz,
        "mesh_size_min": 0.2 * _sz,
        "mesh_size_bound": 3 * _sz,
    }
    here = (
        Path(__file__).parent
        / f"results/test_optimal_scaling/ls{length_scale:.2e}_ss{scalar_scale:.2e}"
    )
    params = BiotParameters(
        # Base
        length_scale=length_scale,
        scalar_scale=scalar_scale,
        folder_name=here,
        time_step=1 * 60,  # 1 minute
        rock=GrimselGranodiorite(),
        # Geometry
        shearzone_names=["S1_2", "S3_1"],
        mesh_args=mesh_args,
        # Mechanics
        stress=stress_tensor(),
        dilation_angle=(np.pi / 180) * 5,
        # Flow
        frac_transmissivity=[1e-9, 3.7e-7],
    )
    setup = ISCBiotContactMechanics(params)
    setup.prepare_simulation()

    A, _ = setup.assembler.assemble_matrix_rhs()
    return A


def condition_number_porepy(A):
    row_sum = np.sum(np.abs(A), axis=1)
    return np.max(row_sum) / np.min(row_sum)


def condition_number_umfpack(A):
    diag = np.abs(A.diagonal())
    cond = np.min(diag) / np.max(diag)
    if np.isclose(cond, 0):
        return np.inf
    else:
        return 1 / cond
