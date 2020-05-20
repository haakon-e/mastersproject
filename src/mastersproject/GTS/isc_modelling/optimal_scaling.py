""" method to find optimal scaling of a problem """
import time

import numpy as np
import pandas as pd
import porepy as pp
import scipy.optimize as optimize

import GTS as gts
from GTS import FlowISC, prepare_directories, SetupParams


def best_cond_numb(initial_guess: np.array = None):
    """ Find best condition numbers

    initial_guess = (length_scale, log10(scalar_scale))
        if not set, then default is length_scale=0.05, scalar_scale=1e+9.
    """
    if not initial_guess:
        initial_guess = np.array([0.05, 9])
    ls_0, log_ss_0 = initial_guess
    assert log_ss_0 >= 2, "log10 scalar_scale < 2 results in negative test values."

    length_scales = np.array((1 / 4, 1 / 2, 1, 2, 4)) * ls_0
    log_scalar_scales = np.array((-2, -1, 0, 1, 2)) + log_ss_0

    results = pd.DataFrame(columns=["ls", "log_ss", "cond_pp", "cond_umfpack"])
    for ls in length_scales:
        for log_ss in log_scalar_scales:
            try:
                A = assemble_isc_matrix([ls, log_ss])
                cond_pp = condition_number_porepy(A)
                cond_umfpack = condition_number_umfpack(A)
            except ValueError:
                cond_pp = "singular"
                cond_umfpack = "singular"
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
    scalar_scale = 10 ** log_scalar_scale

    _sz = 40
    mesh_args = {
        "mesh_size_frac": _sz,
        "mesh_size_min": 0.2 * _sz,
        "mesh_size_bound": 3 * _sz,
    }

    folder = prepare_directories(
        f"test_optimal_scaling/ls{length_scale:.2e}_ss{scalar_scale:.2e}"
    )
    params = SetupParams(
        folder_name=folder,
        length_scale=length_scale,
        scalar_scale=scalar_scale,
        mesh_args=mesh_args,
    )
    setup = FlowISC(params.dict())
    setup.prepare_simulation()

    A, _ = setup.assembler.assemble_matrix_rhs()
    return A


def condition_number_porepy(A):
    row_sum = np.sum(np.abs(A), axis=1)
    return np.max(row_sum) / np.min(row_sum)


def condition_number_umfpack(A):
    diag = np.abs(A.diagonal())
    return np.max(diag) / np.min(diag)
