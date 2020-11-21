from typing import Tuple, List
import porepy as pp
import numpy as np
import pandas as pd
import logging

from GTS import BiotParameters, GrimselGranodiorite
from GTS.isc_modelling.isc_box_model import ISCBoxModel
from GTS.isc_modelling.parameter import shearzone_injection_cell
from GTS.time_machine import NewtonParameters
from GTS.time_protocols import TimeStepProtocol, InjectionRateProtocol

logger = logging.getLogger(__name__)


def _hs1_protocols_optimal_scaling():
    """Stimulation protocol for the rate-controlled phase of HS1"""
    _1min = pp.MINUTE
    _5min = 5 * _1min
    phase_limits = [
        0,
        1 * _5min,  # 5 min
    ]
    rates = [
        15,  # C3, step 1
    ]
    rates = [r / 60 for r in rates]  # Convert to litres / second
    injection_protocol = InjectionRateProtocol.create_protocol(phase_limits, rates)

    time_steps = [1 * _1min]
    time_step_protocol = TimeStepProtocol.create_protocol(phase_limits, time_steps)

    return injection_protocol, time_step_protocol


def prepare_params(
    length_scale,
    scalar_scale,
) -> Tuple[BiotParameters, NewtonParameters, TimeStepProtocol]:
    """Hydro shearing experiment HS1

    HS1 targeted S1.3 through INJ2 at 39.75 - 40.75 m.
    """
    tunnel_equilibration_time = 30 * pp.YEAR
    injection_protocol, time_params = _hs1_protocols_optimal_scaling()

    newton_params = NewtonParameters(
        convergence_tol=1e-6,
        max_iterations=300,
    )
    biot_params = BiotParameters(
        # BaseParameters
        length_scale=length_scale,
        scalar_scale=scalar_scale,
        use_temp_path=True,  # Use a temporary directory for storing mesh
        time_step=time_params.initial_time_step,
        end_time=time_params.end_time,
        rock=GrimselGranodiorite(
            PERMEABILITY=5e-21,  # low k: 5e-21 -- high k: 2 * 5e-21
        ),
        gravity=False,
        # GeometryParameters
        shearzone_names=[
            "S1_1",
            "S1_2",
            "S1_3",
            "S3_1",  # TODO: This is the wrong S3 shear zone.
        ],  # "S3_2"],  # ["S1_2", "S3_1"],
        fraczone_bounding_box={
            "xmin": -1,
            "ymin": 80,
            "zmin": -5,  # -5, # Extended frac zone boundary box test
            "xmax": 86,  # +10,
            "ymax": 151,  # +5,
            "zmax": 41,  # +5,
        },
        # MechanicsParameters
        dilation_angle=np.radians(3),
        # FlowParameters
        source_scalar_borehole_shearzone={
            "shearzone": "S1_3",
            "borehole": "INJ2",
        },
        well_cells=shearzone_injection_cell,
        tunnel_equilibrium_time=tunnel_equilibration_time,  # TODO: Think about initial conditions for scaling
        injection_protocol=injection_protocol,
        frac_transmissivity=[
            5e-8,
            1e-9,
            8.3e-11,  # Pre-stimulation T during HS1, observed in S1.3-INJ2.
            3.7e-7,
            # 1e-9,
        ],
        # Initial transmissivites for
        # ["S1_1", "S1_2", "S1_3", "S3_1", "S3_2"]:
        # [5e-8,   1e-9,   5e-10,  3.7e-7, 1e-9]
        # Note background hydraulic conductivity: 1e-14 m/s
        # near_injection_transmissivity=8.3e-11,
        # near_injection_t_radius=3,
    )

    return biot_params, newton_params, time_params


def assemble_matrix(log_ls, log_ss, gb=None):
    """ Prepare a simulation and assemble left-hand side A matrix."""
    ls = np.float_power(10, log_ls)
    ss = np.float_power(10, log_ss)
    biot_params, *_ = prepare_params(ls, ss)
    setup = ISCBoxModel(biot_params, lcin=5 * 2, lcout=50 * 2)
    if gb:
        logger.info(f"Using existing grid")
        setup.gb = gb
    else:
        logger.info(f"Creating new grid...")
        setup.create_grid(use_logger=False)
    setup.prepare_simulation()
    A, b = setup.assemble_matrix_rhs()
    return A, setup.gb


def condition_number_porepy(A):
    row_sum = np.sum(np.abs(A), axis=1)
    return np.max(row_sum) / np.min(row_sum)


def find_best_cond_num_neighbour(
    log_ls0,
    log_ss0,
    has_searched=None,
    ls_gb=None,
):
    """ Find optimal conditioning for the HS1 experiment"""

    ls_gb: List[Tuple[float, pp.GridBucket]] = ls_gb or []

    has_searched = has_searched or []
    # Find list of scaling coefficients that hasn't already been searched
    to_search = []
    for log_ls in [log_ls0 - 1, log_ls0, log_ls0 + 1]:
        for log_ss in [log_ss0 - 1, log_ss0, log_ss0 + 1]:
            pair = (log_ls, log_ss)
            if pair not in has_searched:
                to_search.append(pair)

    # Iterate through the new coefficients
    agg_res = []
    for pair in to_search:
        log_ls, log_ss = pair
        # Find an existing GridBucket
        try:
            gb_lst = [p[1] for p in ls_gb if p[0] == log_ls]
            gb = gb_lst[0]
        except IndexError:
            gb = None

        # Construct matrix
        logger.info(f"Cond num for log_ls={log_ls}, log_ss={log_ss}")
        A, new_gb = assemble_matrix(log_ls, log_ss, gb)
        cond_pp = condition_number_porepy(A)
        res = {
            "log_ls": log_ls,
            "log_ss": log_ss,
            "cond_pp": cond_pp,
        }
        agg_res.append(res)
        has_searched.append(pair)  # Add search to has_searched list.
        print(f"appended to has_searched: {has_searched}")

        logger.info(
            f"Cond results for log_ls={log_ls}, log_ss={log_ss}: cond_pp={cond_pp:.2e}\n"
        )

        # Add GridBucket to list if a new one was constructed in this iteration
        if gb is None:
            ls_gb.append((log_ls, new_gb))

    return agg_res, has_searched, ls_gb


def search_best_cond_num(log_ls0, log_ss0):
    """ Wrapper for find_best_cond_num_neighbour to do optimal search"""
    results = pd.DataFrame(columns=["log_ls", "log_ss", "cond_pp"])

    _has_searched = []
    _ls_gb = []

    def do_iterative_search(data, has_searched, ls_gb):
        print("\nstarting new search. data:")
        print(data)
        old_results_size = data.index.size
        if old_results_size > 0:
            # Find current optimal scaling
            log_ls, log_ss, _ = data.sort_values("cond_pp").iloc[0, :]
        else:
            # Only the initial, empty DataFrame will pick these values
            log_ls, log_ss = log_ls0, log_ss0

        # Find optimal scaling around this neighbour
        print(f"has_searched before: {has_searched}")
        agg_res, has_searched, ls_gb = find_best_cond_num_neighbour(
            log_ls, log_ss, has_searched, ls_gb
        )
        print(f"has_searched after: {has_searched}")

        if len(agg_res) > 0:
            # If any new searches was executed, append these to results
            d = data.append(agg_res, ignore_index=True)

            # Do the iteration
            print("Search completed. Data:")
            print(d)
            return do_iterative_search(d, has_searched, ls_gb)
        else:
            # Otherwise, we assume we have found the optimal coefficients
            return data

    # Do search
    results = do_iterative_search(results, _has_searched, _ls_gb)

    return results
