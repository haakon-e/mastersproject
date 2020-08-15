from typing import List
import logging

import numpy as np

from GTS.isc_modelling.general_model import CommonAbstractModel, NewtonFailure
from pydantic import BaseModel
from util import timer

logger = logging.getLogger(__name__)


class NewtonParameters(BaseModel):
    max_iterations: int = 10
    convergence_tol: float = 1e-10
    divergence_tol: float = 1e5


class TimeParameters(BaseModel):
    start_time: float = 0
    end_time: float = 1

    time_step: float = 1                # initial time step
    max_time_step: float = 1            # max step size
    must_hit_times: List[float] = [1]   # list of times we must hit exactly


class TimeMachine:
    def __init__(
        self,
        setup: CommonAbstractModel,
        newton_params: NewtonParameters,
        time_params: TimeParameters,
    ) -> None:
        self.setup = setup
        self.newton_params = newton_params
        self.time_params = time_params

        # Fix the current size of the time step
        self.current_time_step = self.time_params.time_step

    @timer(logger)
    def iteration(self, tol):
        sol = self.setup.assemble_and_solve_linear_system(tol)
        return sol

    @timer(logger)
    def time_iteration(self):
        """ Solve a non-linear system using a Newton Method

        Equivalent to NewtonSolver.solve(setup)"""
        setup = self.setup
        # Re-discretize time-dependent terms
        setup.before_newton_loop()

        iteration_counter = 0

        init_sol: np.ndarray = setup.get_state_vector()
        prev_sol = init_sol
        sol = init_sol
        errors = []
        error_norm = 1

        for it in range(self.newton_params.max_iterations):
            logger.info(
                f"Newton iteration number {it} of {self.newton_params.max_iterations}"
            )
            # Re-discretize non-linear terms
            setup.before_newton_iteration()

            # Solve
            lin_tol = np.minimum(1e-4, error_norm)
            sol = self.iteration(lin_tol)

            # After iteration
            setup.after_newton_iteration(sol)

            # Check convergence
            error_norm, is_converged, is_diverged = setup.check_convergence(
                sol, prev_sol, init_sol, self.newton_params.dict()
            )
            prev_sol = sol
            errors.append(error_norm)

            if is_diverged:
                setup.after_newton_failure(sol, errors, iteration_counter)
            elif is_converged:
                setup.after_newton_convergence(sol, errors, iteration_counter)
                return sol

            iteration_counter += 1

        # If max newton iterations reached without convergence, then:
        setup.after_newton_failure(sol, errors, iteration_counter)

    @timer(logger)
    def run_simulation(self, prepare_simulation=True):
        """ Run time-dependent non-linear simulation"""
        setup = self.setup
        if prepare_simulation:
            setup.prepare_simulation()

        k_time = 0
        t_end = self.time_params.end_time
        current_time = self.time_params.start_time
        sol = None

        newton_failure = False
        k_nwtn, k_nwtn_max = 0, 5

        while current_time < t_end:
            k_time += 1
            while True:
                k_nwtn += 1
                time_step = self.determine_time_step(current_time, newton_failure, sol)
                new_time = current_time + time_step
                setup.time, setup.time_step = new_time, time_step

                logger.info(
                    f"Time step {k_time} for time {new_time:.1e}, with "
                    f"time step {time_step:.1e}. Tries ({k_nwtn}/{k_nwtn_max})."
                )
                try:
                    sol = self.time_iteration()
                except NewtonFailure:
                    # If Newton method failed, reset the iterate to STATE variables.
                    init_sol = setup.get_state_vector()
                    setup.update_state(init_sol)
                    newton_failure = True

                    # If we have tried too many times. Raise.
                    if k_nwtn > k_nwtn_max:
                        raise ValueError(f"Time step failed to converge after {k_nwtn} tries.")
                else:
                    # If Newton Failure did not occur, we succeeded.
                    break

            # Before the next time step, update the current time step size.
            self.current_time_step = time_step

        setup.after_simulation()

    def determine_time_step(self, current_time, newton_failure, sol) -> float:
        """ Constant step size with Newton failure adjustment

        Parameters
        ----------
        current_time : float
            the time *before* a new time step is completed
        newton_failure : bool
            indicates whether the previous step failed
        sol : np.ndarray, Optional

        - The standard is that the new time step equals the old time step.
        - If the Newton method failed in the previous step attempt, reduce
            the new step size by 80%.
        - If the new step exceeds some time value we want to hit, set the
            new step to exactly hit this value.
        """
        # Fetch previous step size
        current_step_size = self.current_time_step

        # Reduce step size by 80% if Newton loop failed
        if newton_failure:
            logger.info(f"Newton failure. Reduce time step by 80% and retry time step.")
            current_step_size *= 0.2

        new_time = current_time + current_step_size

        # Check if we skipped a must hit time.
        must_hit_times = np.array(self.time_params.must_hit_times)
        current_bin, new_bin = np.searchsorted(
            must_hit_times,
            [current_time, new_time],
            side="left",
        )
        if current_bin < new_bin:
            new_time = must_hit_times[current_bin]
            current_step_size = new_time - current_time

        return current_step_size


class GrabowskiTimeMachine(TimeMachine):
    def determine_time_step(self) -> float:
        """ Grabowski step

        see Grabowski et al. (1979): A fully implicit general purpose
            finite-difference thermal model for in situ combustion and steam
        and;
        McClure (2012), PhD-thesis: Modeling and characterization of hydraulic
            stimulation and induced seismicity in geothermal and shale gas
            reservoirs
        """
        setup = self.setup
        dt0 = self.time_params.time_step
        # eta
        return 0.0
