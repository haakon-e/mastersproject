from typing import List
import logging

import numpy as np

from GTS.isc_modelling.general_model import CommonAbstractModel, NewtonFailure
from GTS.time_protocols import TimeStepProtocol
from pydantic import BaseModel
from util import timer

logger = logging.getLogger(__name__)


class NewtonParameters(BaseModel):
    max_iterations: int = 10
    convergence_tol: float = 1e-10
    divergence_tol: float = 1e5


class TimeMachine:
    def __init__(
        self,
        setup: CommonAbstractModel,
        newton_params: NewtonParameters,
        time_params: TimeStepProtocol,
        max_newton_failure_retries: int = 0,
    ) -> None:
        self.setup = setup
        self.newton_params = newton_params
        self.time_params = time_params

        # Initialize current time and time step
        self.current_time: float = self.time_params.start_time
        self.current_time_step = self.time_params.active_time_step(
            self.current_time
        )
        self.k_time = 0

        # Max time iteration attempts
        self.k_newton_max = max_newton_failure_retries + 1

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

        sol = None
        while self.current_time < self.time_params.end_time:
            newton_failure = False
            k_nwtn = 0
            self.k_time += 1
            while True:
                k_nwtn += 1
                time_step = self.determine_time_step(newton_failure, sol)
                new_time = self.current_time + time_step
                setup.time, setup.time_step = new_time, time_step

                logger.info(
                    f"Time step no. {self.k_time}. "
                    f"From t={self.current_time:.1e} to t={new_time:.1e}. "
                    f"dt={time_step:.2e}. End time {self.time_params.end_time:.1e}. "
                    f"Tries ({k_nwtn}/{self.k_newton_max})."
                )
                try:
                    sol = self.time_iteration()
                except NewtonFailure:
                    # If Newton method failed, reset the iterate to STATE variables.
                    init_sol = setup.get_state_vector()
                    setup.update_state(init_sol)
                    newton_failure = True

                    # If we have tried too many times. Raise.
                    if k_nwtn >= self.k_newton_max:
                        msg = f"Time step failed to converge after {k_nwtn} tries."

                        # For testing
                        # raise ValueError(msg)
                        logger.critical(msg)
                        break
                else:
                    # If Newton Failure did not occur, we succeeded.
                    break

            if newton_failure:
                # For testing
                break

            # Before the next time step, update the current time step size.
            self.current_time_step: float = time_step
            self.current_time: float = new_time

        setup.after_simulation()

    # Determine the next time step

    def determine_time_step(self, newton_failure, sol) -> float:
        """ Constant step size with Newton failure adjustment

        Parameters
        ----------
        newton_failure : bool
            indicates whether the previous step failed
        sol : np.ndarray, Optional

        - The standard is that the new time step equals the old time step.
        - If the Newton method failed in the previous step attempt, reduce
            the new step size by 80%.
        - If the new step exceeds some time value we want to hit, set the
            new step to exactly hit this value.
        """

        # Reduce step size by 80% if Newton loop failed
        self.reduce_time_step_on_newton_failure(newton_failure)

        # Adjust step size for must-hit times
        current_time_step = self.adjust_time_step_to_must_hit_times()

        return current_time_step

    # --- Common adjustment options ---

    def reduce_time_step_on_newton_failure(self, newton_failure) -> None:
        """ Reduce time step by 80% if previous attempt resulted in Newton failure"""
        # Reduce step size by 80% if Newton loop failed
        if newton_failure:
            logger.info(f"Newton failure. Reduce time step by 80% and retry time step. "
                        f"Old time step: {self.current_time_step:.2e}, "
                        f"new time step: {self.current_time_step * 0.2:.2e}")
            self.current_time_step *= 0.2

    def adjust_time_step_to_must_hit_times(self) -> float:
        """ Make sure the next time step doesn't skip a must-hit time.

        This method only *temporarily* changes the step size.
        """
        # Fetch previous time and step size
        current_time = self.current_time
        current_time_step = self.current_time_step
        new_time = current_time + current_time_step

        # Adjust step size if we would be skipping a must-hit time.
        must_hit_times = self.time_params.phase_end_times
        current_bin, new_bin = np.searchsorted(
            must_hit_times, [current_time, new_time], side="right",
        )
        if current_bin < new_bin:
            new_time = must_hit_times[current_bin]
            current_time_step = new_time - current_time
        return current_time_step


class TimeMachinePhasesConstantDt(TimeMachine):
    """ Time machine with constant time step per phase"""
    def __init__(
        self,
        setup: CommonAbstractModel,
        newton_params: NewtonParameters,
        time_params: TimeStepProtocol,
    ):
        super().__init__(
            setup, newton_params, time_params, max_newton_failure_retries=0
        )

    def determine_time_step(self, *_) -> float:
        """ Constant step size per phase"""
        current_time = self.current_time
        self.current_time_step = self.time_params.active_time_step(current_time)

        # Adjust step size for must-hit times
        current_time_step = self.adjust_time_step_to_must_hit_times()

        return current_time_step


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
