import abc
import logging
import time
from typing import Dict, List, Optional

import numpy as np

import porepy as pp
from GTS.isc_modelling.parameter import BaseParameters
from mastersproject.util.logging_util import timer
from porepy.models.abstract_model import AbstractModel
from pypardiso import spsolve

logger = logging.getLogger(__name__)


class CommonAbstractModel(AbstractModel):
    def __init__(self, params: BaseParameters):
        self.params = params

        # Grid
        self.gb: Optional[pp.GridBucket] = None
        self.bounding_box: Optional[Dict[str, int]] = None
        self.assembler: Optional[pp.Assembler] = None

        # Viz
        self.viz: Optional[pp.Exporter] = None
        self.export_fields: List = []

    def get_state_vector(self):
        """Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state, as stored in the GridBucket.

        """
        size = self.assembler.num_dof()
        state = np.zeros(size)
        for g, var in self.assembler.block_dof.keys():
            # Index of
            ind = self.assembler.dof_ind(g, var)

            if isinstance(g, tuple):
                values = self.gb.edge_props(g)[pp.STATE][var]
            else:
                values = self.gb.node_props(g)[pp.STATE][var]
            state[ind] = values

        return state

    @abc.abstractmethod
    def prepare_simulation(self):
        """Method called prior to the start of time stepping, or prior to entering the
        non-linear solver for stationary problems.

        The intended use is to define parameters, geometry and grid, discretize linear
        and time-independent terms, and generally prepare for the simulation.

        """
        pass

    # --- Newton iterations ---

    @abc.abstractmethod
    def before_newton_loop(self):
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discertizations etc.

        """
        pass

    def before_newton_iteration(self):
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discertizations etc.

        """
        pass

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """Actions after each Newton iteration

        For instance; update_state updates the non-linear terms.

        Parameters:
            solution_vector : np.ndarray
                solution vector for the current iterate.

        """
        self.update_state(solution_vector)

    def update_state(self, solution_vector: np.ndarray) -> None:
        """ Update variables for the current Newton iteration"""
        pass

    def after_newton_convergence(self, solution, errors, iteration_counter) -> None:
        """ On Newton convergence, update STATE for all variables."""
        self.assembler.distribute_variable(solution)
        self.export_step()

    def after_newton_failure(self, solution, errors, iteration_counter) -> None:
        """ Raise ValueError for failed Newton iteration"""
        non_linear_error = "Newton iterations did not converge"
        linear_error = "Tried solving singular matrix for the linear problem."
        error_type = non_linear_error if self._is_nonlinear_problem() else linear_error
        raise NewtonFailure(error_type, solution)

    @abc.abstractmethod
    def after_simulation(self):
        """ Called after a completed simulation """
        pass

    # --- Simulation and solvers ---

    @abc.abstractmethod
    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict,
    ):

        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = np.any(np.isnan(solution))
            converged = not diverged
            error = np.nan if diverged else 0
            return error, converged, diverged

        else:
            raise NotImplementedError(
                "Convergence check for non-linear problems is not yet implemented"
            )

    def assemble_matrix_rhs(self):
        """Wrapper for assembler.assemble_matrix_rhs

        Wrap the assembler method so it can be overwritten elsewhere.
        """
        A, b = self.assembler.assemble_matrix_rhs()
        return A, b

    @timer(logger, level="INFO")
    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """ Assemble a solve the linear system"""

        A, b = self.assemble_matrix_rhs()

        # Estimate condition number
        logger.info(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.info(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and "
            f"min {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )

        # UMFPACK Estimate of condition number
        sum_diag_abs_A = np.abs(A.diagonal())
        logger.info(
            f"UMFPACK Condition number estimate: "
            f"{np.min(sum_diag_abs_A) / np.max(sum_diag_abs_A) :.2e}"
        )

        if self.params.linear_solver == "direct":
            tic = time.time()
            logger.info("Solve Ax=b using scipy")
            # sol = spla.spsolve(A, b)
            sol = spsolve(A, b)  # pypardiso
            logger.info(f"Done. Elapsed time {time.time() - tic}")
            norm = np.linalg.norm(b - A * sol)
            logger.info(f"||b-Ax|| = {norm}")

            rhs_norm = np.linalg.norm(b)
            identical_zero = np.isclose(rhs_norm, 0) and np.isclose(norm, 0)
            rel_norm = norm / rhs_norm if not identical_zero else norm
            logger.info(f"||b-Ax|| / ||b|| = {rel_norm}")
            return sol

        else:
            raise ValueError(f"Unknown linear solver {self.params.linear_solver}")

    # --- Exporting and visualization ---

    @abc.abstractmethod
    def set_viz(self):
        self.viz = pp.Exporter(
            self.gb,
            file_name=self.params.viz_file_name,
            folder_name=self.params.folder_name,
        )

    @abc.abstractmethod
    def export_step(self, write_vtk: bool = True):
        """ Export a step to visualization"""
        pass

    # --- Helper methods ---

    @property
    def Nd(self) -> int:  # noqa
        if self.gb:
            return self.gb.dim_max()

    # @Nd.setter
    # def Nd(self):  # noqa
    #     pass

    def _is_nonlinear_problem(self) -> bool:
        """ Whether the problem is non-linear"""
        return False

    def _nd_grid(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.gb is set."""
        return self.gb.grids_of_dimension(self.Nd)[0]

    def domain_boundary_sides(self, g):
        """
        Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries.
        """
        tol = 1e-10
        box = self.bounding_box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        north = g.face_centers[1] > box["ymax"] - tol
        south = g.face_centers[1] < box["ymin"] + tol
        if self.Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom


class NewtonFailure(Exception):
    def __init__(self, message, solution=None):
        self.message = message
        self.solution = solution
