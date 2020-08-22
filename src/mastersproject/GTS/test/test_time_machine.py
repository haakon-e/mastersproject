from pathlib import Path

import numpy as np
import pytest

from GTS import BaseParameters, Flow

from GTS.time_machine import TimeMachinePhasesConstantDt, NewtonParameters, TimeMachine
from GTS.time_protocols import TimeStepPhase, TimeStepProtocol


class TestTimeMachinePhasesConstantDt:

    def test_determine_time_step_from_phase(self):
        # Create protocol with two phases.
        phase1 = TimeStepPhase(start_time=0, end_time=1, data=0.5)
        phase2 = TimeStepPhase(start_time=1, end_time=2, data=0.2)
        time_params = TimeStepProtocol([phase1, phase2])
        tm = TimeMachinePhasesConstantDt(None, None, time_params)  # noqa

        # Check initial time
        assert tm.current_time == phase1.start_time

        # Check that new (right) phase is chosen at boundary
        tm.current_time = 1.
        current_dt = tm.determine_time_step()
        assert current_dt == 0.2

        # Check that time step is adjusted if boundary is hit
        tm.current_time = 0.8
        current_dt = tm.determine_time_step()
        assert np.isclose(current_dt, 0.2)

        # Check that final time step is hit
        tm.current_time = 1.9
        current_dt = tm.determine_time_step()
        assert np.isclose(current_dt, 0.1)

    def test_run_simulation_one_time_step(self, mocker):
        """ Test procedure with only one time step."""
        # Faux setup of Time Machine
        phase_limits = [0, 1]
        steps = [1.2]
        time_params = TimeStepProtocol.create_protocol(phase_limits, steps)
        setup = Flow(BaseParameters(
            folder_name=Path(__file__).parent / "results/test_time_machine")
        )
        newton = NewtonParameters()
        time_machine = TimeMachinePhasesConstantDt(setup, newton, time_params)

        # Patch time_iteration(), which is called in run_simulation()
        mocker.patch("GTS.time_machine.TimeMachine.time_iteration")
        # Patch setup.after_simulation(), which is called in run_simulation()
        mocker.patch("GTS.Flow.after_simulation")

        # Run the simulation
        time_machine.run_simulation(prepare_simulation=False)

        # Check that time_iteration was called exactly once
        TimeMachine.time_iteration.assert_called_once()
        # Check that after_simulation was called exactly once
        Flow.after_simulation.assert_called_once()  # noqa

        # Check parameters of the Time Machine
        assert time_machine.k_time == 1
        # Time step should be adjusted to dt=1 since the suggested dt=1.2 > 1 (= 1 - 0)
        assert np.isclose(time_machine.current_time_step, 1)
        assert np.isclose(time_machine.current_time, 1)

    def test_run_simulation_adjust_dt_2_steps(self, mocker):
        """ We should have one step in first phase and two steps in second phase"""
        # Faux setup of Time Machine
        phase_limits = [0, 1, 2]
        steps = [1.2, 0.5]
        time_params = TimeStepProtocol.create_protocol(phase_limits, steps)
        setup = Flow(BaseParameters(
            folder_name=Path(__file__).parent / "results/test_time_machine")
        )
        newton = NewtonParameters()
        time_machine = TimeMachinePhasesConstantDt(setup, newton, time_params)

        # Patch time_iteration(), which is called in run_simulation()
        mocker.patch("GTS.time_machine.TimeMachine.time_iteration")
        # Patch setup.after_simulation(), which is called in run_simulation()
        mocker.patch("GTS.Flow.after_simulation")

        # Run the simulation
        time_machine.run_simulation(prepare_simulation=False)

        # Check that time_iteration was called 3 times
        assert TimeMachine.time_iteration.call_count == 3
        # Check that after_simulation was called exactly once
        Flow.after_simulation.assert_called_once()  # noqa

        # Check parameters of the Time Machine
        assert time_machine.k_time == 3
        # Time step should be adjusted to dt=1 since the suggested dt=1.2 > 1 (= 1 - 0)
        assert np.isclose(time_machine.current_time_step, 0.5)
        assert np.isclose(time_machine.current_time, 2)
