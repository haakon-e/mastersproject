import pytest
import numpy as np

from GTS.time_protocols import TimeStepPhase, TimeStepProtocol, InjectionRatePhase, InjectionRateProtocol


class TestTimeStepProtocol:

    def test_monotone_phase_check(self):
        phase1 = TimeStepPhase(start_time=0, end_time=1.1, data=0.1)
        phase2 = TimeStepPhase(start_time=1, end_time=2, data=0.2)

        with pytest.raises(AssertionError):
            tsp = TimeStepProtocol([phase1, phase2])

    def test_active_time_step(self):
        phase1 = TimeStepPhase(start_time=0, end_time=1, data=0.1)
        phase2 = TimeStepPhase(start_time=1, end_time=2, data=0.2)
        tsp = TimeStepProtocol([phase1, phase2])

        # Check middle-of-phase
        assert np.isclose(tsp.active_time_step(0.5), 0.1)

        # Check boundary
        assert np.isclose(tsp.active_time_step(1.), 0.2)

        # Check start time and end time
        assert np.isclose(tsp.active_time_step(0), 0.1)
        assert np.isclose(tsp.active_time_step(2), 0.2)


class TestInjectionProtocol:

    def test_active_injection_rate(self):
        phase1 = InjectionRatePhase(start_time=0, end_time=1, data=0.1)
        phase2 = InjectionRatePhase(start_time=1, end_time=2, data=0.2)
        irp = InjectionRateProtocol([phase1, phase2])

        # Check middle-of-phase
        assert np.isclose(irp.active_rate(0.5), 0.1)

        # Check boundary
        assert np.isclose(irp.active_rate(1.), 0.1)

        # Check start time and end time
        assert np.isclose(irp.active_rate(0), 0.1)
        assert np.isclose(irp.active_rate(2), 0.2)

    def test_create_protocol(self):
        phase_limits = [0, 1, 2]
        rates = [0.5, 0.8]
        irp = InjectionRateProtocol.create_protocol(phase_limits, rates)

        # Check basic construction properties
        assert irp.start_time == 0
        assert irp.end_time == 2
        assert irp.duration == 2
        assert np.allclose(np.array(irp.phase_end_times), np.array(phase_limits[1:]))
        assert np.allclose(np.array(irp.phase_limits), np.array(phase_limits))

        # Test rates
        assert np.isclose(irp.active_rate(0), 0.5)
        assert np.isclose(irp.active_rate(0.5), 0.5)
        assert np.isclose(irp.active_rate(1), 0.5)
        assert np.isclose(irp.active_rate(1.5), 0.8)
        assert np.isclose(irp.active_rate(2), 0.8)

