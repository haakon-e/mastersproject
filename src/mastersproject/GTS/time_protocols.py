from typing import List, Type, Any

import numpy as np

from pydantic import validator
from pydantic.main import BaseModel


# --- Phases ---


class AbstractPhase(BaseModel):
    start_time: float
    end_time: float
    data: Any = None

    metadata: Any = None
    tol: float

    def time_in_phase(self, t: float) -> bool:
        return self.start_time + self.tol < t < self.end_time + self.tol

    # --- Validators ---

    @validator("end_time")
    def validate_scaling(cls, v, values):  # noqa
        start_time = values.get("start_time")
        end_time = v
        assert end_time > start_time
        return v


class InjectionRatePhase(AbstractPhase):
    """ Phase to store an injection rate"""

    data: float = 0

    # For injection phases, we consider the data between the
    # current and the next step. So if we hit a boundary value,
    # we should pick the *LEFT* value.
    #   I.e.: tL <= T < tR
    tol: float = 1.0

    # ^^ time steps are on the order of minutes, so tol can be 1.0


class TimeStepPhase(AbstractPhase):
    """ Phase to store a time step"""

    data: float = 0

    # For time step phases, we consider the data at the current
    # time step. So if we hit a boundary value, we should pick the
    # *RIGHT* value.
    #   I.e.: tL < T <= tR
    # Thus we set the tolerance < 0.
    tol: float = -1.0

    # ^^ time steps are on the order of minutes, so tol can be -1.0


# --- Protocols ---


class AbstractProtocol:
    """ Abstract protocol getting data in discrete time intervals

    Partition of a time interval to phases,
    with arbitrary data within each interval.
    """

    def __init__(self, phases: List[AbstractPhase]):
        self.phases = phases
        self._check_monotone_phases()

    def get_active_phase_data(self, t):
        return self.get_active_phase(t).data

    def get_active_metadata(self, t):
        return self.get_active_phase(t).metadata

    def get_active_phase(self, t) -> AbstractPhase:
        if np.isclose(self.start_time, t):
            return self.phases[0]
        elif np.isclose(self.end_time, t):
            return self.phases[-1]
        active_phases = [p for p in self.phases if p.time_in_phase(t)]
        assert len(active_phases) == 1
        return active_phases[0]

    @property
    def start_time(self):
        return self.phases[0].start_time

    @property
    def end_time(self):
        return self.phases[-1].end_time

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def phase_end_times(self):
        """ Return all phase end times."""
        phase_ends = [p.end_time for p in self.phases]
        return phase_ends

    @property
    def phase_limits(self):
        """ Return union of all phase start and end times."""
        start = [self.start_time]
        start.extend(self.phase_end_times)
        return start

    # Alternative constructor and checks

    @classmethod
    def _create_protocol(
        cls,
        phase_type: Type[AbstractPhase],
        phase_limits: List[float],
        values: List[float],
    ) -> "AbstractProtocol":
        """ Create a (subclass)Protocol from list of phase limits and values."""
        assert (
            len(values) == len(phase_limits) - 1
        ), "All rates must be encapsulated by phase limits"
        phases = [
            phase_type(
                start_time=phase_limits[i], end_time=phase_limits[i + 1], data=value,
            )
            for i, value in enumerate(values)
        ]
        return cls(phases)

    def _check_monotone_phases(self):
        """ Check that the list of phases form a partition of a time interval"""
        assert len(self.phases) >= 1
        p0 = None
        for phase in self.phases:
            p1 = phase
            if p0:
                assert np.isclose(p1.start_time, p0.end_time)
            p0 = p1


class InjectionRateProtocol(AbstractProtocol):
    """ Protocol of injection rates. Units: [litres / second]"""

    def __init__(self, phases: List[InjectionRatePhase]):
        super().__init__(phases)

    def active_rate(self, t):
        return self.get_active_phase_data(t)

    @classmethod
    def create_protocol(
        cls, phase_limits: List[float], rates: List[float]
    ) -> "InjectionRateProtocol":
        """ Create a StimulationProtocol from list of phase limits and rates."""
        return cls._create_protocol(InjectionRatePhase, phase_limits, rates)


class TimeStepProtocol(AbstractProtocol):
    """ Protocol of time steps for each phase"""

    def __init__(self, phases: List[TimeStepPhase]):
        super().__init__(phases)

    def active_time_step(self, t):
        return self.get_active_phase_data(t)

    @property
    def initial_time_step(self):
        return self.phases[0].data

    @classmethod
    def create_protocol(cls, phase_limits, time_steps):
        """ Create a TimeStepProtocol from lists of phase limits and time"""
        return cls._create_protocol(TimeStepPhase, phase_limits, time_steps)
