import math
from typing import Literal

ParamScheduleType = Literal["increasing", "decreasing", "no-op"]
ScheduleFnVariant = Literal["linear", "exponential", "logarithmic", "cosine"]


def create_scalar_scheduler(
    schedule_type: ParamScheduleType,
    *,
    initial_value: float,
    final_value: float,
    total_steps: int,
    variant: ScheduleFnVariant,
):
    match schedule_type:
        case "increasing":
            return ScalarIncrementScheduler(
                initial_value=initial_value,
                final_value=final_value,
                variant=variant,
                total_steps=total_steps,
            )

        case "decreasing":
            return ScalarAnnealingScheduler(
                initial_value=initial_value,
                final_value=final_value,
                variant=variant,
                total_steps=total_steps,
            )
        case "no-op":
            return NoOpScalarScheduler(
                initial_value=initial_value,
                final_value=final_value,
                variant=variant,
                total_steps=total_steps,
            )

        case _:
            raise ValueError(f"{schedule_type} is not a supported scalar scheduler")


class ScalarScheduler:
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        variant: ScheduleFnVariant = "linear",
    ):
        super().__init__()

        assert variant in [
            "linear",
            "exponential",
            "logarithmic",
            "cosine",
        ], f"{variant} is an invalid scalar schedule type"

        self._value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.variant: ScheduleFnVariant = variant
        self.current_step = 0

    def step(self):
        pass

    def get_value(self) -> float:
        self._value


class NoOpScalarScheduler(ScalarScheduler):
    def __init__(
        self,
        initial_value,
        final_value,
        total_steps,
        variant: ScheduleFnVariant = "linear",
    ):
        super().__init__(initial_value, final_value, total_steps, variant)

    def step(self):
        self.current_step += 1


class ScalarAnnealingScheduler(ScalarScheduler):
    def __init__(
        self,
        initial_value,
        final_value,
        total_steps,
        schedule_type: ScheduleFnVariant = "linear",
    ):
        super().__init__(initial_value, final_value, total_steps, schedule_type)

    def step(self):
        """Update the temperature based on the current step and schedule type."""

        self.current_step += 1

        # Ensure progress is capped at 1.0
        progress = min(self.current_step / self.total_steps, 1.0)

        match self.variant:
            case "linear":
                self.current_temperature = self._linear_annealing(progress)
            case "exponential":
                self.current_temperature = self._exponential_annealing(progress)
            case "logarithmic":
                self.current_temperature = self._logarithmic_annealing()
            case "cosine":
                self.current_temperature = self._cosine_annealing(progress)
            case _:
                raise ValueError(
                    f"{self.variant} is an invalid schedule function variant"
                )

    def _linear_annealing(self, progress):
        """Linear annealing formula (works for both increase and decrease)."""
        return self.final_value + (self._value - self.final_value) * (1 - progress)

    def _exponential_annealing(self, progress):
        """Exponential annealing formula (works for both increase and decrease)."""
        return self.final_value + (self._value - self.final_value) * math.exp(-progress)

    def _logarithmic_annealing(self):
        """Logarithmic annealing formula (works for both increase and decrease)."""
        if self.current_step > 0:
            return self.final_value + (self._value - self.final_value) * (
                1 - math.log1p(self.current_step) / math.log(self.total_steps + 1)
            )
        else:
            return self._value

    def _cosine_annealing(self, progress):
        """Cosine annealing formula (works for both increase and decrease)."""
        return self.final_value + 0.5 * (self._value - self.final_value) * (
            1 + math.cos(math.pi * progress)
        )


class ScalarIncrementScheduler(ScalarScheduler):
    pass
