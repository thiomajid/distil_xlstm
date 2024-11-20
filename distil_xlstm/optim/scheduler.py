import math
from typing import Literal

ParamScheduleType = Literal["increasing", "decreasing", "no-op"]
ScheduleFnVariant = Literal["linear", "exponential", "logarithmic", "cosine"]


def create_scalar_scheduler(
    variation: ParamScheduleType,
    *,
    initial_value: float,
    final_value: float,
    total_steps: int,
    variation_fn: ScheduleFnVariant,
):
    """
    Create a scalar scheduler based on the specified variation type.
    The scalar scheduler can be used to anneal or increment a scalar value based
    on the current step and schedule type (linear, exponential, logarithmic, cosine).
    """

    match variation:
        case "increasing":
            return ScalarIncrementScheduler(
                initial_value=initial_value,
                final_value=final_value,
                total_steps=total_steps,
                variation_fn=variation_fn,
            )

        case "decreasing":
            return ScalarAnnealingScheduler(
                initial_value=initial_value,
                final_value=final_value,
                total_steps=total_steps,
                variation_fn=variation_fn,
            )
        case "no-op":
            return NoOpScalarScheduler(
                initial_value=initial_value,
                final_value=final_value,
                total_steps=total_steps,
                variation_fn=variation_fn,
            )

        case _:
            raise ValueError(f"{variation} is not a supported variation for scalar scheduler")


class ScalarScheduler:
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        variation_fn: ScheduleFnVariant = "linear",
    ):
        super().__init__()

        assert variation_fn in [
            "linear",
            "exponential",
            "logarithmic",
            "cosine",
        ], f"{variation_fn} is an invalid scalar schedule type"

        self._value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.variation_fn: ScheduleFnVariant = variation_fn
        self.current_step = 0

    def step(self):
        pass

    def get_value(self) -> float:
        return self._value


class NoOpScalarScheduler(ScalarScheduler):
    def __init__(
        self,
        initial_value,
        final_value,
        total_steps,
        variation_fn: ScheduleFnVariant = "linear",
    ):
        super().__init__(initial_value, final_value, total_steps, variation_fn)

    def step(self):
        self.current_step += 1


class ScalarAnnealingScheduler(ScalarScheduler):
    def __init__(
        self,
        initial_value,
        final_value,
        total_steps,
        variation_fn: ScheduleFnVariant = "linear",
    ):
        super().__init__(initial_value, final_value, total_steps, variation_fn)

    def step(self):
        """Update the temperature based on the current step and schedule type."""

        self.current_step += 1

        # Ensure progress is capped at 1.0
        progress = min(self.current_step / self.total_steps, 1.0)

        match self.variation_fn:
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
