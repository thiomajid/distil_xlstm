import math


class ScalarScheduler:
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        schedule_type: str = "linear",
    ):
        super().__init__()

        assert schedule_type in [
            "linear",
            "exponential",
            "logarithmic",
            "cosine",
        ], f"{schedule_type} is an invalid scalar schedule type"

        self._value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0

    def step(self):
        pass

    def get_value(self) -> float:
        pass


class ScalarAnnealing(ScalarScheduler):
    def __init__(
        self,
        initial_value,
        final_value,
        total_steps,
        schedule_type="linear",
    ):
        super().__init__(initial_value, final_value, total_steps, schedule_type)

    def step(self):
        """Update the temperature based on the current step and schedule type."""

        self.current_step += 1

        # Ensure progress is capped at 1.0
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            self.current_temperature = self._linear_annealing(progress)
        elif self.schedule_type == "exponential":
            self.current_temperature = self._exponential_annealing(progress)
        elif self.schedule_type == "logarithmic":
            self.current_temperature = self._logarithmic_annealing()
        elif self.schedule_type == "cosine":
            self.current_temperature = self._cosine_annealing(progress)

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

    def get_temperature(self):
        """Returns the current temperature value."""
        return self.current_temperature


class ScalarIncrement(ScalarScheduler):
    pass
