import math
from typing import Literal

ParamScheduleType = Literal["increasing", "decreasing", "no-op"]
ScheduleFnVariant = Literal["linear", "exponential", "logarithmic", "cosine"]


class ScalarAnnealingScheduler:
    """
    Scheduler for annealing a scalar value from an initial value to a final value over time.

    This class updates a scalar value according to a specified schedule function,
    decreasing it from `initial_value` towards `final_value` over time based on the
    provided scheduling function. It allows for adjusting the value dynamically
    during training or other iterative processes.

    Attributes:
        initial_value (float):
            The starting value of the scalar to be annealed at the beginning of each epoch.
            It's recomputed at the end of each epoch.

        final_value (float):
            The minimum value to which the scalar can be annealed.

        delta (float):
            The rate at which `initial_value` decreases at the end of each epoch.

        schedule_fn_variant (str):
            The type of scheduling function used for annealing (e.g., "logarithmic").

        current (float):
            The current value of the scalar after each annealing step.

        step (int):
            The number of update steps taken within the current epoch.
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        delta: float,
        schedule_fn_variant: ScheduleFnVariant = "logarithmic",
    ):
        self.current = initial_value
        self.initial_value = initial_value
        self.final_value = final_value
        self.delta = delta
        self.schedule_fn_variant = schedule_fn_variant

        self.step = 0

    def downscale_on_epoch_end(self):
        """
        Decreases the initial value at the end of an epoch and resets the step counter.
        This method updates `self.initial_value` by reducing it by a factor of `delta * self.initial_value`,
        but ensures that it does not go below `self.final_value`.
        """

        self.initial_value = max(
            [self.initial_value - self.delta * self.initial_value, self.final_value]
        )

        self.step = 0

        return self.initial_value

    def update(self):
        """
        Update the current scheduling value based on the selected annealing function.

        This method updates `self.current` by applying the scheduling function specified in
        `self.schedule_fn_variant`. Currently, only the "logarithmic" variant is implemented.

        Raises:
            NotImplementedError: If `self.schedule_fn_variant` is not a supported variant.

        Returns:
            The updated current scheduling value.
        """
        match self.schedule_fn_variant:
            case "logarithmic":
                self.current = self._log_schedule()

            case _:
                raise NotImplementedError(
                    f"{self.schedule_fn_variant} is not yet supported as an annealing function"
                )

        self.step += 1
        return self.current

    def _log_schedule(self):
        numerator = self.initial_value - self.final_value
        denominator = 1 + math.log(1 + self.step)

        return self.final_value + numerator / denominator
