import torch
from transformers import TrainerCallback

from distil_xlstm.optim.scheduler import ScalarAnnealingScheduler


class AnnealingCallback(TrainerCallback):
    def __init__(self, schedulers: dict[str, ScalarAnnealingScheduler]):
        super().__init__()

        self._schedulers = schedulers

    def register_scheduler(self, weight: str, scheduler: ScalarAnnealingScheduler):
        """
        Register a new scheduler to the callback.

        Args:
            weight (str): The name of the weight to be scheduled.
            scheduler (ScalarAnnealingScheduler): The scheduler instance.
        """
        self._schedulers[weight] = scheduler

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            for weight, scheduler in self._schedulers.items():
                setattr(args, weight, scheduler.update())

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            for weight, scheduler in self._schedulers.items():
                setattr(args, weight, scheduler.downscale_on_epoch_end())


# Add a custom callback to log perplexity during training and evaluation
class PerplexityLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Calculate and log perplexity for training
            if "loss" in logs:
                try:
                    perplexity = torch.exp(logs["loss"]).item()
                    logs["perplexity"] = round(perplexity, 4)
                except OverflowError:
                    logs["perplexity"] = float("inf")

            # Calculate and log perplexity for evaluation
            if "eval_loss" in logs:
                try:
                    eval_perplexity = torch.exp(logs["eval_loss"]).item()
                    logs["eval_perplexity"] = round(eval_perplexity, 4)
                except OverflowError:
                    logs["eval_perplexity"] = float("inf")
