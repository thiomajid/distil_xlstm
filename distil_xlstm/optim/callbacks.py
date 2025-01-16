from transformers import TrainerCallback

from distil_xlstm.optim.scheduler import ScalarAnnealingScheduler


class AnnealingCallback(TrainerCallback):
    def __init__(self, schedulers: dict[str, ScalarAnnealingScheduler]):
        super().__init__()

        self.schedulers = schedulers

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            for weight, scheduler in self.schedulers.items():
                setattr(args, weight, scheduler.update())

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            for weight, scheduler in self.schedulers.items():
                setattr(args, weight, scheduler.downscale_on_epoch_end())
