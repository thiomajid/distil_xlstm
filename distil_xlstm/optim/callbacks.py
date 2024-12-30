from transformers import TrainerCallback

from distil_xlstm.optim.scheduler import ScalarAnnealingScheduler


class AnnealingCallback(TrainerCallback):
    def __init__(
        self,
        temperature_scheduler: ScalarAnnealingScheduler,
        alpha_scheduler: ScalarAnnealingScheduler,
    ):
        super().__init__()

        self.temperature_scheduler = temperature_scheduler
        self.alpha_scheduler = alpha_scheduler

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            args.temperature = self.temperature_scheduler.update()
            args.alpha = self.alpha_scheduler.update()
