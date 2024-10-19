from transformers import TrainerCallback


class TemperatureCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_step_end(self, args, state, control, **kwargs):
        return super().on_step_end(args, state, control, **kwargs)
