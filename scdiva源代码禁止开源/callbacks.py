# scdiva/callbacks.py
import psutil
from transformers import TrainerCallback
import swanlab

class MemorySafetyCallback(TrainerCallback):
    def __init__(self, limit_gb=900):
        self.limit_gb = limit_gb
        self.warned = False

    def on_step_begin(self, args, state, control, **kwargs):
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        if used_gb >= self.limit_gb:
            if not self.warned:
                print(f"\n[CRITICAL WARNING] Memory usage ({used_gb:.2f} GB) exceeded limit ({self.limit_gb} GB).")
                print("Stopping training to prevent OOM crash...")
                self.warned = True
            control.should_training_stop = True
            control.should_save = True

class SwanLabCallback(TrainerCallback):
    def __init__(self, project, config):
        self.project = project
        self.config = config
        self._init = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._init and state.is_world_process_zero:
            swanlab.init(project=self.project, config=self.config)
            self._init = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._init and state.is_world_process_zero and logs:
            metrics = {k: v for k, v in logs.items() if k != "epoch"}
            if metrics:
                swanlab.log(metrics, step=state.global_step)
                if any(k.startswith("eval_") for k in metrics):
                    print(f"✅ [SwanLab] Uploaded Eval Metrics: {list(metrics.keys())}")
