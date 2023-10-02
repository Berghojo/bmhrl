import os
from pathlib import Path

def get_model_checkpoint_dir(cfg, epoch, newest=None):
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    if newest is None:
        p = os.path.join(cfg.model_checkpoint_path, "checkpoints", f"E_{epoch}")
    else:
        p = os.path.join(cfg.model_checkpoint_path, "checkpoints", f"E_{newest}")
    Path(p).mkdir(parents=True, exist_ok=True)
    return p
