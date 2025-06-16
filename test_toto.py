from training.perceiver import *
from training.utils import *
from training.losses import *

from training.VIT import *
from training.ResNet import *
from collections import defaultdict
from training import *
import os
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,GradientAccumulationScheduler
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from pytorch_lightning.profilers import AdvancedProfiler
profiler = AdvancedProfiler(dirpath="./profiling", filename="advanced_profile.txt")
import matplotlib.pyplot as plt

seed_everything(42, workers=True)



seed_everything(42, workers=True)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

config_model = read_yaml("./training/configs/config_test-Atomiser_Atos.yaml")
xp_name="testset"


wand = True
wandb_logger = None
if wand:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import wandb
        wandb.init(
            name=config_model['encoder'],
            project="Atomizer_Cloud",
            config=config_model
        )
        wandb_logger = WandbLogger(project="Atomizer_Cloud")

#def __init__(self, config, wand, name)



transform=transformations_config("./data/bands.yaml",config_model)

model = Model(config_model,wand=wand,transform=transform)

data_module=CloudSEN12DataModule(batch_size=config_model["dataset"]["batchsize"], num_workers=8)
# Callbacks
checkpoint_callback_IoU = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename=config_model["encoder"]+str(xp_name)+"-alone_best_model_val_mod_val-{epoch:02d}-{val_mod_val_ap:.4f}",
    monitor="val_IoU",
    mode="max",
    save_top_k=1,
    verbose=True,
)






accumulator = GradientAccumulationScheduler(scheduling={0: 64})

# Trainer
trainer = Trainer(
    #strategy="ddp",
    #strategy='ddp_find_unused_parameters_true',
    devices=1, 
    max_epochs=config_model["trainer"]["epochs"],
    logger=wandb_logger,
    log_every_n_steps=16,
    accelerator="gpu",
    callbacks=[checkpoint_callback_IoU],#,accumulator],
    default_root_dir="./checkpoints/",
    #val_check_interval=0.3,
    precision="bf16-mixed",
    profiler=profiler
)


# Fit the model
trainer.fit(model, datamodule=data_module)






# ... after training completes, within your "if wand" block:
if wand and os.environ.get("LOCAL_RANK", "0") == "0":
    run_id = wandb.run.id
    print("WANDB_RUN_ID:", run_id)
    
    # Create the directory for storing wandb run IDs if it doesn't exist
    runs_dir = "training/wandb_runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    
    # Save the run ID to a file inside wandb_runs
    run_file = os.path.join(runs_dir, xp_name+".txt")
    with open(run_file, "w") as f:
        f.write(run_id)






