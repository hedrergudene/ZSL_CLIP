#
# Callbacks
#

# Requirements
from typing import Dict
import wandb

# Weights and Biases callback
def wandb_update(history:Dict):
    """This method logs the gathered loss and metrics in training epoch into Weights and Biases.

    Args:
        history (Dict): Dictionary with the following keys:
            * 'epoch': The current epoch.
            * 'train': Average train loss through batches in that epoch.
            * 'val': Average validation loss through batches in that epoch.
            * 'lr': Learning rate used for the optimiser in that epoch.
            * Optional metrics.
    """
  
    # Copy input to make changes
    data_log = history.copy()
    # Rename certain keys
    step = data_log['epoch']
    del data_log['epoch']
    wandb.log(data_log, step=step)