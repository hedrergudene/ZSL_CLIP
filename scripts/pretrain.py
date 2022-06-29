# Requierments
import json
import os
import logging as log
import sys
import fire
import torch
from transformers import TrainingArguments
import wandb

# Dependencies
from src.setup import setup_data
from src.dataset import CLIPDataset
from src.model import CLIPModel
from src.fitter import CustomTrainer
from src.callbacks import wandb_update
from src.utils import seed_everything

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
        model_config:str="input/model_config.json",
        freeze_training_config:str="input/freeze_training_config.json",
        training_config:str="input/training_config.json",
        wandb_config:str="input/wandb_config.json",
        ):

    #
    # Part I: Read configuration files & environment variables
    #
    
    # Env. variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Freeze Pretraining
    with open(freeze_training_config) as f:
        freeze_train_dct = json.load(f)
        seed_everything(freeze_train_dct['seed'])
    # Pretraining
    with open(training_config) as f:
        train_dct = json.load(f)
    #Wandb
    with open(wandb_config) as f:
        wandb_dct = json.load(f)
        os.environ['WANDB_API_KEY'] = wandb_dct['WB_KEY']
        os.environ['WANDB_USERNAME'] = wandb_dct['WB_ENTITY']
        os.environ['WANDB_PROJECT'] = wandb_dct['WB_PROJECT']
    
    #
    # Part II: Setup data and model for pretraining with frozen backbones
    #
    
    # Get tools
    log.info(f"Setup tools")
    train_df, val_df, key2name = setup_data(model_config)
    #Model config
    with open(model_config) as f:
        model_dct = json.load(f)
    # Build datasets
    log.info(f"Prepare datasets:")
    train_dts = CLIPDataset(train_df, model_dct['text_backbone'], model_dct['max_length_corpus'], key2name)
    val_dts = CLIPDataset(val_df, model_dct['text_backbone'], model_dct['max_length_corpus'], key2name)
    # Check if freeze training is expected
    if freeze_train_dct['epochs']>0:
        # Define model
        log.info(f"Get model:")
        model = CLIPModel(model_dct['vision_backbone'], model_dct['text_backbone'], True, model_dct['dim'], model_dct['dropout'], freeze_train_dct['device'])
        # Set up arguments
        log.info(f"Prepare training arguments:")
        steps_per_epoch = len(train_df)/(freeze_train_dct['batch_size']*freeze_train_dct['gradient_accumulation_steps'])
        logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
        logging_steps = logging_steps//4
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(os.getcwd(),freeze_train_dct['output_dir']),
            gradient_accumulation_steps=freeze_train_dct['gradient_accumulation_steps'],
            warmup_steps=logging_steps*freeze_train_dct['warmup_steps_factor'],
            learning_rate=freeze_train_dct['learning_rate'],
            weight_decay=freeze_train_dct['weight_decay'],
            per_device_train_batch_size=freeze_train_dct['batch_size'],
            per_device_eval_batch_size=freeze_train_dct['batch_size'],
            dataloader_num_workers = freeze_train_dct['dataloader_num_workers'],
            num_train_epochs=freeze_train_dct['epochs'],
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=logging_steps,
            logging_strategy="steps",
            logging_steps=logging_steps,
            report_to="wandb",  # enable logging to W&B
            run_name='pretraining_freeze',
            seed=freeze_train_dct['seed'],
            fp16=bool(freeze_train_dct['fp16'])
        )
        # Initialise Trainer
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=train_dts,
            eval_dataset=val_dts,
        )
        # Trainer
        log.info(f"Start model training:")
        trainer.train()
        # Save best state_dict
        torch.save(trainer.model.state_dict(), 'output/best_checkpoint_freeze.bin')
        if train_dct['epochs']==0:
            os.replace(f"output/best_checkpoint_freeze.bin", f"{wandb.run.dir}/best_checkpoint_freeze.bin")
        else:
            log.info(f"Not saving freeze backbone weights in WB as complete pretraining is expected.")
        del model, trainer, training_args
        wandb.finish()
    else:
        log.info(f"Freeze pretraining is not expected, jumping straight to standard pretraining.")
    
    #
    # Part III: Setup data and model for pretraining with active backbones
    #
    
    # Pretraining
    seed_everything(train_dct['seed'])
    # Check if standard pretraining is expected
    if train_dct['epochs']>0:
        # Define model
        log.info(f"Get model:")
        model = CLIPModel(model_dct['vision_backbone'], model_dct['text_backbone'], False, model_dct['dim'], model_dct['dropout'], train_dct['device'])
        if freeze_train_dct['epochs']>0:
            model.load_state_dict(torch.load('output/best_checkpoint_freeze.bin'))
        # Set up arguments
        log.info(f"Prepare training arguments:")
        steps_per_epoch = len(train_df)/(train_dct['batch_size']*train_dct['gradient_accumulation_steps'])
        logging_steps = steps_per_epoch if int(steps_per_epoch)==steps_per_epoch else int(steps_per_epoch)+1
        logging_steps = logging_steps//4
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(os.getcwd(),train_dct['output_dir']),
            gradient_accumulation_steps=train_dct['gradient_accumulation_steps'],
            warmup_steps=logging_steps*train_dct['warmup_steps_factor'],
            learning_rate=train_dct['learning_rate'],
            weight_decay=train_dct['weight_decay'],
            per_device_train_batch_size=train_dct['batch_size'],
            per_device_eval_batch_size=train_dct['batch_size'],
            dataloader_num_workers = train_dct['dataloader_num_workers'],
            num_train_epochs=train_dct['epochs'],
            load_best_model_at_end=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=logging_steps,
            logging_strategy="steps",
            logging_steps=logging_steps,
            report_to="wandb",  # enable logging to W&B
            run_name='pretraining',
            seed=freeze_train_dct['seed'],
            fp16=bool(freeze_train_dct['fp16'])
        )
        # Initialise Trainer
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=train_dts,
            eval_dataset=val_dts,
        )
        # Trainer
        log.info(f"Start model training:")
        trainer.train()
        # Save best state_dict
        torch.save(trainer.model.state_dict(), 'output/best_checkpoint.bin')
        os.replace(f"output/best_checkpoint.bin", f"{wandb.run.dir}/best_checkpoint.bin")
        del model, trainer, training_args
        wandb.finish()
    else:
        log.info(f"Complete pretraining is not expected, so this is the end of training.")

if __name__=="__main__":
    fire.Fire(main)
