import os
import sys
import yaml
import argparse
from Trainer import Trainer
from pl_module import LightningModel
from dataloader import ImageDataModule
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training script for SimGroupAttn_v2')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
config_path = args.config
if not os.path.exists(config_path):
    print(f'Configuration file {config_path} not found.')
    sys.exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize Weights and Biases
wandb.login(key="YOUR_API_KEY")

# Initialize DataModule
data_module = ImageDataModule(config['Data'])

# Intialize Trainer
trainer = Trainer(config, data_module)

# Start training
if __name__ == '__main__':
    trainer.train()