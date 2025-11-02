import os
import argparse
from omegaconf import OmegaConf
import wandb
from traniner import Trainer
from dataloader import ImageDataModule

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    config = OmegaConf.load(config_path)
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training script for SimGroupAttn_v2')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize Weights and Biases
    # fallback to hardcoded key if not set
    wandb_key = os.environ.get("WANDB_API_KEY", "e8af882d14d8408f2bbb2c220c22c9499151647f")  
    wandb.login(key=wandb_key)

    # Initialize DataModule
    data_module = ImageDataModule(config['Data'])

    # Initialize Trainer
    trainer = Trainer(config)

    # Start training
    trainer.train(data_module)

if __name__ == '__main__':
    main()
