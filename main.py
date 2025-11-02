import os
import argparse
import yaml
import wandb
from traniner import Trainer
from dataloader import ImageDataModule

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    trainer = Trainer(config, data_module)

    # Start training
    trainer.train()

if __name__ == '__main__':
    main()
