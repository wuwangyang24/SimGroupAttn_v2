import os
import re
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from pl_module import LightningModel
from pipeline import load_ppl
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, config: DictConfig) -> None:
        """Initialize the trainer with configuration.
        
        Args:
            config: Configuration object containing all training parameters
        """
        self.config = config
        self.name = self._generate_experiment_name(config)
        self.wandb_logger = WandbLogger(
            log_model=True,
            entity=config.Logging.entity,
            project=config.Logging.project,
            name=self.name,
            resume='allow'
        )
        self.ppl = load_ppl(config.Pipeline)
        self.pl_module = LightningModel(self.ppl, config.Train)
        self.wandb_logger.watch(self.ppl, log="gradients", log_freq=1000)
        checkpoint_dirpath = os.path.join(self.config.Logging.checkpoint_path, self.name)
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                                                                    dirpath=checkpoint_dirpath,
                                                                    filename='checkpoint_{epoch}',
                                                                    every_n_epochs=self.config.Logging.save_every_n_epoch,
                                                                    save_top_k=1,
                                                                    )
        # Initialize trainer with optimized settings
        self.pl_trainer = pl.Trainer(
            accelerator="gpu",
            devices=self.config.device,
            strategy='ddp_find_unused_parameters_true',
            accumulate_grad_batches=self.config.Training_Dynamics.accumulate_grad_batches,
            max_epochs=self.config.Training_Dynamics.epochs,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=self.config.Training_Dynamics.val_every_n_epoch,
            logger=self.wandb_logger,
            log_every_n_steps=1,
            precision="16-mixed",
            callbacks=[
                checkpoint_callback,
                pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
                pl.pytorch.callbacks.ModelSummary(max_depth=2),
                pl.pytorch.callbacks.DeviceStatsMonitor(),
            ],
            enable_model_summary=True,
            deterministic=True,  # For reproducibility
            profiler="simple"  # For performance monitoring
        )
        self.resume_checkpoint = self._find_latest_checkpoint(checkpoint_dirpath)

    def _generate_experiment_name(self, config: DictConfig) -> str:
        """Generate a structured experiment name from config parameters.
        
        Args:
            config: Configuration object containing model and training parameters
            
        Returns:
            Formatted experiment name string
        """
        return (f"ViT-{config.Model_Configs.depth}"
                f"-H{config.Model_Configs.num_heads}"
                f"-HDim{config.Model_Configs.dim_head}"
                f"-D{config.Model_Configs.embed_dim}"
                f"-S{config.Data.img_size}"
                f"-Grouped{config.Data.stratified}"
                f"-M{config.Similarity_Configs.n_edges_self_image}"
                f"-N{config.Similarity_Configs.n_edges_other_images}"
                f"-MaskT{config.Training_Dynamics.mask.mask_strategy}"
                f"-MaskR{config.Training_Dynamics.mask.mask_ratio}"
                f"-B{config.Training_Dynamics.batch_size}"
                f"-AccuG{config.Training_Dynamics.accumulate_grad_batches}"
                f"-LR{config.Training_Dynamics.optimizer.lr}"
                f"-CLS{config.Model_Configs.cls_token}"
                f"-AttnM{config.Model_Configs.masked_attn}"
                f"-ConnM{config.Similarity_Configs.conn_with_mask}")

    def _find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Find the latest checkpoint in the directory.
        
        Args:
            checkpoint_dir: Path to the directory containing checkpoints
            
        Returns:
            Path to the latest checkpoint file or None if no checkpoints found
        """
        if not os.path.exists(checkpoint_dir):
            print("No checkpoint directory available")
            return None
            
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if not checkpoints:
                print("No checkpoint files found")
                return None
                
            latest = max([int(re.findall(r'\d+', ckpt)[-1]) for ckpt in checkpoints])
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch={latest}.ckpt')
            print(f"Found checkpoint at epoch {latest}")
            return checkpoint_path
        except Exception as e:
            print(f"Error finding checkpoint: {str(e)}")
            return None

    
    def train(self) -> None:
        """Train the model with automatic checkpoint resumption.
        
        This method handles the entire training process including:
        - Setting up dataloaders
        - Configuring the training environment
        - Handling checkpointing
        - Managing the training lifecycle
        
        Raises:
            Exception: If training fails for any reason
        """
        try:
            # Create dataloaders
            print("Creating dataloader...")
            train_loader, val_loader = None,  None

            # Training
            print("Training model...")
            self.pl_trainer.fit(
                model=self.pl_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=self.resume_checkpoint if self.resume_checkpoint else None
            )
            
            print("Training completed successfully")
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
            
        finally:
            wandb.finish()