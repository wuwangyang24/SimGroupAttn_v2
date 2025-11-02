import os
import re
from typing import Optional
import warnings
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from pl_module import LightningModel
from pipeline import load_ppl
# Suppress warnings globally
warnings.filterwarnings("ignore")


class Trainer:
    """High-level trainer for Vision Transformer / Data2Vec / JEPA models using Lightning."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.name = self._generate_experiment_name(config)
        self.wandb_logger = self._init_wandb_logger()
        self.ppl = load_ppl(config.Pipeline)
        self.pl_module = LightningModel(self.ppl, config.Train)
        self.wandb_logger.watch(self.ppl, log="gradients", log_freq=1000)
        self.resume_checkpoint = self._find_latest_checkpoint(self._checkpoint_dir)
        self.pl_trainer = self._init_pl_trainer()

    @property
    def _checkpoint_dir(self) -> str:
        """Construct checkpoint directory path."""
        return os.path.join(self.config.Logging.checkpoint_path, self.name)

    def _init_wandb_logger(self) -> WandbLogger:
        """Initialize Weights & Biases logger."""
        return WandbLogger(
            log_model=True,
            entity=self.config.Logging.entity,
            project=self.config.Logging.project,
            name=self.name,
            resume='allow'
        )

    def _generate_experiment_name(self, cfg: DictConfig) -> str:
        """Generate a structured experiment name from configuration."""
        parts = [
            f"ViT-{cfg.Model_Configs.depth}",
            f"H{cfg.Model_Configs.num_heads}",
            f"HDim{cfg.Model_Configs.dim_head}",
            f"D{cfg.Model_Configs.embed_dim}",
            f"S{cfg.Data.img_size}",
            f"Grouped{cfg.Data.stratified}",
            f"M{cfg.Similarity_Configs.n_edges_self_image}",
            f"N{cfg.Similarity_Configs.n_edges_other_images}",
            f"MaskT{cfg.Training_Dynamics.mask.mask_strategy}",
            f"MaskR{cfg.Training_Dynamics.mask.mask_ratio}",
            f"B{cfg.Training_Dynamics.batch_size}",
            f"AccuG{cfg.Training_Dynamics.accumulate_grad_batches}",
            f"LR{cfg.Training_Dynamics.optimizer.lr}",
            f"CLS{cfg.Model_Configs.cls_token}",
            f"AttnM{cfg.Model_Configs.masked_attn}",
            f"ConnM{cfg.Similarity_Configs.conn_with_mask}",
        ]
        return "-".join(parts)

    def _find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Return path to latest checkpoint or None if not found."""
        if not os.path.exists(checkpoint_dir):
            print("No checkpoint directory available")
            return None
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if not checkpoints:
                print("No checkpoint files found")
                return None
            latest_ckpt = max(checkpoints, key=lambda f: int(re.findall(r'\d+', f)[-1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
            print(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"Error finding checkpoint: {e}")
            return None

    def _init_pl_trainer(self) -> pl.Trainer:
        """Initialize PyTorch Lightning Trainer with optimized settings."""
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=self._checkpoint_dir,
            filename='checkpoint_{epoch}',
            every_n_epochs=self.config.Logging.save_every_n_epoch,
            save_top_k=1,
        )

        return pl.Trainer(
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
                pl.pytorch.callbacks.ModelSummary(max_depth=6),
            pl.pytorch.callbacks.DeviceStatsMonitor(),
            ],
            enable_model_summary=True,
            # deterministic=True,
            profiler="simple",
        )

    def train(self, data_module: pl.LightningDataModule) -> None:
        """Train the model with checkpoint resumption."""
        try:
            print("Starting training...")
            self.pl_trainer.fit(
                model=self.pl_module,
                datamodule=data_module,
                ckpt_path=self.resume_checkpoint
            )
            print("Training completed successfully.")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            import wandb
            wandb.finish()
