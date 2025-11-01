import torch
import wandb
import random
import lightning as pl
import torchvision.utils as vutils
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class LightningModel(pl.LightningModule):
    """Lightning module for training with optimized configuration and monitoring."""

    def __init__(self, ppl, train_config):
        super().__init__()
        self.save_hyperparameters(ignore=['ppl'])
        self.ppl = ppl
        
        # Extract optimizer configuration
        optimizer_config = train_config.Training_Dynamics.optimizer
        self.lr = optimizer_config.lr
        self.beta1 = optimizer_config.beta1
        self.beta2 = optimizer_config.beta2
        self.weight_decay = optimizer_config.weight_decay
        self.eta_min = optimizer_config.eta_min
        self.warmup_epochs = optimizer_config.warmup_epochs
        self.start_factor = optimizer_config.start_factor
        
        # Training configuration
        self.max_epochs = train_config.Training_Dynamics.epochs
        
        # Initialize metrics
        self.train_loss = 0.0
        self.val_loss = 0.0
        
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.model(batch)
        self.log("train_loss", loss.detach().cpu().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized image logging."""
        loss, recons, x, _ = self.model(batch)
        
        # Log validation loss
        self.log("val_loss", loss.detach().cpu().item(), 
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Log sample images only on the first process in distributed training
        if self.global_rank == 0 and batch_idx == 0:  # Log only first batch
            N = min(recons.shape[0], 10)  # Limit to max 10 samples
            indices = random.sample(range(recons.shape[0]), N)
            
            # Process images in batches
            sample_images = []
            for idx in indices:
                original_img = x[idx]
                recon_img = torch.clamp(recons[idx], min=0., max=1.)
                sample_images.extend([original_img, recon_img])
            
            # Create and log image grid
            with torch.no_grad():
                grid_img = vutils.make_grid(
                    sample_images, 
                    nrow=2,
                    normalize=True,
                    scale_each=True
                )
                
            self.logger.experiment.log({
                "val/recon_images": wandb.Image(
                    grid_img,
                    caption=f"Original | Recon Grid (Epoch {self.current_epoch})"
                ),
                "global_step": self.global_step
            })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.start_factor,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.eta_min
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": None,
            }
        }
