import torch
import wandb
import lightning as pl
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from typing import Any, Optional, Sequence, Dict


class LightningModel(pl.LightningModule):

    def __init__(self, ppl: Any, config: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['ppl'])
        self.ppl = ppl
        # Extract optimizer configuration
        optimizer_config = config.optimizer
        self.max_epochs = config.training.max_epochs
        self.lr = optimizer_config.learning_rate
        self.beta1 = optimizer_config.beta1
        self.beta2 = optimizer_config.beta2
        self.weight_decay = optimizer_config.weight_decay
        self.eta_min = optimizer_config.eta_min
        self.warmup_epochs = optimizer_config.warmup_epochs
        self.start_factor = optimizer_config.start_factor
        
    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.ppl(batch).loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Read current LR from trainer's optimizer (cheap) and log as scalar
        try:
            current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except Exception:
            # If trainer/optimizer isn't available yet, skip logging the lr for this step
            pass
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        # Forward pass
        outputs = self.ppl(batch, return_attn=True)
        loss = outputs.loss
        attn_scores = outputs.get('attn_scores', None)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_attention_maps(batch, batch_idx, attn_scores, loss)
        return {"val_loss": loss}

    def log_attention_maps(self, batch: Any, batch_idx: int, attn_scores: Optional[torch.Tensor]) -> None:
        """Log attention maps overlaid on input images to WandB."""
        # --- Log first N images and attention maps to WandB ---
        N = 8  # number of images to log
        if attn_scores is not None and batch_idx == 0:  # only first batch
            images = batch["image"]  # shape [B, C, H, W]
            batch_size = images.shape[0]
            N = min(N, batch_size)
            # CLS token attention to patches
            cls_attn = attn_scores[:, :, 0, 1:]  # [B, heads, tokens]
            cls_attn = cls_attn.mean(dim=1)      # average over heads, shape [B, tokens]
            # Compute patch grid size dynamically
            patch_size = int(cls_attn.shape[1] ** 0.5)
            cls_attn_map = cls_attn.reshape(batch_size, patch_size, patch_size)  # [B, H, W]
            cls_attn_map = torch.nn.functional.interpolate(
                cls_attn_map.unsqueeze(1),  # add channel dim
                size=(images.shape[2], images.shape[3]),  # upsample to original size
                mode='bilinear',
                align_corners=False
            )
            # Normalize attention maps
            cls_attn_map = (cls_attn_map - cls_attn_map.min(dim=(1,2,3), keepdim=True)[0]) / (
                        cls_attn_map.max(dim=(1,2,3), keepdim=True)[0] - cls_attn_map.min(dim=(1,2,3), keepdim=True)[0] + 1e-8)
            for i in range(N):
                img = images[i].cpu()  # [C, H, W]
                attn = cls_attn_map[i].cpu()  # [1, H, W]
                # Convert image to [H, W, C] for overlay
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize 0-1
                attn_np = attn.squeeze(0).numpy()  # [H, W]
                # Overlay attention on image
                overlay = 0.6 * img_np + 0.4 * plt.cm.jet(attn_np)[..., :3]
                overlay = overlay.clip(0, 1)
                # Log to WandB
                self.logger.experiment.log({
                    f"val_image_{i}": wandb.Image((img_np * 255).astype("uint8")),
                    f"val_attention_overlay_{i}": wandb.Image((overlay * 255).astype("uint8")),
                    "epoch": self.current_epoch
                })

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.ppl.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.start_factor,
            total_iters=self.warmup_epochs
        )
        # Ensure T_max is positive to avoid scheduler errors when warmup == max_epochs
        t_max = max(1, int(self.max_epochs - self.warmup_epochs))
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=self.eta_min,
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