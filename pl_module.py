import torch
import lightning as pl
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from typing import Any, Optional, Sequence, Dict


class LightningModel(pl.LightningModule):

    def __init__(self, ppl: Optional[Sequence[Any]], train_config: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['ppl'])
        self.ppl = ppl
        # Extract optimizer configuration
        optimizer_config = train_config.optimizer
        self.max_epochs = train_config.epochs
        self.lr = optimizer_config.lr
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
        loss = self.ppl(batch).loss
        print(loss)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}


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