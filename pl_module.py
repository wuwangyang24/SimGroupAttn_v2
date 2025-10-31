import torch
import wandb
import random
import lightning as pl
import torchvision.utils as vutils
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class LightningModel(pl.LightningModule):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.lr = config.Training_Dynamics.optimizer.lr
        self.beta1 = config.Training_Dynamics.optimizer.beta1
        self.beta2 = config.Training_Dynamics.optimizer.beta2
        self.weight_decay = config.Training_Dynamics.optimizer.weight_decay
        self.eta_min = config.Training_Dynamics.optimizer.eta_min
        self.max_epochs = config.Training_Dynamics.epochs
        self.warmup_epochs = config.Training_Dynamics.optimizer.warmup_epochs
        self.start_factor = config.Training_Dynamics.optimizer.start_factor
        self.num_samples = config.Logging.num_samples
        # self.sample_images = []

    def training_step(self, batch, batch_idx):
        # x = batch
        loss, _, _, _ = self.model(batch)
        self.log("train_loss", loss.detach().cpu().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recons, x, _ = self.model(batch)
        self.log("val_loss", loss.detach().cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        N = recons.shape[0]
        sample_images = []
        for n in random.sample(range(N), min(N, 10)):
            original_img = x[n]
            recon_img = torch.clamp(recons[n], min=0., max=1.)
            sample_images.extend([original_img, recon_img])
        grid_img = vutils.make_grid(sample_images, nrow=2, normalize=True, scale_each=True)
        self.logger.experiment.log({
            "val/recon_images": wandb.Image(grid_img, caption="Original | Recon Grid"),
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
