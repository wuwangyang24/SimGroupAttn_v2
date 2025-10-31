import os
import re
import wandb
import lightning as pl
from glob import glob
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, config):
        self.config = config
        self.name = f"ViT-{config.Model_Configs.depth}-H{config.Model_Configs.num_heads}-HDim{config.Model_Configs.dim_head}-D{config.Model_Configs.embed_dim}-S{config.Data.img_size}-Grouped{config.Data.stratified}-M{config.Similarity_Configs.n_edges_self_image}-N{config.Similarity_Configs.n_edges_other_images}-MaskT{config.Training_Dynamics.mask.mask_strategy}-MaskR{config.Training_Dynamics.mask.mask_ratio}-B{config.Training_Dynamics.batch_size}-AccuG{config.Training_Dynamics.accumulate_grad_batches}-LR{config.Training_Dynamics.optimizer.lr}-CLS{config.Model_Configs.cls_token}-AttnM{config.Model_Configs.masked_attn}-ConnM{config.Similarity_Configs.conn_with_mask}"
        # setting up wandb logging
        self.wandb_logger = WandbLogger(log_model=True,
                                        entity=config.Logging.entity,
                                        project=config.Logging.project,
                                        name=self.name,
                                        resume='allow'
                                       )

        model = self.create_model()
        self.wandb_logger.watch(model, log="gradients", log_freq=1000)
        self.pl_module = LightningModel(model, self.config)
        checkpoint_dirpath = os.path.join(self.config.Logging.checkpoint_path, self.name)
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                                                                   dirpath=checkpoint_dirpath,
                                                                   filename='checkpoint_{epoch}',
                                                                   every_n_epochs=self.config.Logging.save_every_n_epoch,
                                                                   save_top_k=1,
                                                                  )
        devices = torch.cuda.device_count()
        self.pl_trainer = pl.Trainer(accelerator="gpu",
                                     strategy='ddp_find_unused_parameters_true',
                                     devices=self.config.device,
                                     accumulate_grad_batches=self.config.Training_Dynamics.accumulate_grad_batches,
                                     max_epochs=self.config.Training_Dynamics.epochs,
                                     gradient_clip_val=1.0,
                                     check_val_every_n_epoch=self.config.Training_Dynamics.val_every_n_epoch, 
                                     logger=self.wandb_logger,
                                     log_every_n_steps=1,
                                     precision="16-mixed",
                                     callbacks=[checkpoint_callback])
        try:
            self.current_epoch = max([int(re.findall(r'\d+', p)[-1]) for p in glob(f'{checkpoint_dirpath}/**')])
            self.resume_checkpoint = f'{checkpoint_dirpath}/checkpoint_epoch={self.current_epoch}.ckpt'
            print("Checkpoint available")
        except Exception as e:
            print("No checkpoint available")
            self.resume_checkpoint = 'None'

    def create_model(self):
        print('Initializing model ...')
        # create encoder
        encoder = VisionTransformerSimMIM(
            self.config.Data.img_size,
            self.config.Data.patch_size,
            self.config.Data.channels,
            self.config.Model_Configs.dim_head,
            self.config.Model_Configs.embed_dim,
            self.config.Model_Configs.depth,
            self.config.Model_Configs.num_heads,
            self.config.Model_Configs.mlp_ratio,
            self.config.Model_Configs.drop_rate,
            self.config.Model_Configs.drop_path_rate,
            self.config.Model_Configs.cls_token,
            self.config.Model_Configs.use_abs_pos_emb,
            self.config.Similarity_Configs.post_sim
        )
        # create decoder
        decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.config.Model_Configs.embed_dim,
                out_channels=self.config.Data.encoder_stride ** 2 * 3,
                kernel_size=1
            ),
            nn.PixelShuffle(self.config.Data.encoder_stride)
        )
        # create simmim model
        model = SimMIM(encoder,
                       decoder,
                       self.config.Training_Dynamics.batch_size,
                       self.config.Data.img_size,
                       self.config.Data.num_patches,
                       self.config.Data.patch_size,
                       self.config.Data.channels,
                       self.config.Model_Configs.embed_dim,
                       self.config.Training_Dynamics.mask.mask_strategy,
                       self.config.Training_Dynamics.mask.mask_ratio,
                       self.config.Similarity_Configs.n_edges_self_image,
                       self.config.Similarity_Configs.n_edges_other_images,
                       self.config.Similarity_Configs.post_sim,
                       self.config.Similarity_Configs.conn_with_mask,
                       self.config.Model_Configs.masked_attn,
                       self.config.Model_Configs.cls_token
                      )
        return model


    def train(self):
        # Create dataloaders
        print("Creating dataloader...")
        train_loader, val_loader = create_dataloaders(self.config.Data.root_folder,
                                                      self.config.Data.val_ratio,
                                                      self.config.Training_Dynamics.batch_size,
                                                      self.config.Data.img_size,
                                                      self.config.Data.stratified
                                                     )
        print("Training model...")
        if os.path.isfile(self.resume_checkpoint):
            print(f"Start from checkpoint at epoch {self.current_epoch}")
            self.pl_trainer.fit(self.pl_module,
                                train_dataloaders=train_loader,
                                val_dataloaders=val_loader,
                                ckpt_path=self.resume_checkpoint
                               )
        else:
            try:
                self.pl_trainer.fit(self.pl_module,
                                    train_dataloaders=train_loader,
                                    val_dataloaders=val_loader
                                   )
            except Exception as e:
                print(e)
        wandb.finish()
        print("Training finished")