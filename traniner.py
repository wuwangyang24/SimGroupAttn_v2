import os
import re
import wandb
import torch
from torch import nn
from lightning.pytorch.loggers import WandbLogger
from pl_module import LightningModel
import warnings
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, config):
        self.config = config
        self.name = f"ViT-{config.Model_Configs.depth}-H{config.Model_Configs.num_heads}-HDim{config.Model_Configs.dim_head}-D{config.Model_Configs.embed_dim}-S{config.Data.img_size}-Grouped{config.Data.stratified}-M{config.Similarity_Configs.n_edges_self_image}-N{config.Similarity_Configs.n_edges_other_images}-MaskT{config.Training_Dynamics.mask.mask_strategy}-MaskR{config.Training_Dynamics.mask.mask_ratio}-B{config.Training_Dynamics.batch_size}-AccuG{config.Training_Dynamics.accumulate_grad_batches}-LR{config.Training_Dynamics.optimizer.lr}-CLS{config.Model_Configs.cls_token}-AttnM{config.Model_Configs.masked_attn}-ConnM{config.Similarity_Configs.conn_with_mask}"
        self.wandb_logger = WandbLogger(log_model=True,
                                        entity=config.Logging.entity,
                                        project=config.Logging.project,
                                        name=self.name,
                                        resume='allow'
                                       )

        self.ppl = self.load_ppl(config.Pipeline)
        self.pl_module = LightningModel(self.ppl, config.Train)
        self.wandb_logger.watch(self.ppl, log="gradients", log_freq=1000)
        checkpoint_dirpath = os.path.join(self.config.Logging.checkpoint_path, self.name)
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                                                                   dirpath=checkpoint_dirpath,
                                                                   filename='checkpoint_{epoch}',
                                                                   every_n_epochs=self.config.Logging.save_every_n_epoch,
                                                                   save_top_k=1,
                                                                  )
        self.pl_trainer = pl.Trainer(accelerator="gpu",
                                     strategy='ddp_find_unused_parameters_true',
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

    def load_ppl(self, ppl_config):
        if ppl_config.name == 'ijepa':
            from transformers import AutoProcessor, IJepaConfig, IJepaModel
            cfg = IJepaConfig(
                image_size=ppl_config.img_size,
                patch_size=ppl_config.patch_size,
                num_channels=ppl_config.chans,
                hidden_size=ppl_config.D,
                num_hidden_layers=ppl_config.layers,
                num_attention_heads=ppl_config.heads,
                mlp_ratio=ppl_config.mlp_ratio,
            )
            ppl = IJepaModel(cfg)
            processor = AutoProcessor.from_pretrained(ppl_config.id)
            return ppl, processor
        elif ppl_config.name == 'mae':
            from transformers import MaeConfig, MaeModel
            cfg = MaeConfig(
                image_size=ppl_config.img_size,
                patch_size=ppl_config.patch_size,
                num_channels=ppl_config.chans,
                encoder_layers=ppl_config.enc_layers,
                encoder_attention_heads=ppl_config.enc_heads,
                encoder_hidden_size=ppl_config.enc_D,
                decoder_layers=ppl_config.dec_layers,
                decoder_attention_heads=ppl_config.dec_heads,
                decoder_hidden_size=ppl_config.dec_D,
                mask_ratio=ppl_config.mlp_ratio,
            )
            ppl = MaeModel(cfg)
            return ppl

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