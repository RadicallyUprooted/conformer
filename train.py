import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import wandb
import random
import gc
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.text import CharErrorRate, WordErrorRate
from model.conformer import Conformer
from data.custom_dataloader import DataModule
from text_processor.processor import CharTextTransform

class ConformerLightningModule(pl.LightningModule):
    """
    A LightningModule that encapsulates the Conformer model, training logic,
    and optimizer configuration.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.text_transform = CharTextTransform()
        self.model = Conformer(
            vocab_size=self.cfg.data.vocab_size,
            n_mels=self.cfg.data.n_mels,
            n_layers=self.cfg.model.n_layers,
            d_model=self.cfg.model.d_model,
            n_heads=self.cfg.model.n_heads,
            conv_kernel_size=self.cfg.model.conv_kernel_size,
            dropout=self.cfg.model.dropout,
        )

        self.criterion = nn.CTCLoss(blank=self.text_transform.blank, zero_infinity=True)
        self.learning_rate = self.cfg.train.learning_rate
        self.decoder = hydra.utils.instantiate(self.cfg.decoder, text_transform=self.text_transform, blank='<blank>')
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()
        self.optimizer_cfg = self.cfg.optimizer

    def forward(self, inputs, input_lengths):
        return self.model(inputs, input_lengths)

    def training_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths, _ = batch
        outputs, output_lengths = self(inputs, input_lengths)
        
        log_probs = outputs.permute(1, 0, 2) # (T, B, C)
        
        loss = self.criterion(log_probs, targets, output_lengths, target_lengths)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths, waveforms = batch
        outputs, output_lengths = self(inputs, input_lengths)
        
        log_probs = outputs.permute(1, 0, 2) # (T, B, C)
        
        loss = self.criterion(log_probs, targets, output_lengths, target_lengths)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        decoded_preds = []
        decoded_targets = []
        for i in range(outputs.size(0)):
            pred = self.decoder(outputs[i][:output_lengths[i]].cpu())
            decoded_preds.append(pred)
            
            target = self.text_transform.int_to_text(targets[i][:target_lengths[i]].cpu().numpy())
            decoded_targets.append(target)

        self.wer.update(decoded_preds, decoded_targets)
        self.cer.update(decoded_preds, decoded_targets)

        if batch_idx == 0:
            batch_size = outputs.size(0)
            random_idx = random.randint(0, batch_size - 1)

            waveform = waveforms[random_idx].cpu()
            ground_truth_text = self.text_transform.int_to_text(targets[random_idx][:target_lengths[random_idx]].cpu().numpy())
            
            pred_text = self.decoder(outputs[random_idx][:output_lengths[random_idx]].cpu())

            self.logger.experiment.log({
                "examples": [
                    wandb.Audio(waveform.squeeze().numpy(), caption=f"Pred: {pred_text} | GT: {ground_truth_text}", sample_rate=16000)
                ]
            })

        return loss
    
    def on_validation_epoch_end(self):
        avg_wer = self.wer.compute()
        avg_cer = self.cer.compute()
        
        self.log('val_wer', avg_wer, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_cer', avg_cer, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.wer.reset()
        self.cer.reset()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        return optimizer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    data_module = DataModule(cfg=cfg)
    model_module = ConformerLightningModule(cfg=cfg)

    wandb_logger = WandbLogger(project="conformer", name="conformer-run")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="conformer-{epoch:02d}",
        every_n_epochs=10,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model_module, data_module, ckpt_path=cfg.train.checkpoint)

if __name__ == '__main__':
    main()
