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
from data.dataset import LibriSpeechDataModule
from text_processor.processor import CharTextTransform

class ConformerLightningModule(pl.LightningModule):
    """
    A LightningModule that encapsulates the Conformer model, training logic,
    and optimizer configuration.
    """
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, data_cfg: DictConfig, optimizer_cfg: DictConfig, decoder_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.text_transform = CharTextTransform()
        self.model = Conformer(
            vocab_size=data_cfg.vocab_size,
            n_layers=model_cfg.n_layers,
            n_mels=data_cfg.n_mels,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            conv_kernel_size=model_cfg.conv_kernel_size,
            dropout=model_cfg.dropout,
        )

        self.criterion = nn.CTCLoss(blank=self.text_transform.blank, zero_infinity=True)
        self.learning_rate = train_cfg.learning_rate
        self.decoder = hydra.utils.instantiate(decoder_cfg, text_transform=self.text_transform, blank='<blank>')
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()
        self.optimizer_cfg = optimizer_cfg

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
                    wandb.Audio(waveform.numpy(), caption=f"Pred: {pred_text} | GT: {ground_truth_text}", sample_rate=16000)
                ]
            })

        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log average WER and CER
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

    data_module = LibriSpeechDataModule(
        path=cfg.data.path,
        train_url=cfg.data.train_url,
        val_url=cfg.data.val_url,
        batch_size=cfg.train.batch_size, 
        n_mels=cfg.data.n_mels, 
        vocab_size=cfg.data.vocab_size,
        time_mask_param=cfg.data.time_mask_param,
        freq_mask_param=cfg.data.freq_mask_param,
        num_workers=cfg.data.num_workers,
    )

    model_module = ConformerLightningModule(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        data_cfg=cfg.data,
        optimizer_cfg=cfg.optimizer,
        decoder_cfg=cfg.decoder,
    )

    wandb_logger = WandbLogger(project="conformer", name="conformer-run")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="conformer-{epoch:02d}",
        every_n_epochs=cfg.train.epochs,
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
