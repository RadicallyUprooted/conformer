import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model.conformer import Conformer
from data.dataset import LibriSpeechDataModule, CharTextTransform


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i.item()] for i in indices])


class ConformerLightningModule(pl.LightningModule):
    """
    A LightningModule that encapsulates the Conformer model, training logic,
    and optimizer configuration.
    """
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig, data_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = Conformer(
            vocab_size=data_cfg.vocab_size,
            n_layers=model_cfg.n_layers,
            n_mels=data_cfg.n_mels,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            conv_kernel_size=model_cfg.conv_kernel_size,
            dropout=model_cfg.dropout,
        )

        self.criterion = nn.CTCLoss(blank=data_cfg.vocab_size - 1, zero_infinity=True)
        self.learning_rate = train_cfg.learning_rate
        self.text_transform = CharTextTransform()
        self.decoder = GreedyCTCDecoder(self.text_transform.index_map, blank=data_cfg.vocab_size - 1)

    def forward(self, inputs, input_lengths):
        return self.model(inputs, input_lengths)

    def training_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths, _ = batch
        outputs, output_lengths = self(inputs, input_lengths)
        
        log_probs = outputs.permute(1, 0, 2) # (T, B, C)
        
        loss = self.criterion(log_probs, targets, output_lengths, target_lengths)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths, waveforms = batch
        outputs, output_lengths = self(inputs, input_lengths)
        
        log_probs = outputs.permute(1, 0, 2) # (T, B, C)
        
        loss = self.criterion(log_probs, targets, output_lengths, target_lengths)
        
        self.log('val_loss', loss)

        if batch_idx == 0:
            waveform = waveforms[0].cpu()
            ground_truth_text = self.text_transform.int_to_text(targets[0][:target_lengths[0]].cpu().numpy())
            
            pred_text = self.decoder(outputs[0][:output_lengths[0]].cpu())

            self.logger.experiment.log({
                "examples": [
                    wandb.Audio(waveform.numpy(), caption=f"Pred: {pred_text} | GT: {ground_truth_text}", sample_rate=16000)
                ]
            })

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    data_module = LibriSpeechDataModule(
        path=cfg.data.path,
        train_url=cfg.data.train_url,
        val_url=cfg.data.val_url,
        batch_size=cfg.train.batch_size, 
        n_mels=cfg.data.n_mels, 
        vocab_size=cfg.data.vocab_size
    )

    model_module = ConformerLightningModule(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        data_cfg=cfg.data,
    )

    wandb_logger = WandbLogger(project="conformer", name="conformer-run")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="conformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model_module, data_module)

if __name__ == '__main__':
    main()
