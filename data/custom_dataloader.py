import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from text_processor.processor import CharTextTransform
from omegaconf import DictConfig

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.text_transform = CharTextTransform()

    def setup(self, stage=None):
        self.train_dataset = hydra.utils.instantiate(self.cfg.data.train)
        self.val_dataset = hydra.utils.instantiate(self.cfg.data.val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def collate_fn(self, batch):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        waveforms = []

        for spectrogram, label, input_length, label_length, waveform in batch:
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(label)
            input_lengths.append(input_length.item())
            label_lengths.append(label_length.item())
            waveforms.append(waveform)

        padded_spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return (
            padded_spectrograms,
            padded_labels,
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(label_lengths, dtype=torch.long),
            waveforms
        )
