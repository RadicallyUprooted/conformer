import os
import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from text_processor.processor import CharTextTransform

from typing import Union
from pathlib import Path

class LibriSpeechDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for the LibriSpeech dataset.
    """
    def __init__(self,
                path: Union[str, Path],
                train_url: str,
                val_url: str,
                batch_size: int,
                n_mels: int,
                vocab_size: int,
                time_mask_param: int,
                freq_mask_param: int,
                num_workers: int):
        super().__init__()
        self.path = path
        self.train_url = train_url
        self.val_url = val_url
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_workers = num_workers
        self.text_transform = CharTextTransform()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)
        self.spectrogram_augment = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        )

    def prepare_data(self):
        os.makedirs(self.path, exist_ok=True)
        torchaudio.datasets.LIBRISPEECH(self.path, url=self.train_url, download=True)
        torchaudio.datasets.LIBRISPEECH(self.path, url=self.val_url, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchaudio.datasets.LIBRISPEECH(self.path, url=self.train_url)
        self.val_dataset = torchaudio.datasets.LIBRISPEECH(self.path, url=self.val_url)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_train,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_val,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def collate_fn_val(self, batch):
        """
        A custom collate function to process the audio and text data for validation.
        """
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        waveforms = []

        for waveform, _, utterance, _, _, _ in batch:
            spec = self.mel_transform(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            
            label = torch.Tensor(self.text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            
            input_lengths.append(spec.shape[0])
            label_lengths.append(len(label))
            waveforms.append(waveform.squeeze(0))

        padded_spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return (
            padded_spectrograms,
            padded_labels,
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(label_lengths, dtype=torch.long),
            waveforms
        )

    def collate_fn_train(self, batch):
        """
        A custom collate function to process the audio and text data.
        """
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        waveforms = []

        for waveform, _, utterance, _, _, _ in batch:
            spec = self.mel_transform(waveform).squeeze(0).transpose(0, 1)
            
            spec = self.spectrogram_augment(spec)

            spectrograms.append(spec)
            
            label = torch.Tensor(self.text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            
            input_lengths.append(spec.shape[0])
            label_lengths.append(len(label))
            waveforms.append(waveform.squeeze(0))

        padded_spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return (
            padded_spectrograms,
            padded_labels,
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(label_lengths, dtype=torch.long),
            waveforms
        )
