import os
import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class CharTextTransform:
    """
    A transform to convert text to a sequence of integers and back.
    """
    def __init__(self):
        self.char_map = {
            "'": 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8,
            'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16,
            'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24,
            'x': 25, 'y': 26, 'z': 27
        }
        self.index_map = {v: k for k, v in self.char_map.items()}

    def text_to_int(self, text):
        return [self.char_map[c] for c in text.lower()]

    def int_to_text(self, labels):
        return "".join([self.index_map[i] for i in labels])

class LibriSpeechDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for the LibriSpeech dataset.
    """
    def __init__(self, path, train_url, val_url, batch_size, n_mels, vocab_size):
        super().__init__()
        self.path = path
        self.train_url = train_url
        self.val_url = val_url
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.text_transform = CharTextTransform()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)

    def prepare_data(self):
        os.makedirs(self.path, exist_ok=True)
        torchaudio.datasets.LIBRISPEECH(self.path, url=self.train_url, download=True)
        torchaudio.datasets.LIBRISPEECH(self.path, url=self.val_url, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchaudio.datasets.LIBRISPEECH(self.path, url=self.train_url)
        self.val_dataset = torchaudio.datasets.LIBRISPEECH(self.path, url=self.val_url)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def collate_fn(self, batch):
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
