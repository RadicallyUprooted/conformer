import torch
import torchaudio
from torch.utils.data import Dataset
from text_processor.processor import CharTextTransform
from pathlib import Path
from typing import Union


class LibriSpeechCustom(Dataset):
    def __init__(self,
                n_mels: int,
                freq_mask_param: int,
                time_mask_param: int,
                split: str,
                root: Union[str, Path]):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_mels = n_mels
        self.text_processor = CharTextTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, transcript, _, _, _ = self.dataset[idx]

        tokens = self.text_processor.text_to_int(transcript.lower().replace(" ", "|"))
        token_lengths = torch.tensor([len(tokens)]).int()

        spectrogram = self.get_spectrogram(waveform)
        input_lengths = torch.tensor([spectrogram.shape[-1]]).int()

        return spectrogram, torch.tensor(tokens).int(), input_lengths, token_lengths, waveform

    def get_spectrogram(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)(waveform)
        if self.freq_mask_param is not None:
            spectrogram = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self.freq_mask_param
            )(spectrogram)
        if self.time_mask_param is not None:
            spectrogram = torchaudio.transforms.TimeMasking(
                time_mask_param=self.time_mask_param
            )(spectrogram)
        return spectrogram


class CommonVoiceCustom(Dataset):
    def __init__(self, 
                n_mels: int,
                freq_mask_param: int,
                time_mask_param: int,
                root: Union[str, Path]):
        self.dataset = torchaudio.datasets.COMMONVOICE(root=root, tsv="validated.tsv")
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_mels = n_mels
        self.text_processor = CharTextTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, data = self.dataset[idx]
        sentence = data["sentence"]

        tokens = self.text_processor.text_to_int(sentence.lower().replace(" ", "|"))
        token_lengths = torch.tensor([len(tokens)]).int()

        spectrogram = self.get_spectrogram(waveform)
        input_lengths = torch.tensor([spectrogram.shape[-1]]).int()

        return spectrogram, torch.tensor(tokens).int(), input_lengths, token_lengths, waveform

    def get_spectrogram(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)(waveform)
        if self.freq_mask_param is not None:
            spectrogram = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self.freq_mask_param
            )(spectrogram)
        if self.time_mask_param is not None:
            spectrogram = torchaudio.transforms.TimeMasking(
                time_mask_param=self.time_mask_param
            )(spectrogram)
        return spectrogram
