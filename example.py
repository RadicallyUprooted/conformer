import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample, TimeMasking, FrequencyMasking

import matplotlib.pyplot as plt

from model.conformer import Conformer


if __name__ == '__main__':

    wav, sr = torchaudio.load("example.wav")

    mel_spec = MelSpectrogram(
        win_length=int(16000 * 0.025),
        hop_length=int(16000 * 0.01),
        n_mels=80,
        n_fft=2048
    )(wav)
    logmel = mel_spec.log2()
    def plot_mel_spectrogram(wave, title=None, y_label="freq_bin", aspect="auto", x_max=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel(y_label)
        axs.set_xlabel("frame")
        im = axs.imshow(wave.transpose(0, 1), origin="lower", aspect=aspect)
        if x_max:
            axs.set_xlim((0, x_max))
        fig.colorbar(im, ax=axs)
        plt.show(block=False)
        plt.waitforbuttonpress(0) 
        plt.close(fig)

    print(logmel.size())



