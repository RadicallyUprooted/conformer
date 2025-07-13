import torch
import hydra
import torchaudio
from omegaconf import DictConfig
from train import ConformerLightningModule
from pathlib import Path

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform

def get_spectrogram(waveform, n_mels):
    spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)(waveform)
    return spectrogram

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function for running inference.
    """
    if not cfg.inference.audio_path:
        print("Error: audio_path not provided in config for inference.")
        return

    model = ConformerLightningModule.load_from_checkpoint(cfg.train.checkpoint, cfg=cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio_path = Path(cfg.inference.audio_path)
    audio_files = []
    if audio_path.is_dir():
        for extension in ["*.wav", "*.flac", "*.mp3"]:
            audio_files.extend(audio_path.glob(extension))
    elif audio_path.is_file():
        audio_files.append(audio_path)

    if not audio_files:
        print(f"No audio files found at {audio_path}")
        return

    for audio_file in sorted(audio_files):
        waveform = load_audio(audio_file)
        spectrogram = get_spectrogram(waveform, cfg.data.n_mels)
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)
        
        spectrogram = spectrogram.unsqueeze(0).to(device)
        input_lengths = torch.tensor([spectrogram.shape[1]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs, _ = model(spectrogram, input_lengths)
        
        prediction = model.decoder(outputs[0].cpu())
        
        print(f"File: {audio_file.name}")
        print(f"Prediction: {prediction}\n")

if __name__ == '__main__':
    main()
