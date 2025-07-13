# Conformer ASR

This repository contains a PyTorch Lightning implementation of the Conformer model by [(Gulati et al., 2020)](https://arxiv.org/abs/2005.08100) for Automatic Speech Recognition (ASR).

## Project Structure

```
├── checkpoints/      # Saved model checkpoints
├── configs/          # Hydra configuration files
│   ├── config.yaml   # Main configuration file
│   ├── data/         # Data-related configs
│   ├── decoder/      # Decoder configs
│   ├── model/        # Model architecture configs
│   └── optimizer/    # Optimizer configs
├── data/             # Datasets and dataloaders
├── model/            # Model implementation
├── text_processor/   # Text processing utilities
├── train.py          # Training script
├── inference.py      # Inference script
└── run.py            # Main entry point
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry.**

2.  **Install dependencies:**

    From the root of the project, run:

    ```bash
    poetry install
    ```
3. **Activate virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. The main configuration file is `configs/config.yaml`. This file composes configurations from other files in the `configs` directory.

You can override any configuration parameter from the command line. For example, to change the batch size:

```bash
python run.py train.batch_size=64
```

## How to Run

Use the `run.py` script as the main entry point for both training and inference.

### Training

To start a training run, simply execute:

```bash
python run.py
```

This will use the default configuration defined in `configs/config.yaml`. The `inference.audio_path` parameter in this file is set to `null` by default, which triggers the training script.

### Inference

To run inference on a single audio file or all audio files in a directory, you need to provide the path via the command line. This will trigger the `inference.py` script.

Make sure you have a trained model checkpoint specified in `configs/config.yaml` under `train.checkpoint`.

To process a single file:
```bash
python run.py inference.audio_path=/path/to/your/audio.wav
```

To process all audio files in a directory:
```bash
python run.py inference.audio_path=/path/to/your/audio_directory/
```

The script will load the model from the specified checkpoint, process the audio file(s), and print the predicted transcription(s).

### Example

```bash
python run.py \
    inference.audio_path=data/librispeech/LibriSpeech/test-clean/260/123286 \
    train.checkpoint=checkpoints/conformer49.ckpt
```

**Output:**

```text
File: 260-123286-0000.flac
Prediction:  saturday august fifteenth the sea unbroken all round no land in sight  

File: 260-123286-0001.flac
Prediction:  the horizon seems extremely distant  

File: 260-123286-0002.flac
Prediction:  all my danger and sufferings were needed to struck a spark of human feeling out of him but now that i am well his nature has resumed its sway

...
```
