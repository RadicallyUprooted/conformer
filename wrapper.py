import torch
import torchmetrics
from torch import nn

import pytorch_lightning as pl

from model.conformer import Conformer
from text_processor import TextProcessor

class ConformerModel(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        input_dim: int,
        encoder_dim: int,
        attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout_p: float,
        lr: float,
        text_processor: TextProcessor,
    ):
        super().__init__()

        self.conformer = Conformer(
            num_classes,
            num_layers,
            input_dim,
            encoder_dim,
            attention_heads,
            depthwise_conv_kernel_size,
            dropout_p,
        )
        self.lr = lr
        self.text_processor = text_processor
        self.wer = torchmetrics.WordErrorRate()
        self.criterion = nn.CTCLoss()
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):

        inputs, input_lengths, targets, target_lengths = batch
        outputs, outputs_lengths = self.conformer(inputs, input_lengths)

        loss = self.criterion(outputs.transpose(0, 1), targets, outputs_lengths, target_lengths) 

        self.log("train_loss", loss)

        return loss
    
    def test_step(self, batch, batch_idx):

        inputs, input_lengths, targets, target_lengths = batch

        outputs, outputs_lengths = self.conformer(inputs, input_lengths)

        loss = self.criterion(outputs.transpose(0, 1), targets, outputs_lengths, target_lengths) 
        
        decode = outputs.argmax(dim=-1)

        predicts = [self.text_process.decode(sent) for sent in decode]
        targets = [self.text_process.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.wer(i, j).item() for i, j in zip(targets, predicts)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("test_loss", loss)
        self.log("test_wer", wer)

        return loss
    
    def validation_step(self, batch, batch_idx):

        inputs, input_lengths, targets, target_lengths = batch

        outputs, outputs_lengths = self.conformer(inputs, input_lengths)

        loss = self.criterion(outputs.transpose(0, 1), targets, outputs_lengths, target_lengths) 
        
        decode = outputs.argmax(dim=-1)

        predicts = [self.text_process.decode(sent) for sent in decode]
        targets = [self.text_process.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.wer(i, j).item() for i, j in zip(targets, predicts)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("val_loss", loss)
        self.log("val_wer", wer)

        return loss





