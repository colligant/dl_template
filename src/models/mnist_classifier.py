from __future__ import annotations

import pdb

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class MNISTModel(pl.LightningModule):
    def __init__(self, hidden_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(28 * 28, hidden_dimension)
        self.l2 = torch.nn.Linear(hidden_dimension, 10)
        self.hidden_dimension = hidden_dimension

        # record all of your hyperparameters here by using self.hyperparameter = hyperparameter
        self.save_hyperparameters()

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.view(x.size(0), -1))))

    def _shared_step(self, batch):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x["loss"] for x in outputs])
        loss = torch.mean(torch.stack(train_loss))
        self.log("train_loss", loss)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
