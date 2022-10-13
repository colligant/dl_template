from __future__ import annotations

import logging

import torch

from src.datasets.mnist_dataset import MNISTDataset
from src.evaluators import Evaluator
from src.util import torch_compute_matches

logger = logging.getLogger("evaluate")


class MNISTAccuracy(Evaluator):
    def __init__(self, batch_size, mnist_data_path, device):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        # set up validation dataset
        self.dataset = MNISTDataset(training=False, mnist_data_path=mnist_data_path)

    def evaluate(self, model_class):
        """Evaluation logic can be difficult, so this function
        is pretty much required to implement __all__ of it.
        Model_class is passed in with its weights already loaded."""

        model_class = model_class.to(self.device)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn(),
        )

        logger.info("Starting evaluation loop.")

        correct = 0
        total = 0
        for features, labels in dataloader:
            preds = model_class(features.to(self.device)).argmax(dim=-1)
            # notice i've imported this function from util.py because it'll
            # be useful in many more cases than just this file.
            correct += torch_compute_matches(preds, labels)
            total += preds.shape[0]

        logger.info(f"Overall accuracy: {correct/total:.3f}")
