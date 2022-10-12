from __future__ import annotations

import logging

from sacred import Experiment

from src.util import to_dict

logger = logging.getLogger("train")
logger.setLevel(logging.WARNING)

train_ex = Experiment()


@train_ex.config
def config():

    description = (
        "This description of your experiment will be saved along with model "
        "hyperparameters in log_dir."
    )

    model_name = "MNISTModel"
    dataset_name = "MNISTDataset"
    log_dir = "model_data/"
    # instead of using print, use
    log_verbosity = logging.INFO

    # custom decorator to make declaring model/dataset arguments easier
    # requires name of the class to be model_args
    @to_dict
    class model_args:
        # notice python code execution
        hidden_dimension = 28 * 28

    @to_dict
    class train_dataset_args:
        training = True
        mnist_data_path = "mnist_data/"

    @to_dict
    class val_dataset_args:
        training = False
        mnist_data_path = "mnist_data/"

    # also can specify val dataset args

    @to_dict
    class dataloader_args:
        batch_size = 32
        num_workers = 0

    @to_dict
    class trainer_args:
        gpus = 0
        num_nodes = 0
        max_epochs = 3
        check_val_every_n_epoch = 1
