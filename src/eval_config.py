import logging
import pdb
from copy import deepcopy
from glob import glob

from sacred import Experiment
from src.util import to_dict

logger = logging.getLogger("evaluate")

evaluation_ex = Experiment()


@evaluation_ex.config
def config():

    device = "cpu"
    model_name = "MNISTModel"
    evaluator_name = "MNISTAccuracy"
    checkpoint_path = "model_data/MNISTModel/3/checkpoints/best_loss_model.ckpt"
    log_verbosity = logging.INFO

    @to_dict
    class evaluator_args:
        device = "cpu"
        batch_size = 32
        mnist_data_path = "mnist_data/"





    
