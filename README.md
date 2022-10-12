### A template for creating a deep learning project.

Your really detailed description goes here.

### Using this template
1. Create a new environment with ```pipenv shell```
2. Install flit with  ```pip install flit```
3. Install the template via flit: ```flit install -s```
4. Train a mnist model: ```train```
5. Evaluate the trained model: ```evaluate```

Simple, huh?
Well, kind of.

The guts of the project are in `src/__init__.py`, `src/train_config.py` and `src/eval_config.py`.
`src/__init__.py` dynamically loads the classes required for training and evaluation (a model class, defined in `src/models/`,
a dataset, in `src/datasets/`, and an evaluator class `src/evaluators/`). It then smushes the models/datasets/evaluators
together to either train or evaluate a model. Pytorch Lightning takes care of all of the training
specifics through the `trainer_args` class in `src/train_config.py`.

All arguments to the model/dataset/evaluator classes are defined in the config files: `src/train_config.py`, and `src/eval_config.py`.

This structure tries to disentangle model, dataset, and evaluation code. Good rules of thumb: Don't import anything from
src.models, src.datasets, or src.evaluators into other scripts. Rely on util/util.py to store helper functions and other
operations that you use frequently.

It's ok to hardcode variables for certain experiments since you can always copy/paste code when you want
to start a new experiment.

Another cool feature: This project relies on `sacred (https://github.com/IDSIA/sacred)` to manage its CLI.
This means that you can change arguments for training or evaluating via the command line. For example, if I wanted
to train the MNISTModel for 10 epochs, I'd run:
```shell
train with trainer_args.max_epochs=10
```
You can go on and configure your whole experiment via the command line if you want - training hyperparameters will
be saved in the model log directory (configured in `train_config.py`).

### Adding new models/datasets/evaluators

Easy! Just create a new file in src/models: `touch src/models/my_cool_idea.py` and create a class that inherits from
`pl.LightningModule` (just as the `MNISTModel` does in src/models/mnist_classifier.py). Be sure to fill out the same methods
as the MNISTModel.
Creating a new dataset/evaluator is the exact same, just replace `models/` with `datasets`/`evaluators`.
