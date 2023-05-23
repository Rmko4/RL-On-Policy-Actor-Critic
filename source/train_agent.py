from os import name
import random
from pathlib import Path
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger # type: ignore

from argparser import get_args
from policy_gradient_module import PolicyGradientModule

PROJECT_NAME = "RL-Policy-Gradient"
LOGS_DIR = Path("logs/")


def train(hparams, config=None): # type: ignore
    logger = WandbLogger(name=f"{hparams.run_name}_{hparams.algorithm}",
                         project=PROJECT_NAME,
                         save_dir=LOGS_DIR,
                         log_model=True,
                         anonymous=True,)  # Simply allows for anonymous logging but doesn't force
    csv_logger = CSVLogger(save_dir=LOGS_DIR)

    hparams: dict = vars(hparams)
    hparams.pop('run_name')
    max_epochs = hparams.pop('max_epochs')
    gradient_clip_val = hparams.pop('max_grad_norm')

    callbacks = []

    hparams = {**hparams, **config} if config else hparams

    catch_module = PolicyGradientModule(**hparams)
    trainer = Trainer(max_epochs=max_epochs,
                      logger=[logger, csv_logger],
                      log_every_n_steps=10,
                      callbacks=callbacks,
                      gradient_clip_algorithm='norm',
                      gradient_clip_val=gradient_clip_val,
                      )
    trainer.fit(catch_module)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
