#python3.10 

"""
    Logger class for training process
"""

import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:

    SUM_FREQ = 100

    def __init__(self, args,
                 model=None, scheduler=None):
        # get the arguments
        self.args = args
        # get the model and scheduler
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}

        # get the summary writer
        dir_name = os.path.join(self.args.ckpt_path,
                                 f"runs_{self.args.exp_name}")
        self.writer = SummaryWriter(log_dir=dir_name)


    def _print_training_status(self):

        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(
        f"Training Metrics ({self.total_steps}): {training_str + metrics_str}"
        )

        if self.writer is None:
            dir_name = os.path.join(
                self.args.ckpt_path,
                f"runs_{self.args.exp_name}"
                )
            self.writer = SummaryWriter(log_dir=dir_name)

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            dir_name = os.path.join(
                self.args.ckpt_path,
                f"runs_{self.args.exp_name}"
                )
            self.writer = SummaryWriter(log_dir=dir_name)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()