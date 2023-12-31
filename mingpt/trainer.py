"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import CfgNode as CN

from tqdm import trange

@dataclass
class TrainConfig:
    num_epochs: int = 250
    logging_ivl: int = 100
    device: str = "auto"
    # dataloder parameters
    num_workers: int = 4
    # optimizer parameters
    max_iters: int | None = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1  # only applied on matmul weights
    grad_norm_clip: float = 1.0


class Trainer:
    def __init__(self, config: TrainConfig, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks: dict[str, list[callable]] = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.epoch_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        # So I can sit and stare at the training run.
        for e in trange(config.num_epochs):
            for i in range(len(train_loader)):
            # fetch the next batch (x, y) and re-init iterator if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter) 
                batch = [t.to(self.device) for t in batch]
                batch = torch.stack(batch, dim=0)            
                
                # forward the model
                logits, self.loss = model(batch)
                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                self.trigger_callbacks("on_batch_end")
                self.iter_num += 1

                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                if config.max_iters is not None and self.iter_num >= config.max_iters: return

            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1
