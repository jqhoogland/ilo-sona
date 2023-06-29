import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from ilosona.data import TokiPonaDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import TrainConfig, Trainer
from mingpt.utils import CfgNode, set_seed


def train(seed=0, corpus_dir="../corpus-test-cleaned"):
    set_seed(seed)

    dataset = TokiPonaDataset(corpus_dir, max_length=1024)

    model_config = GPTConfig(
        model_type="gpt",
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=128 // 8,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT(model_config)

    print(f"number of parameters: {model.num_params}")

    train_config = TrainConfig(
        num_epochs=250,
        logging_ivl=100,
        device="auto",
        # dataloder parameters
        num_workers=4,
        # optimizer parameters
        max_iters=None,
        batch_size=512,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,  # only applied on matmul weights
        grad_norm_clip=1.0,
    )

    trainer = Trainer(train_config, model, dataset)

    def log_to_wandb(trainer: Trainer):
        if trainer.iter_num % trainer.config["logging_ivl"] == 0:
            loss = trainer.loss.item()
            print(
                f"Epoch: {trainer.epoch_num}, Batch: {trainer.iter_num}, Loss: {loss}"
            )
            wandb.log({"loss": loss})

    def save_model(trainer: Trainer):
        torch.save(
            {
                "epoch": trainer.epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                # "scheduler_state_dict": scheduler.state_dict(),
                "loss": trainer.loss,
            },
            f"./checkpoints/checkpoint_{trainer.epoch_num}.pt",
        )

    wandb.init(
        project="toki-pona", config=asdict(trainer.config) | asdict(model_config)
    )
    wandb.watch(model, log="all")

    trainer.add_callback("on_batch_end", log_to_wandb)
    trainer.add_callback("on_epoch_end", save_model)

    trainer.run()

    wandb.finish()


if __name__ == "__main__":
    train()
