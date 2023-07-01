import torch

from mini.minidata import TokiPonaDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import TrainConfig, Trainer
from mingpt.utils import set_seed


def train(corpus_dir, seed=0):
    set_seed(seed)

    max_length = 64
    dataset = TokiPonaDataset(corpus_dir, max_length=max_length)

    model_config = GPTConfig(
        model_type="gpt",
        n_layer=4,
        n_head=4,
        n_embd=128,
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=max_length
    )
    model = GPT(model_config)

    train_config = TrainConfig(
        num_epochs=10000,
        logging_ivl=100
    )
    trainer = Trainer(train_config, model, dataset)

    def save_model(trainer: Trainer):
        if trainer.epoch_num % train_config.logging_ivl != 0: return
        torch.save(
            {
                "epoch": trainer.epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss": trainer.loss,
            },
            f"snapshots/{trainer.epoch_num}.pt",
        )

    trainer.add_callback("on_epoch_end", save_model)
    trainer.run()


if __name__ == "__main__":
    train("../corpus")
