import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import wandb
from ilosona.data import TokiPonaDataset
from ilosona.model import Decoder


def train(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize Weights & Biases run
    wandb.init(project="toki-pona", config=config)
    wandb.watch(model, log="all")

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    for epoch in range(config["epochs"]):
        for i, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % config["log_interval"] == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})

        # Save a checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f'checkpoint_{epoch}.pt')

    wandb.finish()


# Define the configuration for the training
config = {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "gamma": 0.95,
    "step_size": 1,
    "log_interval": 10,
}
MAX_LENGTH = 1024

# Load the dataset
dataset = TokiPonaDataset("../corpus/Independent", max_length=MAX_LENGTH)

print(dataset.samples)

# Load the model (make sure to adjust the parameters according to your needs)
model = Decoder(vocab_size=dataset.tokenizer.vocab_size, max_length=MAX_LENGTH)


# Train the model
# train(model, dataset, config)
