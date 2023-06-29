"""
Code to compile a large dataset of toki pona text.

Sources:
- https://www.fincxjejo.com/toki-pona-stories
- https://lipukule.org/
- https://tokipona.org/sitata/ (The Bible)
- https://wikipesija.org/wiki/lipu_open
- https://janketami.wordpress.com/lipu-pini-mi/
- https://tokipona.org/toki_lili_27_poems_in_toki_pona.pdf
- https://lipumonsuta.neocities.org/

"""
import os

import torch
from torch.utils.data import Dataset
from torchtyping import TensorType

from ilosona.tokinizer import Tokinizer


class TokiPonaDataset(Dataset):
    def __init__(self, corpus_dir, max_length=1024):
        self.corpus_dir = corpus_dir
        self.max_length = max_length
        self.tokenizer = Tokinizer()
        self.samples = self.get_samples()

    def get_samples(self):
        samples = []
        for root, dirs, files in os.walk(self.corpus_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    text = f.read()

                tokens = self.tokenizer.encode(text).squeeze()
                for i in range(0, len(tokens), self.max_length):
                    samples.append(tokens[i : i + self.max_length])

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample: TensorType["sample_len"] = self.samples[idx]
        sample_length = len(sample)
        if sample_length < self.max_length:
            padding_length = self.max_length - sample_length
            sample = torch.cat(
                [
                    sample,
                    torch.zeros(
                        (padding_length), dtype=torch.long
                    ),
                ],
                dim=0,
            )
        #     mask = torch.cat(
        #         [
        #             torch.ones(sample_length, dtype=torch.long),
        #             torch.zeros(padding_length, dtype=torch.long),
        #         ],
        #         dim=0,
        #     )
        # else:
        #     mask = torch.ones(self.max_length, dtype=torch.long)

        return sample


if __name__ == "__main__":
    dataset = TokiPonaDataset("./corpus")
    print(len(dataset))
