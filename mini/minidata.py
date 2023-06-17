"""
Compile a dataset of Toki Pona.

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
import torch as t
from torch.utils.data import Dataset
from minitokinizer import Tokinizer


class TokiPonaDataset(Dataset):
    def __init__(self, corpus_dir, max_length=1024, samples_path=None):
        self.corpus_dir = corpus_dir
        self.max_length = max_length
        self.tokenizer = Tokinizer()
        if samples_path is None:
            self.samples = self.get_samples()
        else:
            self.samples = t.load(samples_path)

    def get_samples(self):
        samples = []
        for root, dirs, files in os.walk(self.corpus_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    text = f.read()
                tokens = self.tokenizer.encode(text).squeeze()
                for i in range(0, len(tokens), self.max_length):
                    samples.append(tokens[i:i+self.max_length])
        return t.Tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_length = len(sample)
        if sample_length < self.max_length:
            padding_length = self.max_length - sample_length
            sample = t.cat([sample, t.zeros(padding_length, dtype=t.long)], dim=0)
            mask = t.cat([t.ones(sample_length, dtype=t.long), t.zeros(padding_length, dtype=t.long)], dim=0)
        else:
            mask = t.ones(self.max_length, dtype=t.long)
        return sample, mask
