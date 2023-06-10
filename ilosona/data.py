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
import torch
from torch.utils.data import Dataset
from ilosona.tokinizer import Tokinizer
import os

class TokiPonaDataset(Dataset):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.tokenizer = Tokinizer()
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for root, dirs, files in os.walk(self.corpus_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with open(file_path, 'r') as f:
            text = f.read()
        tokens = self.tokenizer.encode_plus(
            text, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()
