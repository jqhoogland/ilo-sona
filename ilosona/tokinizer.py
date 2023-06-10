"""
The toki pona tokenizer.
"""
import warnings

import torch
import torchtext
from torchtext.data.utils import get_tokenizer


def unpack_lists(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(unpack_lists(item))
        else:
            flattened_list.append(item)
    return flattened_list


DEFAULT_VOCABULARY = [
    "a",
    "akesi",
    "ala",
    "alasa",
    "ale",
    "ali",
    "anpa",
    "ante",
    "anu",
    "awen",
    "e",
    "en",
    "esun",
    "ijo",
    "ike",
    "ilo",
    "insa",
    "jaki",
    "jan",
    "jelo",
    "jo",
    "kala",
    "kalama",
    "kama",
    "kasi",
    "ken",
    "kepeken",
    "kili",
    "kin",
    "kiwen",
    "ko",
    "kon",
    "kule",
    "kulupu",
    "kute",
    "la",
    "lape",
    "laso",
    "lawa",
    "len",
    "lete",
    "li",
    "lili",
    "linja",
    "lipu",
    "loje",
    "lon",
    "luka",
    "lukin",
    "lupa",
    "ma",
    "mama",
    "mani",
    "meli",
    "mi",
    "moku",
    "moli",
    "monsi",
    "mu",
    "mun",
    "musi",
    "mute",
    "nanpa",
    "nasa",
    "nasin",
    "nena",
    "ni",
    "nimi",
    "noka",
    "o",
    "oko",
    "olin",
    "ona",
    "open",
    "pakala",
    "pali",
    "palisa",
    "pan",
    "pana",
    "pi",
    "pilin",
    "pimeja",
    "pini",
    "pipi",
    "poka",
    "poki",
    "pona",
    "pu",
    "sama",
    "seli",
    "selo",
    "seme",
    "sewi",
    "sijelo",
    "sike",
    "sin",
    "sina",
    "sinpin",
    "sitelen",
    "sona",
    "soweli",
    "suli",
    "suno",
    "supa",
    "suwi",
    "tan",
    "taso",
    "tawa",
    "telo",
    "tenpo",
    "toki",
    "tomo",
    "tu",
    "unpa",
    "uta",
    "utala",
    "walo",
    "wan",
    "waso",
    "wawa",
    "weka",
    "wile",
]

LETTERS = list("abcdefghijklmnopqrstuvwxyz")
DEFAULT_PUNCTUATION = [".", ",", "?", ")", "(", "!", "[", "]", ":", ";", '"', "'"]
VOCABULARY = DEFAULT_VOCABULARY + DEFAULT_PUNCTUATION + LETTERS


class Tokinizer:
    def __init__(self, punctuation=DEFAULT_PUNCTUATION, vocabulary=DEFAULT_VOCABULARY):
        self.vocabulary = vocabulary
        self.punctuation = punctuation
        self.token_to_id = {token: i for i, token in enumerate(vocabulary)}
        self.vocab_size = len(self.vocabulary)

    def unpack_lists(self, nested_list):
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self.unpack_lists(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def toki_pona_tokenizer(self, text):
        text = text.lower()

        for char in self.punctuation:
            text = text.replace(char, " " + char + " ")
        
        tokens = text.split()

        for token in tokens:
            if token not in self.vocabulary:
                warnings.warn(f"{token} not in vocab", UserWarning)
                
        return self.unpack_lists(
            [token if token in self.vocabulary else list(token) for token in tokens]
        )
    
    def tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def encode_plus(self, text, truncation=False, return_tensors='pt'):
        warnings.warn("Truncation is not supported", UserWarning)

        tokens = self.toki_pona_tokenizer(text)
        input_ids = self.tokens_to_ids(tokens)

        if return_tensors == 'pt':
            input_ids = torch.tensor([input_ids])

        return {'input_ids': input_ids}
