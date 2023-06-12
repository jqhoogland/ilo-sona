"""
The toki pona tokenizer.
"""
import warnings

import numpy as np

# import torch


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

VOCABULARY = DEFAULT_VOCABULARY
LETTERS = list("abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper())
DEFAULT_PUNCTUATION = list(".,?!()[]{}:;\"'\\/-+=~<>`@#$%^&*_\n")
NUMBERS = list("0123456789")
EXTRA = DEFAULT_PUNCTUATION + LETTERS + NUMBERS


class Tokinizer:
    def __init__(self, vocabulary=DEFAULT_VOCABULARY, extra=EXTRA):
        self.vocabulary = vocabulary
        self.extra = extra
        self.tokens = vocabulary + extra
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.vocab_size = len(self.vocabulary)
        self.final_punctuation = ".,;:?!)-_"

    def unpack_lists(self, nested_list):
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self.unpack_lists(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def split(self, text):
        tokens = []

        curr_word = ""

        for c in text:
            if c in LETTERS:
                curr_word += c
                continue
            elif curr_word in self.vocabulary:
                tokens.append(curr_word)
                curr_word = ""
                continue
            else:
                tokens.extend(list(curr_word))
                curr_word = ""
            
            if c in self.tokens or c == " ":
                tokens.append(c)
            else: 
                warnings.warn(f"{c} not in vocab", UserWarning)
                                
        return tokens
    
    def tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def encode_ids(self, text, truncation=False, return_tensors='pt'):
        warnings.warn("Truncation is not supported", UserWarning)

        tokens = self.split(text)
        input_ids = self.tokens_to_ids(tokens)

        # if return_tensors == 'pt':
        #     input_ids = torch.tensor([input_ids])

        return input_ids

    def encode_ids(self, text, truncation=False, return_tensors='pt'):
        warnings.warn("Truncation is not supported", UserWarning)

        tokens = self.split(text)
        input_ids = self.tokens_to_ids(tokens)

        # if return_tensors == 'pt':
        #     input_ids = torch.tensor([input_ids])

        return input_ids

    def encode(self, text):
        """Get a one-hot encoded array"""
        ids = self.encode_ids(text)
        return np.onehot(ids)
    
    def combine(self, words):
        text = ""

        full_word = ""

        for word in words:
            if word in self.vocabulary and not full_word:
                word += " "
                text += word 
            elif word in self.final_punctuation and text[-1] == " ":
                text = text[:-1]
                text += word 
            else:
                full_word += word


        return text


if __name__ == "__main__":
    with open("./corpus-test/jan lawa Oliki Soweli Elepanto.txt", "r") as f:
        text = f.read()

    tokinizer = Tokinizer()
    tokens = tokinizer.split(text)

    print(text)
    print("---")
    print(tokens)

    print(tokinizer.combine(tokens))
