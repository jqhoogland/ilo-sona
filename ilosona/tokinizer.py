"""
The toki pona tokenizer.
"""
import warnings

import numpy as np
import torch

# import torch


def unpack_lists(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(unpack_lists(item))
        else:
            flattened_list.append(item)
    return flattened_list


WORDS = [
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


LETTERS = list("abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper())
DEFAULT_PUNCTUATION = list(".,?!()[]{}:;\"'\\/-+=~<>`@#$%^&*_\n")
NUMBERS = list("0123456789")
EXTRA = DEFAULT_PUNCTUATION + NUMBERS
VOCABULARY = WORDS + EXTRA


class Tokinizer:
    def __init__(self, words=WORDS, extra=EXTRA):
        self.words = words
        self.extra = extra
        self.tokens = words + extra
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        self.final_punctuation = "'.,;:?!)-_\\n"
        self.default_punctuation = DEFAULT_PUNCTUATION
        self.numbers = NUMBERS

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
            elif curr_word in self.words:
                tokens.append(curr_word)
                curr_word = ""
            else:  # curr_word not in vocabulary
                warnings.warn(f"Removing unknown word: {curr_word}", UserWarning)
                curr_word = ""
            if c in self.tokens:  # or c == " ":
                tokens.append(c)

        return tokens

    def tokens_to_ids(self, tokens):
        return np.array(
            [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        )

    def encode_ids(self, text, truncation=False, return_tensors="pt"):
        warnings.warn("Truncation is not supported", UserWarning)

        tokens = self.split(text)
        input_ids = self.tokens_to_ids(tokens)

        if return_tensors == "pt":
            input_ids = torch.tensor([input_ids])

        return input_ids

    def encode(self, text, truncation=False, return_tensors="pt"):
        """Get a one-hot encoded array"""
        ids = self.encode_ids(text, truncation=truncation, return_tensors="np")
        ids_1hot = np.zeros((ids.size, self.vocab_size + 1))
        ids_1hot[np.arange(ids.size), ids] = 1

        if return_tensors == "pt":
            ids_1hot = torch.tensor(ids_1hot)

        return ids_1hot

    def combine(self, tokens):
        text = ""

        for token in tokens:
            if token in self.words:
                token += " "
                text += token
            elif token in self.final_punctuation:
                if text and text[-1] == " ":
                    text = text[:-1]
                text += token

        return text


if __name__ == "__main__":
    with open("./corpus-test/jan_mika.txt", "r") as f:
        text = f.read()

    tokinizer = Tokinizer()
    tokens = tokinizer.split(text)

    print(text)
    print("---")
    print(tokinizer.encode(text))

    print("---")

    # print(tokinizer.combine(tokens))

    print(len(VOCABULARY))
