"""
The toki pona tokenizer.
"""

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
    'a', 'akesi', 'ala', 'alasa', 'ale', 'ali', 'anpa', 'ante', 'anu', 'awen',
    'e', 'en', 'esun', 'ijo', 'ike', 'ilo', 'insa', 'jaki', 'jan', 'jelo', 'jo',
    'kala', 'kalama', 'kama', 'kasi', 'ken', 'kepeken', 'kili', 'kin', 'kiwen',
    'ko', 'kon', 'kule', 'kulupu', 'kute', 'la', 'lape', 'laso', 'lawa', 'len',
    'lete', 'li', 'lili', 'linja', 'lipu', 'loje', 'lon', 'luka', 'lukin', 'lupa',
    'ma', 'mama', 'mani', 'meli', 'mi', 'moku', 'moli', 'monsi', 'mu', 'mun',
    'musi', 'mute', 'nanpa', 'nasa', 'nasin', 'nena', 'ni', 'nimi', 'noka', 'o',
    'oko', 'olin', 'ona', 'open', 'pakala', 'pali', 'palisa', 'pan', 'pana',
    'pi', 'pilin', 'pimeja', 'pini', 'pipi', 'poka', 'poki', 'pona', 'pu',
    'sama', 'seli', 'selo', 'seme', 'sewi', 'sijelo', 'sike', 'sin', 'sina',
    'sinpin', 'sitelen', 'sona', 'soweli', 'suli', 'suno', 'supa', 'suwi', 'tan',
    'taso', 'tawa', 'telo', 'tenpo', 'toki', 'tomo', 'tu', 'unpa', 'uta', 'utala',
    'walo', 'wan', 'waso', 'wawa', 'weka', 'wile'
]

DEFAULT_PUNCTUATION = ['.', ',', '?', ')', '(', '!']
VOCABULARY = DEFAULT_VOCABULARY + DEFAULT_PUNCTUATION

class Tokinizer:
    def __init__(self, punctuation = DEFAULT_PUNCTUATION, vocabulary = DEFAULT_VOCABULARY):
        self.vocabulary = vocabulary
        self.punctuation = punctuation
        self.tokenizer = self.get_tokenizer(self.toki_pona_tokenizer)

    def unpack_lists(nested_list):
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(unpack_lists(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def toki_pona_tokenizer(text):
        text = text.lower()
        for char in DEFAULT_PUNCTUATION:
            text = text.replace(char, ' ' + char + ' ')
        tokens = text.split()
        for token in tokens:
            if token not in VOCABULARY:
                print(f"Error: {token} not in vocab")
        return unpack_lists([token if token in VOCABULARY else list(token) for token in tokens])