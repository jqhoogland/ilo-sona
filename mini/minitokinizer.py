import yaml
import warnings
import torch as t
import re


def get_vocab(path):
    with open(path, 'r') as f: 
        data = yaml.safe_load(f)

    words = []
    for entry in data:
        word = list(entry.keys())[0].split(" ")[0]
        for x in word.split("/"):
            words.append(x)

    return list(set(words))

'''
All tokens: words from dictionary, each on its own and with a space prepended +
            letters of the alphabet in lower case and upper case +
            standard puncuation marks +
            digits +
            space
'''
# Get words from dictonary file (living in ilosona dir).
VOCABULARY = get_vocab("../ilosona/dictionary.yaml")
VOCABULARY += [f" {w}" for w in VOCABULARY]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")+list("abcdefghijklmnopqrstuvwxyz".upper())
PUNCTUATION = list(".,?!()[]{}:;\"'\\/-+=~<>`@#$%^&*_\n")
DIGITS = list("0123456789")

class Tokinizer:
    def __init__(self, vocabulary=VOCABULARY):
        self.vocabulary = vocabulary
        self.tokens = set(vocabulary + LETTERS + PUNCTUATION + DIGITS + [" "])
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.vocabulary)

        # Difficult to find a comprehensive dictionary.
        # So we will store a list of new words and warn the user to deal with this shit.
        self.need_retoki = False
        self.new_tokens = []

    def retoki(self):
        '''
        This needs to be called if we find new tokens in some text.
        It will add them to our token_to_id dictionary so that things don't break.
        The problem with this is that we might end up adding tokens we don't want!
        '''
        start = len(self.token_to_id)
        for i, token in enumerate(self.new_tokens):
            self.token_to_id[token] = start+i
            self.id_to_token[start+i] = token
        self.need_retoki = False

    def encode(self, text, truncation=False, return_tensors=True):
        warnings.warn("Truncation is not supported.", UserWarning)

        tokinized = self.tokinize(text)
        # If we need to add new tokens, now is the time.
        if self.need_retoki: 
            self.retoki()

        encoded_tks = [self.token_to_id[tk] for tk in tokinized]
        if return_tensors:
            return t.tensor([encoded_tks])
        else:
            return encoded_tks
        
    def decode(self, tks):
        if isinstance(tks, t.Tensor):
            return "".join([self.id_to_token[i.item()] for i in tks.flatten()])
        else:
            return "".join([self.id_to_token[i] for i in tks])

    def tokinize(self, text):
        tokens = []
        space = False
        # I learned regex for the billionth time to write this.
        # If you ask me how it works I have probably forgotten again.
        for token in re.findall(r"\w+|[^\w\s]|[\s]", text):
            # Check if proper noun (we need to split into chars later).
            proper_noun = True if token[0].isupper() else False

            # Deal with spaces.
            if token.isalpha() and space: 
                tokens[-1] = " " + token
            else:
                tokens.append(token)
            
            # Split if we just had a proper noun.
            if proper_noun:
                tokens = tokens[:-1] + list(tokens[-1])
            
            # Record if we just looked at a space (to prepend to next token if necessary).
            space = True if token == " " else False

            # Finally check if we have seen this token before.
            # If we haven't, we yell at the user.
            if not(token in self.tokens or proper_noun or token in self.new_tokens):
                warnings.warn(f"{token} wasn't found in our dictionary! retoki will be run before encoding.", UserWarning)
                self.new_tokens.append(token)
                self.new_tokens.append(" " + token)
                self.need_retoki = True
        return tokens

if __name__ == "__main__":
    with open("../corpus-test/jan lawa Oliki Soweli Elepanto.txt", "r") as f:
        text = f.read()

    tokinizer = Tokinizer()
    tokens = tokinizer.tokinize(text)

    print(text)
    print("-"*30)
    print(tokens)
    print("-"*30)
    tokitext = tokinizer.decode(tokinizer.encode(text))
    print(tokitext)
    print("-"*30)
    print(f"Test text equals tokinized text: {text == tokitext}")