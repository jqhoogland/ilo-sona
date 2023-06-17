import pickle
import warnings
import torch as t
import re


def get_vocab(path):
    infile = open(path, "rb")
    words = pickle.load(infile); infile.close()
    return words

'''
All tokens: words from dictionary, each on its own and with a space prepended +
            letters of the alphabet in lower case and upper case +
            standard puncuation marks +
            digits +
            space
'''
# Get words from dictonary file (living in ilosona dir).
VOCABULARY = get_vocab("dictionary.pickle")
VOCABULARY += [f" {w}" for w in VOCABULARY]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")+list("abcdefghijklmnopqrstuvwxyz".upper())
PUNCTUATION = list(".,?!()[]{}:;\"'\\/-+=~<>`@#$%^&*_\n")
DIGITS = list("0123456789")

class Tokinizer:
    def __init__(self, vocabulary=VOCABULARY):
        self.vocabulary = vocabulary
        self.tokens = vocabulary + LETTERS + PUNCTUATION + DIGITS + [" "]
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.token_to_id["<undefined>"] = -1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.vocabulary)

    def encode(self, text, truncation=False, return_tensors=True):
        warnings.warn("Truncation is not supported.", UserWarning)

        tokinized = self.tokinize(text)
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
        # Just trust me bro.
        for token in re.findall(r"\w+|[^\w\s]|[\s]", text):
            # Check if proper noun (we need to split into chars later).
            proper_noun = True if token[0].isupper() else False

            # Deal with spaces.
            if token.isalpha() and space: 
                tokens[-1] = " " + token
            else:
                tokens.append(token)
            
            if proper_noun or token.isnumeric():
                # Split if we just had a proper noun, or a number.
                # But check that there aren't any funky characters in there.
                split = list(tokens[-1])
                for c in split:
                    if c not in self.tokens:
                        # warnings.warn(f"'{c}' wasn't found in my dictionary!", UserWarning)
                        tokens[-1] = "<undefined>"
                        break
                else:
                    tokens = tokens[:-1] + list(tokens[-1])
            else:
                # Yell at the user if we haven't seen this token before.
                if tokens[-1] not in self.tokens:
                    # warnings.warn(f"'{token}' wasn't found in my dictionary!", UserWarning)
                    tokens[-1] = "<undefined>"                

            # Record if we just looked at a space (to prepend to next token if necessary).
            space = True if token == " " else False

        # Collapse continuous undefined tokens into one single token.
        result = []
        for token in tokens:
            if token == "<undefined>" and (not result or result[-1] != "<undefined>"):
                result.append(token)
            elif token != "<undefined>":
                result.append(token)
        return result

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