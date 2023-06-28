# In this file, we will clean toki pona corpus. 
import warnings
from tokinizer import Tokinizer

class TokiPondaCleaner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def clean_text(self, input_file, output_file):
        with open(input_file, 'r') as f:
            text = f.read()

        tokens = self.tokenizer.split(text)
        cleaned_text = self.tokenizer.combine(tokens)

        with open(output_file, 'w') as f:
            f.write(cleaned_text)




if __name__ == '__main__':
    tokinizer = Tokinizer()
    cleaner = TokiPondaCleaner(tokinizer)
    cleaner.clean_text("./corpus-test/jan_mika.txt", "./corpus-test/cleaned/jan_mika.txt")
