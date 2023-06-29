# In this file, we will clean toki pona corpus. 
import warnings
from tokinizer import Tokinizer
import os

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
        
        return cleaned_text


    def clean_directory(self, input_dir, output_dir):
       # Ensure the output directory exists.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over all directories and files in the input directory.
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                input_file_path = os.path.join(root, file)

                # Create a corresponding path in the output directory.
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))

                # Ensure the output directory for the file exists.
                output_file_dir = os.path.dirname(output_file_path)
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)

                # Clean the file.
                self.clean_text(input_file_path, output_file_path)

            for dir in dirs:
                input_dir_path = os.path.join(root, dir)

                # Create a corresponding path in the output directory.
                output_dir_path = os.path.join(output_dir, os.path.relpath(input_dir_path, input_dir))

                # Ensure the output directory for the directory exists.
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path) 


if __name__ == '__main__':
    tokinizer = Tokinizer()
    cleaner = TokiPondaCleaner(tokinizer)
    cleaned_text = cleaner.clean_text("./corpus-test/jan_mika.txt", "./corpus-test-basic-cleaned/jan_mika.txt")
    tokens = tokinizer.split("./corpus-test/cleaned/jan_mika.txt")
    combined = tokinizer.combine(tokens)

    cleaner.clean_directory("./corpus-test", "./corpus-test-basic-cleaned")
    cleaner.clean_directory("./corpus", "./corpus-basic-cleaned")

