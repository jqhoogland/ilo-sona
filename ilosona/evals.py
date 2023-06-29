import random
import re
from itertools import permutations
from pprint import pp

import yaml


def gen_colors_tests():
    # Define the colors in Toki Pona
    colors = ["jelo", "loje", "laso", "walo", "pimeja"]

    # Generate all permutations of the colors
    all_permutations = list(permutations(colors))

    sentences = []

    # Loop over each permutation
    for i, permutation in enumerate(all_permutations):
        # Create the color phrases for this permutation
        color_phrases = [f"Ni li {color} ala." for color in permutation[:-1]]
        # Create the full sentence for this permutation
        sentences.append("Ni li kule. " + " ".join(color_phrases) + " Tan ni, ni li " + permutation[-1] + ".")
        # Print the sentence
        
    return sentences


def get_words_by_type(word_type, filepath):
    with open(filepath, 'r') as file:
        dictionary = yaml.safe_load(file)
    words = []
    for item in dictionary:
        for word, descriptions in item.items():
            # Extract the word before parentheses or colon
            word = re.split('[(|:]', word)[0].strip()
            if word_type in descriptions:
                # Ignore unconventional definitions
                if isinstance(descriptions[word_type], dict) and 'unconventional' in descriptions[word_type]:
                    continue
                words.append(word)
    return words


def get_opposites(filepath):
    with open(filepath, 'r') as file:
        opposites = yaml.safe_load(file)
    return opposites


def generate_sentence(dictionary_path, opposites_path, num_sentences, seed=0):
    nouns = get_words_by_type('noun', dictionary_path)
    opposites = get_opposites(opposites_path)

    # Calculate total possible sentences
    total_possible_sentences = len(nouns) * len(opposites) * 2
    sentences = []

    random.seed(seed)

    if num_sentences >= total_possible_sentences:
        # If requested sentences are more than possible, generate all combinations
        for noun in nouns:
            for opposite_pair in opposites:
                for word, opposite in opposite_pair.items():
                    sentences.append(f"{noun} li {word} ala. {noun} li {opposite}")
                    sentences.append(f"{noun} li {opposite} ala. {noun} li {word}")
                    
    else:
        # If requested sentences are less than possible, generate random combinations
        for _ in range(num_sentences):
            noun = random.choice(nouns)
            opposite_pair = random.choice(opposites)
            word, opposite = list(opposite_pair.items())[0]
            sentences.append(f"{noun} li {word} ala. {noun} li {opposite}" if random.randint(0, 1) == 0 else f"{noun} li {opposite} ala. {noun} li {word}")
            
    return sentences


def generate_tense_sentences(dictionary_path, num_sentences, seed=0):
    verbs = get_words_by_type('verb', dictionary_path) 
    tenses = ['tenpo ni la', 'tenpo pini la', 'tenpo kama la']
    nouns = get_words_by_type('noun', dictionary_path)
    
    # Calculate total possible sentences
    total_possible_sentences = len(nouns) * len(verbs) * len(tenses)
    
    sentences = []

    random.seed(seed)

    if num_sentences >= total_possible_sentences:
        # If requested sentences are more than possible, generate all combinations
        for noun in nouns:
            for verb in verbs:
                for tense in tenses:
                    sentence = f"{tense} {noun} li {verb}."
                    sentences.append(sentence)
    else:
        # If requested sentences are less than possible, generate random combinations
        for _ in range(num_sentences):
            noun = random.choice(nouns)
            verb = random.choice(verbs)
            tense = random.choice(tenses)
            sentence = f"{tense} {noun} li {verb}."
            sentences.append(sentence)

    return sentences


if __name__ == "__main__":
    
    # with open("./evals/colors.txt", "w") as f:
    #    f.write("\n\n".join(gen_colors_tests()))

    # with open("./evals/opposites.txt", "w") as f:
    #    f.write("\n\n".join(generate_sentence('./evals/dictionary.yaml', './evals/opposites.yaml', 99999)))
    
    print(generate_tense_sentences('./evals/dictionary.yaml', 10))


    print("Finished.")