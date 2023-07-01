import requests
import pickle
import re


url = "https://devurandom.xyz/tokipona/dictionary"
response = requests.get(url)
text = response.text
lines = text.split("\n")
words = [l for l in lines if l.startswith("<a name=")]
words = [w.split("\"")[1] for w in words]
words_full = []
for word in words:
    subwords = re.findall(r"-[a-z]+-", f"-{word}-")
    for sw in subwords:
        words_full.append(sw[1:-1])
words = list(set(words_full))
outfile = open("dictionary.pickle", "wb")
pickle.dump(words, outfile); outfile.close()