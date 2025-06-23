import nltk
import numpy as np

# Hatanın düzeltildiği ve daha anlaşılır hale getirildiği kısım
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' verisi bulunamadı. İndiriliyor...")
    nltk.download('punkt')
    print("'punkt' başarıyla indirildi.")

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Splits a sentence into an array of words/tokens.
    A token can be a word or punctuation character, or a number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Finds the root form of a word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Returns a bag of words array: 1 for each known word that exists in the sentence, 0 otherwise.
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag