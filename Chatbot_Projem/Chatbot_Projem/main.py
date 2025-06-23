# This part of the code would typically be in your main application file

# A sample vocabulary (in a real scenario, this would be built from your training data)
all_words = ["hello", "how", "are", "you", "bye", "thank", "cool", "day", "see", "later"]

# Get a sentence from the user
sentence = input("Please enter a sentence: ")

# --- 1. Tokenization ---
tokenized_sentence = tokenize(sentence)
print("\n--- Tokenized Sentence ---")
print(tokenized_sentence)

# --- 2. Stemming ---
stemmed_words = [stem(w) for w in tokenized_sentence]
print("\n--- Stemmed Words ---")
print(stemmed_words)

# --- 3. Bag-of-Words ---
bog = bag_of_words(tokenized_sentence, all_words)
print("\n--- Bag-of-Words ---")
print(bog)
print("\nThis vector represents which words from our vocabulary are in your sentence.")