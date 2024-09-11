import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths using current_dir
sentences_file = os.path.join(current_dir, '../pickle/sentences.pkl')
intents_file_path = os.path.join(current_dir, '../pickle/intents.pkl')
word_embeddings_file = os.path.join(current_dir, '../pickle/word_embeddings.pkl')

# Load preprocessed data from files
# 'sentences.pkl' contains a list of tokenized and cleaned sentences
with open(sentences_file, 'rb') as f:
    sentences = pickle.load(f)

# 'intents.pkl' contains a list of intent labels corresponding to the sentences
with open(intents_file_path, 'rb') as f:
    intents = pickle.load(f)

# Load mean_embeddings from pickle file (assuming mean_embeddings is a dictionary)
with open(word_embeddings_file, 'rb') as f:
    word_embeddings = pickle.load(f)

def sentence_embedding(sentences, word_embeddings):
    embedded_sentences = []

    for sentence in sentences:
        embedded_sentence = []

        for word in sentence:
            if word in word_embeddings:
                embedded_sentence.append(word_embeddings[word])

        if embedded_sentence:
            # Compute the mean embedding for the sentence
            mean_embedding = np.mean(embedded_sentence, axis=0)
            embedded_sentences.append(mean_embedding)

    if not embedded_sentences:
        raise ValueError("No valid embeddings found for sentences")

    return np.array(embedded_sentences)

# Generate the sentence embeddings using mean of word embeddings
sentence_embeddings = sentence_embedding(sentences, word_embeddings)

# Encode the intent labels as integers
encoder = LabelEncoder()
encoded_intents = encoder.fit_transform(intents)

# Store the unique intents in the order as they are encoded
intents_file = list(encoder.classes_)

# Save the intents_file variable to a pickle file if needed
with open(os.path.join(current_dir, '../pickle/ordered_intents.pkl'), 'wb') as f:
    pickle.dump(intents_file, f)

# Convert the encoded intent labels to one-hot vectors
one_hot_intents = to_categorical(encoded_intents)

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(0.8 * len(sentence_embeddings))
train_x = sentence_embeddings[:train_size]
train_y = one_hot_intents[:train_size]
test_x = sentence_embeddings[train_size:]
test_y = one_hot_intents[train_size:]

