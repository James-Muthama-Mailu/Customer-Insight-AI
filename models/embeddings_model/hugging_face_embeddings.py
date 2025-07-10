import os
import pickle
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path for sentences.pkl
sentences_file = os.path.join(current_dir, '../pickle/sentences.pkl')

# Load preprocessed data from sentences.pkl (list of tokenized and cleaned sentences)
with open(sentences_file, 'rb') as f:
    sentences = pickle.load(f)

# Get all unique words in the sentences
all_words = []
for sent in sentences:
    all_words.extend(sent)
all_words = list(set(all_words))
all_words.sort()

print("Unique words:", all_words)

# Path to the locally downloaded model
local_model_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\word2vec_model"  # Replace with the actual path, e.g., /home/user/models/multilingual-e5

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)


def get_word_embeddings(texts):
    """
    Generate embeddings for a list of words using the multilingual-e5-large-instruct model.
    Args:
        texts (list): List of words to embed.
    Returns:
        numpy.ndarray: Array of embeddings for the input words.
    """
    # Tokenize the input words
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling over the token embeddings to get a single vector per word
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings


# Generate embeddings for all unique words
word_embeddings = {}
batch_size = 32  # Process words in batches to avoid memory issues
for i in range(0, len(all_words), batch_size):
    batch_words = all_words[i:i + batch_size]
    batch_embeddings = get_word_embeddings(batch_words)
    for word, embedding in zip(batch_words, batch_embeddings):
        word_embeddings[word] = embedding

# Print the embeddings for each word
for word in all_words:
    print(f"{word} : {word_embeddings[word]}")

# Save the word embeddings to a pickle file
output_file = os.path.join(current_dir, '../pickle/word_embeddings.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(word_embeddings, f)

print(f"Word embeddings saved to {output_file}")