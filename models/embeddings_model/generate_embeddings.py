import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Path to the locally downloaded model
local_model_path = r"C:\Users\james\PycharmProjects\CustomerInsightAI\Customer-Insight-AI\models\embeddings_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

def get_sentence_embedding(sentence):
    """
    Generate embedding for a sentence using the multilingual-e5-large-instruct model.
    Args:
        sentence (str): Input sentence.
    Returns:
        numpy.ndarray: 1024-dimensional embedding (normalized).
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling over the token embeddings to get a single vector for the sentence
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    # Normalize the embedding to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# Sentences to compare
sentence1 = "I am happy"
sentence2 = "I am happy"

# Get embeddings
embedding1 = get_sentence_embedding(sentence1)
embedding2 = get_sentence_embedding(sentence2)

# Compute dot product
dot_product = np.dot(embedding1.flatten(), embedding2.flatten())

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Dot product (normalized embeddings): {dot_product}")