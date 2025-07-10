import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths using current_dir
sentences_file = os.path.join(current_dir, '../pickle/sentences.pkl')
intents_file_path = os.path.join(current_dir, '../pickle/intents.pkl')
word_embeddings_file = os.path.join(current_dir, '../pickle/word_embeddings.pkl')

# Load preprocessed data from files
with open(sentences_file, 'rb') as f:
    sentences = pickle.load(f)

with open(intents_file_path, 'rb') as f:
    intents = pickle.load(f)

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
        else:
            # Handle sentences with no embeddings - use zero vector
            print(f"Warning: Sentence with no embeddings found, using zero vector")
            mean_embedding = np.zeros(1024)  # Assuming 1024-dim embeddings
            embedded_sentences.append(mean_embedding)

    return np.array(embedded_sentences)


# Generate the sentence embeddings using mean of word embeddings
sentence_embeddings = sentence_embedding(sentences, word_embeddings)

print(f"Generated embeddings for {len(sentence_embeddings)} sentences")
print(f"Embedding dimension: {sentence_embeddings.shape[1]}")

# Encode the intent labels as integers
encoder = LabelEncoder()
encoded_intents = encoder.fit_transform(intents)

# Store the unique intents in the order as they are encoded
intents_file = list(encoder.classes_)

# Save the intents_file variable to a pickle file if needed
with open(os.path.join(current_dir, '../pickle/ordered_intents.pkl'), 'wb') as f:
    pickle.dump(intents_file, f)

print(f"Number of unique classes: {len(intents_file)}")
print(f"Classes: {intents_file}")

# Convert the encoded intent labels to one-hot vectors
one_hot_intents = to_categorical(encoded_intents)

# Check class distribution before splitting
unique, counts = np.unique(encoded_intents, return_counts=True)
print(f"\nClass distribution in full dataset:")
for class_idx, count in zip(unique, counts):
    print(f"Class {class_idx} ({intents_file[class_idx]}): {count} samples")

# Filter out classes with very few samples (less than 2 samples can't be split)
min_samples_per_class = 2
valid_indices = []
valid_class_counts = {}

for i, intent_idx in enumerate(encoded_intents):
    if counts[intent_idx] >= min_samples_per_class:
        valid_indices.append(i)
        valid_class_counts[intent_idx] = valid_class_counts.get(intent_idx, 0) + 1

print(f"\nFiltered dataset:")
print(f"Original samples: {len(sentence_embeddings)}")
print(f"Valid samples (classes with ≥{min_samples_per_class} samples): {len(valid_indices)}")

if len(valid_indices) < len(sentence_embeddings):
    print("⚠️  Some samples were removed due to insufficient class representation")

# Filter the data
sentence_embeddings = sentence_embeddings[valid_indices]
encoded_intents = encoded_intents[valid_indices]
one_hot_intents = one_hot_intents[valid_indices]

# Use stratified split to ensure all classes are represented in both train and test
try:
    train_x, test_x, train_y_encoded, test_y_encoded = train_test_split(
        sentence_embeddings,
        encoded_intents,
        test_size=0.2,
        random_state=42,
        stratify=encoded_intents  # This ensures each class is represented in both splits
    )

    # Convert back to one-hot encoding
    train_y = to_categorical(train_y_encoded, num_classes=len(intents_file))
    test_y = to_categorical(test_y_encoded, num_classes=len(intents_file))

    print(f"\n✅ Successful stratified split:")
    print(f"Training samples: {len(train_x)}")
    print(f"Test samples: {len(test_x)}")

    # Verify all classes are present in both sets
    train_classes = set(train_y_encoded)
    test_classes = set(test_y_encoded)

    print(f"Classes in training: {len(train_classes)}")
    print(f"Classes in test: {len(test_classes)}")
    print(f"Classes in both: {len(train_classes.intersection(test_classes))}")

    if train_classes == test_classes:
        print("✅ All classes present in both training and test sets")
    else:
        print("⚠️  Class distribution mismatch between train and test")

except ValueError as e:
    print(f"❌ Stratified split failed: {e}")
    print("Falling back to random split...")

    # Fallback to regular train_test_split
    train_x, test_x, train_y, test_y = train_test_split(
        sentence_embeddings,
        one_hot_intents,
        test_size=0.2,
        random_state=42
    )

    print(f"Training samples: {len(train_x)}")
    print(f"Test samples: {len(test_x)}")

print(f"\nFinal data shapes:")
print(f"train_x: {train_x.shape}")
print(f"train_y: {train_y.shape}")
print(f"test_x: {test_x.shape}")
print(f"test_y: {test_y.shape}")

# Save the properly split data
np.save(os.path.join(current_dir, '../pickle/train_x.npy'), train_x)
np.save(os.path.join(current_dir, '../pickle/train_y.npy'), train_y)
np.save(os.path.join(current_dir, '../pickle/test_x.npy'), test_x)
np.save(os.path.join(current_dir, '../pickle/test_y.npy'), test_y)

print(f"\n✅ Fixed preprocessed data saved to pickle directory")