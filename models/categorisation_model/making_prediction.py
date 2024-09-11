import string  # Import the string module for string manipulation
from nltk import word_tokenize, sent_tokenize  # Import word_tokenize and sent_tokenize for tokenizing sentences
from nltk.corpus import stopwords, words  # Import stopwords and words corpus from NLTK
from nltk.stem import WordNetLemmatizer  # Import the WordNetLemmatizer for lemmatization
from spellchecker import SpellChecker  # Import SpellChecker for spelling correction
import pickle  # Import pickle for saving data to files
import numpy as np
import os
from collections import Counter
from models.categorisation_model.loading_categorisation_model import load_or_train_model  # Assuming this function loads your trained model

# Load stopwords and words corpus
stop_words = set(stopwords.words('english'))  # Set of English stopwords
words_corpus = set(words.words())  # Set of English words from NLTK corpus
model = load_or_train_model()  # Load or train your categorization model

# Initialize lemmatizer and spell checker
lemmatizer = WordNetLemmatizer()  # Create an instance of WordNetLemmatizer
spell = SpellChecker()  # Create an instance of SpellChecker

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths using current_dir
sentences_file = os.path.join(current_dir, '../pickle/sentences.pkl')
intents_file = os.path.join(current_dir, '../pickle/intents.pkl')
word_embeddings_file = os.path.join(current_dir, '../pickle/word_embeddings.pkl')
customer_agent_sentences_file = os.path.join(current_dir, '../pickle/customer_agent_sentences.pkl')


# Load word embeddings
with open(word_embeddings_file, 'rb') as f:
    word_embeddings = pickle.load(f)  # Load pre-trained word embeddings from a pickle file

# Load intents (assuming 'intents.pkl' contains a list of intent names corresponding to your model output)
with open(intents_file, 'rb') as f:
    intents_list = pickle.load(f)  # Load intents from a pickle file

with open(customer_agent_sentences_file, 'rb') as f:
    customer_agent_sentences = pickle.load(f)


# Function to clean text
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Correct spelling and replace with closest correctly spelled word
    corrected_tokens = []
    for word in tokens:
        if word in words_corpus:
            corrected_tokens.append(word)  # If word is in corpus, add it as-is
        else:
            corrected_word = spell.correction(word)  # Correct misspelled words
            if corrected_word is not None:
                corrected_tokens.append(corrected_word)  # Add corrected word if available

    # Remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in corrected_tokens if
             word and word not in stop_words and word.isalpha()]  # Lemmatize and filter stopwords
    return words  # Return list of cleaned words instead of joined sentence


def process_text(transcribed_audio, word_embeddings, clean_text, customer_agent_sentences):
    # Clean the sentence
    cleaned_sent = clean_text(transcribed_audio)

    sentence_embeddings = []
    for sentence in cleaned_sent:
        # Initialize an array to store words not found in word_embeddings
        cleaned_sentences = []

        for word in sentence:
            if word not in customer_agent_sentences:
                cleaned_sentences.append(word)  # Append word not found in word_embeddings to cleaned_sentences

        # Retrieve word embeddings for each word in the cleaned sentence
        embeddings = []
        for word in cleaned_sentences:
            if word in word_embeddings:
                embeddings.append(word_embeddings[word])  # Add word embedding if available
            else:
                embeddings.append(np.zeros(len(next(iter(word_embeddings.values())))))  # Assuming the vector size

        # If no valid embeddings are found, continue to next sentence
        if not embeddings:
            continue

        # Average the embeddings to get a single vector representing the sentence
        sentence_embedding = np.mean(embeddings, axis=0)

        # Check if sentence_embedding is not all zeros
        if not np.all(sentence_embedding == np.zeros_like(sentence_embedding)):
            sentence_embeddings.append(sentence_embedding)

    return sentence_embeddings  # Return list of sentence embeddings


# Function to make predictions using the model
def predictions(model, averaged_embeddings, intents_list, threshold=0.75):
    results = []
    for sentence_embedding in averaged_embeddings:
        # Reshape the embedding to match model input shape
        sentence_embedding = np.expand_dims(sentence_embedding, axis=0)

        # Get prediction from the model
        prediction = model.predict(sentence_embedding)

        # Find the prediction with the highest probability and its index
        prediction_index = np.argmax(prediction)
        probability = prediction[0][prediction_index]

        # Check if probability is above the threshold
        if probability > threshold:
            results.append((intents_list[prediction_index], probability))  # Append intent name and probability to results

    return results  # Return list of intent names and probabilities above threshold


def classify_text_results(transcribed_audio):
    # Process the customer sentences to get sentence embeddings
    sentence_embeddings = process_text(transcribed_audio, word_embeddings, clean_text, customer_agent_sentences)

    # Determine the number of sentence embeddings to average
    num_sentences = len(sentence_embeddings)

    if num_sentences < 5:
        num_averages = 1
    else:
        num_averages = num_sentences // 10

    # Get averaged sentence embeddings
    averaged_embeddings = []
    for i in range(num_averages):
        start_index = i * (num_sentences // num_averages)
        end_index = (i + 1) * (num_sentences // num_averages)
        averaged_embedding = np.mean(sentence_embeddings[start_index:end_index], axis=0)
        averaged_embeddings.append(averaged_embedding)

    # Get predictions for intents above 75% probability
    classification_results = predictions(model, averaged_embeddings, intents_list, threshold=0.75)

    if not classification_results:
        return ["No category found"], ["No category found"]

    categories = []
    descriptions = []

    # Extract categories and probabilities
    for category, probability in classification_results:
        categories.append(category)
        descriptions.append(f"{probability:.2f}%")

    # Ensure they are lists even if there is only one item
    return categories, descriptions


