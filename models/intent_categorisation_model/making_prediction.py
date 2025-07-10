import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import pickle
import numpy as np
import os
from .loading_categorisation_model import load_or_train_model  # Ensure this loads CustomerInsightAI_improved.keras

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/words.zip')
except LookupError:
    print("Downloading words corpus...")
    nltk.download('words')

# Load stopwords and words corpus
stop_words = set(stopwords.words('english'))  # Set of common English stopwords to filter out
words_corpus = set(words.words())  # Set of valid English words for spell checking
model = load_or_train_model()  # Load the trained model (CustomerInsightAI_improved.keras)

# Initialize lemmatizer and spell checker
lemmatizer = WordNetLemmatizer()  # Tool to reduce words to their base form
spell = SpellChecker()  # Tool to correct spelling mistakes

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths using current_dir
word_embeddings_file = os.path.join(current_dir, '../pickle/word_embeddings.pkl')  # Path to word embeddings file
ordered_intents_file = os.path.join(current_dir, '../pickle/ordered_intents.pkl')  # Path to intents list file
# Removed customer_agent_sentences_file as training data doesn't filter agent sentences

# Load word embeddings
with open(word_embeddings_file, 'rb') as f:
    word_embeddings = pickle.load(f)  # Load pre-trained word embeddings (1024 dimensions)

# Load ordered intents (this matches the order used during training)
with open(ordered_intents_file, 'rb') as f:
    intents_list = pickle.load(f)  # Load list of intent categories


def clean_text(text):
    """
    Clean text using the SAME preprocessing as training data.
    This is crucial for consistency with the training pipeline.
    """
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
            corrected_tokens.append(word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_tokens.append(corrected_word)
            else:
                corrected_tokens.append(word)  # Retain original word if no correction found

    # Remove stopwords and apply lemmatization (same as training preprocessing)
    words = [lemmatizer.lemmatize(word) for word in corrected_tokens if
             word and word not in stop_words and word.isalpha()]

    print(f"Cleaned tokens: {words}")  # Debug: Log cleaned tokens to diagnose filtering issues
    return words


def sentence_embedding(sentence_tokens, word_embeddings):
    """
    Generate sentence embedding using the SAME method as training (mean of word embeddings).
    This ensures consistency between training and prediction.
    """
    embedded_sentence = []

    for word in sentence_tokens:
        if word in word_embeddings:
            embedded_sentence.append(word_embeddings[word])

    if embedded_sentence:
        # Compute the mean embedding for the sentence (same as training)
        mean_embedding = np.mean(embedded_sentence, axis=0)
        print(f"Sentence embedding: {mean_embedding}")  # Display the embedding for valid sentences
        return mean_embedding
    else:
        # Handle sentences with no embeddings - use zero vector (same as training)
        print(f"Warning: Sentence with no embeddings found, using zero vector")
        return np.zeros(1024)  # Match the 1024-dimensional embedding used in training


def process_text_for_prediction(transcribed_audio, word_embeddings):
    """
    Process text for prediction using consistent methodology with training.
    Removed customer_agent_sentences filtering to match training data generation.
    """
    # Clean the input text (returns list of tokens)
    cleaned_tokens = clean_text(transcribed_audio)

    # Generate sentence embedding using the same method as training
    sentence_emb = sentence_embedding(cleaned_tokens, word_embeddings)
    # print(f"Processed embedding for sentence: {sentence_emb}")  # Display the processed embedding
    return [sentence_emb]  # Return as list for consistency


def predictions(model, sentence_embeddings, intents_list):
    """
    Make predictions and return full probability distribution for all classes.
    """
    results = []
    all_probabilities = []

    for sentence_embedding in sentence_embeddings:
        # Reshape the embedding to match model input shape (1024 dimensions)
        sentence_embedding = np.expand_dims(sentence_embedding, axis=0)

        # Get prediction from the model (full softmax output)
        prediction = model.predict(sentence_embedding, verbose=0)
        all_probabilities.append(prediction[0])  # Store full probability distribution

        # Optionally find the top prediction (for reference)
        prediction_index = np.argmax(prediction)
        probability = prediction[0][prediction_index]
        results.append((intents_list[prediction_index], probability))

    return results, all_probabilities


def classify_text_results(transcribed_audio):
    """
    Main function to classify text and return results.
    """
    # Process the text to get sentence embeddings
    sentence_embeddings = process_text_for_prediction(
        transcribed_audio,
        word_embeddings
    )

    # Get predictions and full probability distributions
    classification_results, all_probabilities = predictions(model, sentence_embeddings, intents_list)

    if not classification_results:
        return ["No category found"], ["No category found"], [np.zeros(len(intents_list))]

    categories = []
    descriptions = []

    # Extract categories and probabilities for those above 75%
    for category, probability in classification_results:
        if probability > 0.75:  # 75% threshold
            categories.append(category)
            descriptions.append(f"{probability:.2%}")

    if not categories:
        return ["No category found"], ["No category found"], all_probabilities[0]

    return categories, descriptions, all_probabilities[0]  # Return the first (and only) probability distribution


def classify_sentences(sentences_list):
    """
    Process multiple sentences and aggregate high-confidence predictions (>75%).
    Tracks the maximum probability for each category across all sentences.
    """
    all_embeddings = []
    all_class_probabilities = []
    sentence_results = []  # Store results for each sentence
    category_max_probs = {}  # Track max probability for each category

    # Process each sentence individually
    for sentence in sentences_list:
        embeddings = process_text_for_prediction(
            sentence,
            word_embeddings
        )
        all_embeddings.extend(embeddings)
        # print(f"Embeddings for sentence '{sentence}': {embeddings}")

        # Get predictions for the current sentence
        classification_results, probabilities = predictions(model, embeddings, intents_list)
        all_class_probabilities.extend(probabilities)

        # Print probability distribution for debugging
        if probabilities:
            print(f"\nProbability distribution for sentence '{sentence}':")
            for intent, prob in zip(intents_list, probabilities[0] * 100):  # Convert to percentages
                print(f"{intent}: {prob:.2f}%")

        # Extract categories and probabilities above 75%
        categories = []
        descriptions = []
        if probabilities:
            for intent, prob in zip(intents_list, probabilities[0]):
                if prob > 0.75:  # 75% threshold
                    categories.append(intent)
                    descriptions.append(f"{prob:.2%}")
                    # Update max probability for this category
                    if intent not in category_max_probs or prob > category_max_probs[intent]:
                        category_max_probs[intent] = prob

        # Store results for this sentence
        if categories:
            sentence_results.append((sentence, categories, descriptions))
        else:
            sentence_results.append((sentence, ["No category found"], ["No category found"]))

    # Aggregate high-confidence predictions across all sentences
    aggregated_categories = set()
    for _, cats, _ in sentence_results:
        if cats != ["No category found"]:
            aggregated_categories.update(cats)

    # Convert to list for output
    aggregated_categories = list(aggregated_categories) if aggregated_categories else ["No category found"]

    # Get corresponding max probabilities
    if aggregated_categories != ["No category found"]:
        aggregated_descriptions = [f"{category_max_probs[cat] * 100:.2f}%" for cat in aggregated_categories]
    else:
        aggregated_descriptions = ["No category found"]

    return aggregated_categories, aggregated_descriptions, all_class_probabilities

# Update the main execution block
# sentences = sent_tokenize(
#     'Hello, I ordered a pair of shoes from your website last week, and they still haven’t arrived. I’m really frustrated because I needed them for an event tomorrow. Can you please check on the status? . Thank you for reaching out, I apologize for the inconvenience. Let me check the tracking information for you. . It looks like there was a delay due to a shipping issue, and the package is now scheduled to arrive by tomorrow afternoon. Would you like me to expedite it or issue a refund if it doesn’t meet your timeline? . I’d prefer an expedited delivery if possible, as I really need the shoes for the event. Thank you! . Great, I’ve requested the expedited shipping, and you should receive an updated tracking link shortly. Please let me know if there’s anything else I can assist with!'
# )
# categories, probabilities, class_probabilities = classify_sentences(sentences)
# print("\nPredicted Categories:", categories)
# print("Top Probabilities:", probabilities)
