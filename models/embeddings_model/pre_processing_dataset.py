import string  # Import the string module for string manipulation
from nltk import word_tokenize  # Import word_tokenize for tokenizing sentences
from loading_dataset import sheet  # Import the sheet object from the loading_dataset module
from nltk.corpus import stopwords, words  # Import stopwords and words corpus from NLTK
from nltk.stem import WordNetLemmatizer  # Import the WordNetLemmatizer for lemmatization
from spellchecker import SpellChecker  # Import SpellChecker for spelling correction
import pickle  # Import pickle for saving data to files
import os


# Load stopwords and words corpus
stop_words = set(stopwords.words('english'))
words_corpus = set(words.words())

# Initialize lemmatizer and spell checker
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


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
            corrected_tokens.append(word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_tokens.append(corrected_word)

    # Remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in corrected_tokens if
             word and word not in stop_words and word.isalpha()]
    return words  # Return list of words instead of joined sentence


# List to store cleaned sentences
sentences = []
intents = []

# Iterate through all rows in the first column and clean the sentences
for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=1, max_col=1, values_only=True):
    if row[0] is not None:  # Check if cell value is not empty
        cleaned_sentence = clean_text(row[0])
        sentences.append(cleaned_sentence)

# Iterate through all rows in the second column and get the intents
for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=2, max_col=2, values_only=True):
    if row[0] is not None:  # Check if cell value is not empty
        intents.append(row[0])

print(sentences)
print(intents)

# Remove entries where intent is 'intent'
filtered_sentences = []
filtered_intents = []

for sentence, intent in zip(sentences, intents):
    if intent != 'intent':
        filtered_sentences.append(sentence)
        filtered_intents.append(intent)

# Determine the path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(script_dir, '..', 'pickle')

# Ensure the directory exists
os.makedirs(pickle_dir, exist_ok=True)

# Use this path to save your files
with open(os.path.join(pickle_dir, 'sentences.pkl'), 'wb') as f:
    pickle.dump(filtered_sentences, f)
with open(os.path.join(pickle_dir, 'intents.pkl'), 'wb') as f:
    pickle.dump(filtered_intents, f)
