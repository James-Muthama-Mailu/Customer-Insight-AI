import os
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths using current_dir
sentences_file = os.path.join(current_dir, '../pickle/sentences.pkl')

# Load preprocessed data from files
# 'sentences.pkl' contains a list of tokenized and cleaned sentences
with open(sentences_file, 'rb') as f:
    sentences = pickle.load(f)

# Creating Bigrams
bigrams = []

for word_list in sentences:
    for i in range(len(word_list) - 1):
        for j in range(i + 1, len(word_list)):
            bigrams.append([word_list[i], word_list[j]])

# Getting all unique words in our sentences
all_words = []

for sent in sentences:
    all_words.extend(sent)

all_words = list(set(all_words))

all_words.sort()

# Performing One Hot Encoding and Creating
words_dict = {}

counter = 0

for word in all_words:
    words_dict[word] = counter
    counter += 1

onehot_data = np.zeros((len(all_words), len(all_words)))

for i in range(len(all_words)):
    onehot_data[i][i] = 1


print(all_words)


onehot_dict = {}

for i in range(len(all_words)):
    onehot_dict[all_words[i]] = onehot_data[i]

X = []
Y = []

for bi in bigrams:
    X.append(onehot_dict[bi[0]])
    Y.append(onehot_dict[bi[1]])

X = np.array(X)
Y = np.array(Y)

# Training word2vec model
model = Sequential()

vocab_size = len(onehot_data[0])

model.add(Input(shape=(vocab_size,)))
model.add(Dense(100, activation='linear'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X, Y, epochs=100)

weights = model.get_weights()[0]

word_embeddings = {}

for word in all_words:
    word_embeddings[word] = weights[words_dict[word]]

for word in all_words:
    print(word, ":", word_embeddings[word])

with open('../pickle/word_embeddings.pkl', 'wb') as f:
    pickle.dump(word_embeddings, f)
