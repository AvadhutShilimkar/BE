from keras.datasets import imdb

# Load IMDB dataset with only the 10,000 most frequently occurring words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Find the maximum index value of words in the dataset
max_word_index = max([max(sequence) for sequence in train_data])

# Retrieve the word index dictionary
word_index = imdb.get_word_index()

# Reverse the word index dictionary to map indices to words
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])

# Decode the first review in the training data to its original text form
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np

# Function to vectorize sequences into binary matrix representation
def vectorize(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# Vectorize the training and test data
x_train = vectorize(train_data)
x_test = vectorize(test_data)

# Convert the labels to numpy arrays
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

# Define the model architecture
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Split validation set from the training data
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# Train the model
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# Evaluate the model on the test data
results = model.evaluate(x_test, y_test)
print(results)
