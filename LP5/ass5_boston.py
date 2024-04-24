import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import preprocessing

# Load Boston housing dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

# Display shapes of training and test data
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train output data shape:", Y_train.shape)
print("Actual Test output data shape:", Y_test.shape)

##Normalize the data

# Normalize the training and test data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Model Building

# Shape of input data
X_train[0].shape

# Create a Sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(128, activation='relu', input_shape=X_train[0].shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

# Print model summary
model.summary()

# Compile the model
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, validation_data=(X_test, Y_test))

# Evaluate the model on test data
results = model.evaluate(X_test, Y_test)
print(results)
