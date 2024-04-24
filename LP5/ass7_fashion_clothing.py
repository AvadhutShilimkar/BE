from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_img = train_img / 255.0
test_img = test_img / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input image
    keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_img, train_labels, epochs=10)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_img, test_labels)
print("Accuracy on testing data:", test_acc)

# Make predictions on the test data
predictions = model.predict(test_img)

# Get the predicted labels for each image
predicted_labels = np.argmax(predictions, axis=1)

# Display a grid of images along with their predicted labels and prediction probabilities
num_rows = 5 
num_cols = 5
num_imgs = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.imshow(test_img[i], cmap='gray')
    plt.axis("off")
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.title(f"Predicted Label: {predicted_labels[i]}")
    plt.show()
