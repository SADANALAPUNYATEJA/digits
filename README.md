# digit
 It recognise the digits with the help of ML techinques
 import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image, ImageOps
import numpy as np
# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0
# Display the first image in the training dataset
plt.imshow(x_train[0], cmap='gray')
plt.show()
# Build the model
model = Sequential([
 Flatten(input_shape=(28, 28)),
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')])
# Compile the model
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
# Build the model
model = Sequential([
 Flatten(input_shape=(28, 28)),
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=10)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Function to preprocess the image
def preprocess_image(image_path):
 img = Image.open(image_path).convert('L')
 img = ImageOps.invert(img)
 img = img.resize((28, 28))
 img = np.array(img) / 255.0
 img = img.reshape(1, 28, 28)
 return img
# Path to the handwritten digit image
image_path = 'digit.jpg'
new_image = preprocess_image(image_path)
# Predict the digit
prediction = model.predict(new_image)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")
# Display the test image and the predicted digit
plt.imshow(new_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
def preprocess_image(image_path):
 img = Image.open(image_path).convert('L')
 img = ImageOps.invert(img)
 img = img.resize((28, 28))
 img = np.array(img) / 255.0
 img = img.reshape(1, 28, 28)
 return img
# Path to the handwritten digit image
image_path = 'digit.jpg'
new_image = preprocess_image(image_path)
# Predict the digit
prediction = model.predict(new_image)
print(f"Predicted Digit: {np.argmax(prediction)}")
# Path to the handwritten digit image
image_path = 'digit.jpg'
new_image = preprocess_image(image_path)

