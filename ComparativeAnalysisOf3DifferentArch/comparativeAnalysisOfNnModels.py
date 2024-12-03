import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to range [0, 1] and reshape to include channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode the labels (digits 0-9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to build a Convolutional Neural Network (CNN)
def build_cnn():
    model = models.Sequential([
        # First convolutional layer with 32 filters and ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Max pooling to reduce spatial dimensions
        layers.MaxPooling2D((2, 2)),
        # Second convolutional layer with 64 filters and ReLU activation
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Another max pooling layer
        layers.MaxPooling2D((2, 2)),
        # Flatten the feature maps into a 1D vector
        layers.Flatten(),
        # Fully connected (dense) layer with 64 units and ReLU activation
        layers.Dense(64, activation='relu'),
        # Output layer with 10 units (one for each digit) and softmax activation
        layers.Dense(10, activation='softmax')
    ])
    # Compile the CNN with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to build an AutoEncoder model
def build_autoencoder():
    # Encoder: Compress input into a lower-dimensional representation
    encoder = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),  # Flatten the input
        layers.Dense(128, activation='relu'),  # First dense layer
        layers.Dense(64, activation='relu'),   # Compressed latent space
    ])
    # Decoder: Reconstruct input from the compressed representation
    decoder = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(64,)),  # First dense layer
        layers.Dense(28*28, activation='sigmoid'),  # Output layer with sigmoid activation
        layers.Reshape((28, 28, 1)),  # Reshape back to original image dimensions
    ])
    # Combine encoder and decoder into an AutoEncoder model
    autoencoder = models.Model(encoder.input, decoder(encoder.output))
    # Compile the encoder and autoencoder models with Mean Squared Error (MSE) loss
    encoder.compile(optimizer='adam', loss='mse')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Function to build a Recurrent Neural Network (RNN)
def build_rnn():
    model = models.Sequential([
        # Flatten the 2D input images row-wise using TimeDistributed
        layers.TimeDistributed(layers.Flatten(), input_shape=(28, 28, 1)),
        # RNN layer with 128 units
        layers.SimpleRNN(128, activation='relu', return_sequences=False),
        # Fully connected layer with 64 units
        layers.Dense(64, activation='relu'),
        # Output layer with 10 units (one for each digit) and softmax activation
        layers.Dense(10, activation='softmax'),
    ])
    # Compile the RNN with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the CNN model
cnn = build_cnn()
cnn.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)
cnn_acc = cnn.evaluate(x_test, y_test, verbose=0)[1]

# Train the AutoEncoder and evaluate its performance
autoencoder, encoder = build_autoencoder()
autoencoder.fit(x_train, x_train, epochs=3, batch_size=64, verbose=0)
# Extract encoded representations from the encoder
x_train_encoded = encoder.predict(x_train)
x_test_encoded = encoder.predict(x_test)
# Define a classifier model for encoded data
encoder_output_shape = x_train_encoded.shape[1:]
classification_layer = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=encoder_output_shape),  # Dense layer for classification
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes
])
classification_layer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classification_layer.fit(x_train_encoded, y_train, epochs=3, batch_size=64, verbose=0)
autoencoder_acc = classification_layer.evaluate(x_test_encoded, y_test, verbose=0)[1]

# Train and evaluate the RNN model
rnn = build_rnn()
rnn.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64)
rnn_acc = rnn.evaluate(x_test, y_test, verbose=0)[1]

# Print the accuracy results for each model
print(f"CNN Accuracy: {cnn_acc:.2f}")
print(f"AutoEncoder + Classifier Accuracy: {autoencoder_acc:.2f}")
print(f"RNN Accuracy: {rnn_acc:.2f}")
