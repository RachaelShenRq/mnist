import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
import struct
import os

def load_mnist_images(filename):
    # Load MNIST images from the specified file
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    # Load MNIST labels from the specified file
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def run():
    # Load the dataset from the .idx files
    train_x = load_mnist_images('./data/train-images.idx3-ubyte')
    train_y = load_mnist_labels('./data/train-labels.idx1-ubyte')
    test_x = load_mnist_images('./data/t10k-images.idx3-ubyte')
    test_y = load_mnist_labels('./data/t10k-labels.idx1-ubyte')

    print("train_x.shape:", train_x.shape, ",  train_y.shape:", train_y.shape)
    print("test_x.shape:", test_x.shape, ",  test_y.shape:", test_y.shape)

    # Normalize the images (0-255 -> 0-1)
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # One-hot encode the labels
    train_y = tf.keras.utils.to_categorical(train_y, 10)
    test_y = tf.keras.utils.to_categorical(test_y, 10)
    print("train_y.shape:", train_y.shape, ",  test_y.shape:", test_y.shape)

    # Build the neural network model
    model = tf.keras.models.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(1000, activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    # Model summary
    model.summary()

    # Compile the model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), batch_size=128)

    # Evaluate the model
    score = model.evaluate(test_x, test_y, batch_size=128)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Predict with the model
    result = model.predict(test_x, batch_size=128)
    print("Original labels:", np.argmax(test_y[:20], axis=1))
    print("Predicted labels:", np.argmax(result[:20], axis=1))

    # Save the trained model
    model_filename = "./model.h5"
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

run()
