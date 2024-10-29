# README

## MNIST Neural Network Classifier

This project demonstrates a neural network trained to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. It includes two main scripts:

    1. main.py - trains a model on the MNIST dataset and saves it.
    2. check.py - loads the saved model and predicts classes for custom test images of handwritten digits.

### Project Structure

    - main.py: Script for training and saving the MNIST classifier model.
    - check.py: Script to load and test the model on custom images.
    - model.h5: The saved model after training (main.py).
    - data/:
        - train-images.idx3-ubyte,  train-labels.idx1-ubyte: MNIST training images and labels.
        - t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte: MNIST test images and labels.
        -num0.png to num9.png: Custom test images, each containing a single 28x28 grayscale digit for testing in check.py.

### Requirements

Ensure Python 3.6 or later and install 
required packages:
    bash
        pip install tensorflow numpy pillow matplotlib

### Code overview  

#### main.py

    1. Data Loading and Preprocessing:

        - Reads MNIST image and label files using custom functions load_mnist_images and load_mnist_labels.
        - Normalizes images to [0, 1] and one-hot encodes labels.

    2. Model Architecture:
        - Defines a Sequential neural network with dense layers and dropout for regularization.
        - Uses ReLU activations and softmax for multi-class classification.

    3. Training and Evaluation:
        -Trains the model on MNIST data for 10 epochs and evaluates accuracy on the test set.
        - Outputs the first 20 predictions on the test set and saves the model as model.h5.

    4. Run main.py
        bash
                python main.py

#### Sample Output (main.py):

    train_x.shape: (60000, 28, 28), train_y.shape: (60000,)
test_x.shape: (10000, 28, 28), test_y.shape: (10000,)

Epoch 1/10
469/469 [5s] - accuracy: 0.8809 - loss: 0.3948 - val_accuracy: 0.9740 - val_loss: 0.0881
...
Epoch 10/10
469/469 [5s] - accuracy: 0.9958 - loss: 0.0133 - val_accuracy: 0.9830 - val_loss: 0.0741

Test loss: 0.0741
Test accuracy: 0.9830
Original labels: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
Predicted labels: [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 8 4]
Model saved to ./model.h5

#### check.py

        1. Model Loading:
            - Loads the trained model.h5 for prediction on custom images.

        2. Custom Image Preprocessing:
            -   Converts each image (num0.png to num9.png) to grayscale, inverts colors, resizes to 28x28, normalizes to [0, 1], and reshapes for model input.

        3. Prediction:
            - Predicts the digit class for each image and prints the result.

        4. Run check.py:
            bash
                python check.py

### Improvements & Tips

#### To improve accuracy:

    - Enhance custom image preprocessing to standardize digit position and contrast.
    - Retrain or fine-tune the model for more robustness to variations in handwritten digits.

#### Conclusion

    This project illustrates digit recognition with a neural network trained on MNIST. While predictions are fairly accurate, further customization of images or model adjustments can refine results.

