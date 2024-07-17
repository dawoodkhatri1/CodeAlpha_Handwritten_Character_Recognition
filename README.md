# Handwritten_Character_Recognition

You can run the file in Google Colab and Jupiter Notebook

I divided the task into 2 parts:
> Recognition by Digits
>> Recognition by Words

Recognition by Digits:

Downloading Libraries

Purpose: Ensure TensorFlow and Keras libraries are installed.
Action: Use the package manager to install the required libraries if they are not already installed.

Importing Libraries

NumPy: For numerical operations and data manipulation.
Matplotlib: For visualizing images and plots.
TensorFlow and Keras: For building and training the neural network.
Datasets: To load the MNIST dataset.
Layers: To define different types of neural network layers.
Models: To create and manage the structure of the neural network.
Utilities: For tasks like converting labels to categorical format.

Load and Preprocess Data

Loading Data: Fetch the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).
Normalizing Images: Scale pixel values to the range [0, 1] to improve the efficiency and performance of the neural network.
Reshaping Data: Reshape the images to include a single channel dimension, making them suitable for processing by the CNN.
One-Hot Encoding Labels: Convert the labels to a categorical format to facilitate multi-class classification.

Build the Convolutional Neural Network (CNN)

Model Creation: Define a sequential model, which is a linear stack of layers.
Convolutional Layers: Apply convolution operations to extract features from the input images.
Max Pooling Layers: Downsample the feature maps to reduce the dimensionality and computational load.
Flatten Layer: Convert the 2D feature maps into a 1D vector before passing them to the dense layers.
Dense Layers: Fully connected layers for learning the final classification.
Output Layer: Use softmax activation to output probabilities for each class (digit 0-9).
Compilation: Configure the model with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.

Train the Model

Training Process: Train the model on the training dataset for a specified number of epochs, using the test dataset for validation.
History Object: Store training history, which includes metrics like accuracy and validation accuracy for each epoch.

Evaluate the Model

Model Evaluation: Assess the trained model's performance on the test dataset to obtain accuracy and loss metrics.
Print Test Accuracy: Output the test accuracy to see how well the model generalizes to unseen data.

Visualize Training History

Plotting: Create plots for training and validation accuracy over epochs to visualize the model's learning progress and check for overfitting or underfitting.

Recognize Handwritten Characters

Prediction Function: Define a function to preprocess a given image, predict its label using the trained model, and display the image along with the predicted label.
Preprocessing: Ensure the input image is in the correct shape and normalized.
Prediction: Use the model to predict the label and display the result.
Test the Function: Use the function to predict and display the label for a sample image from the test dataset, and compare the predicted label with the actual label.

Output looks like this:

![image](https://github.com/user-attachments/assets/4340a39a-a393-4687-ba5b-756f7cc53311)

![image](https://github.com/user-attachments/assets/47ada4a3-9e5b-4269-8c74-10fa800ca618)

![image](https://github.com/user-attachments/assets/fd0eb541-8c80-421f-8802-50ae442e902f)

Recognition by Words:

Data Collection

Downloading and Extracting Data:
Download: The code downloads the IAM Words dataset archive from a specified URL.
Extraction: It extracts the contents of the downloaded archive.
Setup Directories: Creates necessary directories to store the data.
Inspecting Data:
View Sample Data: Displays the first 20 lines of the words.txt file to understand the dataset's structure.

Importing Libraries

TensorFlow and Keras: Libraries for building and training neural networks.
Matplotlib: For visualizing images and plots.
NumPy: For numerical operations.
OS: For operating system dependent functionality like file paths.

Dataset Splitting

Read Data: Reads the words.txt file, which contains metadata and paths to the images.
Filter Data: Filters out lines with errors and prepares a list of valid entries.
Shuffle Data: Randomly shuffles the dataset to ensure randomness.
Split Data: Divides the dataset into training (90%), validation (5%), and test (5%) sets.

Data Input Pipeline

Prepare Image Paths and Labels:
Extract Paths: Constructs full image paths from the metadata and checks their existence.
Clean Labels: Processes labels to remove unnecessary characters and whitespace.
Build Character Vocabulary:
Unique Characters: Identifies all unique characters in the training labels to create a vocabulary.
String Lookup: Uses TensorFlow’s StringLookup to map characters to integers and vice versa.
Resizing Images without Distortion:
Resize and Pad: Resizes images while preserving their aspect ratio and pads them to maintain uniform dimensions.
Preprocessing Functions:
Image Preprocessing: Reads, decodes, resizes, and normalizes images.
Label Vectorization: Converts labels to sequences of integers, padding them to a fixed length.
Dataset Preparation:
Create TensorFlow Dataset: Combines image paths and labels into TensorFlow datasets.
Batching and Caching: Batches the data, caches it, and prefetches it for efficient processing during training.

Prepare tf.data.Dataset Objects

Training, Validation, and Test Datasets: Prepares separate datasets for training, validation, and testing using the previously defined preprocessing functions and batching mechanisms.

Visualization

Sample Display: Visualizes a few samples from the training dataset to check if the preprocessing steps have been correctly applied.
Image Transformation: Transforms images back to their original form for display.
Label Display: Converts numerical labels back to string format and displays them along with corresponding images.

Output looks like this:

![image](https://github.com/user-attachments/assets/ab619e61-f4db-4b7c-843a-74faec534e69)

![image](https://github.com/user-attachments/assets/5708553f-b200-407d-8822-ed69fe15e89b)

![image](https://github.com/user-attachments/assets/091e5680-095f-495d-9076-8cb9aa3a0d47)

![image](https://github.com/user-attachments/assets/7b662a8b-18ff-4313-b880-46311d986a4b)
