# bthelmer Braden T Helmer
## Stage 1: Installing dependencies and notebook gpu setup
import os
from sys import argv

# Get input argument from command line.
EVALUATOR = -1
if not len(argv) > 1:
    print("Please provide input for evaluator (-1) or worker (0..)")
    exit(1)

ID = argv[1];

os.environ["KERAS_BACKEND"] = "tensorflow"

## Stage 2: Importing dependencies for the project

os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
os.system("ln -s /mnt/beegfs/fmuelle/.keras ~/")
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

#for RTX GPUs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## Stage 3: Dataset preprocessing
### Loading the Cifar10 dataset
#Setting class names for the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#Loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

### Image normalization
X_train = X_train / 255.0
X_train.shape
X_test = X_test / 255.0

## Stage 4: Building a Convolutional neural network
### Defining the model
model = tf.keras.models.Sequential()

### Adding the first CNN Layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

### Adding the second CNN Layer and max pool layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

### Adding the third CNN Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

###  Adding the fourth CNN Layer and max pool layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

### Adding the Flatten layer
model.add(tf.keras.layers.Flatten())

### Adding the first Dense layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

### Adding the second Dense layer (output layer)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

### Compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam", metrics=["sparse_categorical_accuracy"])

## HOMEWORK SOLUTION

### Training the model
model.fit(X_train, y_train, epochs=15)

### Model evaluation and prediction
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

