# bthelmer Braden T Helmer
# Ckavana  Colin WB Kavanaugh
## Stage 1: Installing dependencies and notebook gpu setup
import os
from sys import argv
import time

# Get input argument from command line.
EVALUATOR = -1
if not len(argv) > 1:
    print("Please provide input for evaluator (-1) or worker (0..)")
    exit(1)

ID = argv[1]

if ID == "-1":
  node_role = "evaluator"
elif ID.isdigit():
  node_role = "worker"
else:
  raise ValueError("Invalid node role specified")

os.environ["KERAS_BACKEND"] = "tensorflow"

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

## Stage 2: Importing dependencies for the project

os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
os.system("ln -s /mnt/beegfs/fmuelle/.keras ~/")
import tensorflow as tf
import keras
# from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
print(tf.__version__, keras.__version__)
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


def get_dataset():
  #Loading the dataset
  #moved to get_dataset
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  ### Image normalization
  X_train = X_train / 255.0
  X_train.shape
  X_test = X_test / 255.0

  return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = get_dataset()
## Stage 4: Building a Convolutional neural network
def get_compiled_model():
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

    return model

## HOMEWORK SOLUTION

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()

### Training the model
def run_training(train_dataset, epochs=15):
    print("Doing training")
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

        callbacks = [
            # This callback saves a SavedModel every epoch
            # We include the current epoch in the folder name.
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir + "/ckpt-{epoch}.keras",
                save_freq="epoch",
            )
        ]

        model.fit(
            *train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
        )
          

# model.fit(X_train, y_train, epochs=15)
if node_role == "worker":
  run_training((X_train, y_train), epochs=15)

### Model evaluation and prediction
def run_testing():
    print("Doing testing")
    test_done = 0
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    
    while test_done == 0:
      subdirectories = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
      if subdirectories:
        for directory in subdirectories:
          try:
            ### Model evaluation and prediction
            # Open a strategy scope and create/restore the model
            print("running on", directory)
            with strategy.scope():
              model =keras.models.load_model(directory)
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            print("Test accuracy: {}".format(test_accuracy))
          except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
        test_done = 1
      else:
        print(".", end="", flush=True)    
        time.sleep(1)   
if node_role == "evaluator":
  run_testing()
