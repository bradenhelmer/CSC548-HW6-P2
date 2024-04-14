# bthelmer Braden T Helmer
## Stage 1: Installing dependencies and notebook gpu setup
import os
from sys import argv
import tensorflow as tf
import keras
import time

# from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

print(tf.__version__, keras.__version__)
# for RTX GPUs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Get input argument from command line.
EVALUATOR = -1
if not len(argv) > 1:
    print("Please provide input for evaluator (-1) or worker (0..)")
    exit(1)

Role = argv[1]

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"

if Role == "-1":
    node_role = "evaluator"
elif Role.isdigit():
    node_role = "worker"
    os.system(f"rm -rf {checkpoint_dir}/*")
else:
    raise ValueError("Invalid node role specified")

os.environ["KERAS_BACKEND"] = "tensorflow"


os.environ["NCCL_P2P_DISABLE"] = "0"
# Commented out IPython magic to ensure Python compatibility.
# get a local copy of datasets
os.system("ln -s /mnt/beegfs/fmuelle/.keras ~/")

## Stage 3: Dataset preprocessing
### Loading the Cifar10 dataset
# Setting class names for the dataset
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_dataset():
    # Loading the dataset
    # moved to get_dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    ### Image normalization
    X_train = X_train / 255.0
    X_train.shape
    X_test = X_test / 255.0

    return (X_train, y_train), (X_test, y_test)


## Stage 4: Building a Convolutional neural network
def get_compiled_model():
    ### Defining the model
    model = tf.keras.models.Sequential()

    ### Adding the first CNN Layer
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=[32, 32, 3],
        )
    )

    ### Adding the second CNN Layer and max pool layer
    model.add(
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

    ### Adding the third CNN Layer
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu"
        )
    )

    ###  Adding the fourth CNN Layer and max pool layer
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

    ### Adding the Flatten layer
    model.add(tf.keras.layers.Flatten())

    ### Adding the first Dense layer
    model.add(tf.keras.layers.Dense(units=128, activation="relu"))

    ### Adding the second Dense layer (output layer)
    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

    model.summary()

    ### Compiling the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="Adam",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


## HOMEWORK SOLUTION

MAX_EPOCH = 15


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    print(checkpoints)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    if node_role == "worker":
        print("Creating a new model")
        return get_compiled_model()
    else:
        print("No new checkpoint waiting...")


# Execution start

if node_role == "worker":
    print("DOING WORKER NODE")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # moved to get_dataset
    (X_train, y_train), (X_test, y_test) = get_dataset()

    with strategy.scope():
        # get the compiled model
        model = make_or_restore_model()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )

        ### Training the model
        model.fit(X_train, y_train, epochs=MAX_EPOCH, callbacks=[checkpoint_callback])

    print(f"Ending node code")

elif node_role == "evaluator":
    print(f"DOING EVALUATOR NODE")
    strategy = tf.distribute.MirroredStrategy()
    CKPT_DIR = checkpoint_dir
    (X_train, y_train), (X_test, y_test) = get_dataset()

    for epoch in range(1, MAX_EPOCH + 1):

        # Ensure checkpoint path exists
        while not os.path.exists(checkpoint_dir):
            print(".", end="", flush=True)
            time.sleep(1)

        if epoch != MAX_EPOCH:
            curr_ckpt, next_ckpt = f"ckpt-{epoch}", f"ckpt-{epoch+1}"
            while next_ckpt not in os.listdir(checkpoint_dir):
                print(".", end="", flush=True)
                time.sleep(1)
        else:
            curr_ckpt = f"ckpt-{epoch}"

        try:
            ### Model evaluation and prediction
            with strategy.scope():
                model = keras.models.load_model(os.path.join(checkpoint_dir, curr_ckpt))
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                print(f"Test accuracy for epoch {epoch}: {test_accuracy}")
                if epoch == MAX_EPOCH:
                    exit(0)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
