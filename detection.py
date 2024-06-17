import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 300
IMG_HEIGHT = 300
NUM_CATEGORIES = 2
TEST_SIZE = 0.3

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python detection.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    # 0 for Normal Oral Cavity
    # 1 for OSCC
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    (pretrained_base, model) = get_model()
    # Fit model on training data
    model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS)
    pretrained_base.trainable = True
    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )  # Very low learning rate

    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split = 0.2)
    history_frame = pd.DataFrame(history.history)
    print(history_frame)
    history_frame.loc[:, ['loss', 'val_loss']].plot().get_figure().savefig('loss_')
    history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot().get_figure().savefig('accuracy_')
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # actual_0 = 0
    # actual_1 = 0
    # predicted_0 = 0
    # predicted_1 = 0
    
    # for actual, predicted in zip(labels, predictions):
    #     if actual:
    #         actual_positive += 1
    #         if predicted:
    #             predicted_positive += 1
    #     else:
    #         actual_negative += 1
    #         if not predicted:
    #             predicted_negative += 1
    # sensitivity = predicted_positive / actual_positive
    # specificity = predicted_negative / actual_negative

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    os.chdir(os.path.join(os.getcwd(), data_dir))
    for label in os.listdir():
        os.chdir(os.path.join(os.getcwd(), label))
        for img in os.listdir():
            image = cv2.imread(os.path.join(os.getcwd(), img))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            labels.append(label)
        os.chdir("..")
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    pretrained_base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        pooling='None',
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )
    pretrained_base.trainable = False
    model = tf.keras.models.Sequential([
        pretrained_base,
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding='same'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding='same'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            512, (3, 3), activation="relu", padding='same'
        ),
        tf.keras.layers.Conv2D(
            1024, (3, 3), activation="relu", padding='same'
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    return (pretrained_base, model)


if __name__ == "__main__":
    main()
