import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from sklearn.decomposition import PCA

# Implementing CNN with PCA on radiography data (1GB data)

def load_radiography_data():
    """
    This method will load radiography data from "COVID-19 Radiography Database" folder

    :return: (train_images, train_labels), (test_images, test_labels)
    """
    # Load all Covid Images
    images = []
    labels = []
    for filename in os.listdir(
            os.path.join("COVID-19 Radiography Database", "COVID-19")):
        img = cv2.imread(
            os.path.join("COVID-19 Radiography Database", "COVID-19", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append("covid")

    count_covid_images = len(images)

    # Load all Normal (non-covid) Images
    for filename in os.listdir(
            os.path.join("COVID-19 Radiography Database", "NORMAL")):
        img = cv2.imread(
            os.path.join("COVID-19 Radiography Database", "NORMAL", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None and count_covid_images > 0:
            images.append(img)
            labels.append("normal")
            count_covid_images = count_covid_images - 1

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True)
    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))


def preprocess_data():
    """
    This method will normalize the data by dividing by 255.
    The normalized values will lie between 0 and 1
    Number of features before applying PCA 1048576 (1024*1024)
    We are trying to reduce to n_components

    :return: train_images, test_images, train_labels, test_labels
    """
    le = preprocessing.LabelEncoder()

    # applying pca and reducing to 512*512
    pca = PCA(n_components=196)

    train_images_reshaped = train_images.reshape(len(train_images), 1024 * 1024)/255.
    test_images_reshaped = test_images.reshape(len(test_images), 1024 * 1024)/255.

    pca_train_images = pca.fit_transform(train_images_reshaped)
    pca_test_images = pca.transform(test_images_reshaped)

    train_images_reshaped = pca_train_images.reshape(len(train_images), 14, 14, 1)
    test_images_reshaped = pca_test_images.reshape(len(test_images), 14, 14, 1)

    # Normalize pixel values to be between 0 and 1
    return train_images_reshaped, test_images_reshaped, le.fit_transform(train_labels), le.fit_transform(test_labels)


def create_model():
    """
    This method will create a tensorflow Sequential model and return the same.
    It will also print the summary of the model.

    :return: model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(14, 14, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model():
    """
    This method will fit the model

    :return: history
    """
    return model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), shuffle='True')


def plot_observations():
    """
    This method will plt all the observations captured by fitting the model

    :return: nothing
    """
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='val_loss ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Test Accuracy:", test_acc)

if __name__ == '__main__':
    """
    Main Method : Execution starts here
    """

    # load data
    (train_images, train_labels), (test_images, test_labels) = load_radiography_data()

    # pre-process data
    train_images, test_images, train_labels, test_labels = preprocess_data()

    # create CNN
    model = create_model()

    # train and retrieve history from model
    history = train_model()

    # plot observation from history
    plot_observations()
