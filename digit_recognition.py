import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class Recognizer:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        # for i in range(self.x_train.shape[0]):
        #     print(self.x_train[i])
        #     plt.imshow(self.x_train[i])
        #     plt.show()
        #     plt.cla()
        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)
        self.model = None

    def train_data(self):
        print("TRAINING THE DATA")
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(28,28)))
        self.model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        self.model.add(tf.keras.layers.Dense(250,activation="relu"))
        self.model.add(tf.keras.layers.Dense(100,activation="relu"))
        self.model.add(tf.keras.layers.Dense(10,activation="softmax"))

        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=10)

        print("TRAINING IS DONE")

    def test_data(self):
        print("TESTING THE DATA")

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

        print("TESTING IS DONE")

    def save_model(self, save):
        if save == "n":
            print("Module not saved")
        else:
            self.model.save("digit_detection.model")
            print("Module saved")

    def load_model(self):
        folder_path = "digit_detection.model"
        self.model = tf.keras.models.load_model(folder_path)

    def detect_digit(self, im):
        prediction = self.model.predict(im[np.newaxis, :, :])
        return np.argmax(prediction)

    def test_png(self):
        image_number = 0
        digit_path = self.path + "/src/digits"
        while os.path.isfile(f"{digit_path}/{image_number}.png"):
            img = cv2.imread(f"{digit_path}/{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = self.model.predict(img)
            print(f"The digit detected is: {np.argmax(prediction)}")

            image_number += 1

