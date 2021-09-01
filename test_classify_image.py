#Classify Image using CNN - testing file

# libary untuk arsitekture CNN
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam


from sklearn.preprocessing import LabelBinarizer # untuk binarisasi label 
# from sklearn.model_selection import train_test_split # untuk membagi data train dan data test 
# from sklearn.metrics import classification_report # performance measurement , untuk menghasilkan confusion matrix

# untuk load data 
from PIL import Image 
from imutils import paths
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt #untuk menampilkan contoh data dalam gambar
from google.colab.patches import cv2_imshow
import cv2
# init_notebook_mode(connected=False)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gambar", type=str, default="",
	help="input data test")
args = vars(ap.parse_args())

# image = Image.open("/content/drive/My Drive/Colab Notebooks/Classify-Image/forest_test.jpg")
image = Image.open(args["gambar"])


image = np.array(image.resize((32, 32))) / 255.0


labels = ["coast", "forest", "highway", "room"]
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


# define our Convolutional Neural Network architecture
model_test = Sequential()
model_test.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model_test.add(Activation("relu"))
model_test.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_test.add(Conv2D(16, (3, 3), padding="same"))
model_test.add(Activation("relu"))
model_test.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_test.add(Conv2D(32, (3, 3), padding="same"))
model_test.add(Activation("relu"))
model_test.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_test.add(Flatten())
model_test.add(Dense(4))
model_test.add(Activation("softmax"))

model_test.load_weights("/content/drive/My Drive/Colab Notebooks/Classify-Image/cnn_classify_image_train.hdf5")
image2 = np.expand_dims(image, axis=0)
pred = model_test.predict(image2)
print(pred)
y_classes = pred.argmax(axis=-1)
print(y_classes)
print('Gambar '+ str(args["gambar"]) + ' diklasifikasikan sebagai '+ str(lb.classes_[y_classes][0]))
# plt.imshow(image)
# plt.show()
# cv2_imshow(image)
