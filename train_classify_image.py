#Classify Image using CNN - training file

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
from sklearn.model_selection import train_test_split # untuk membagi data train dan data test 
from sklearn.metrics import classification_report # performance measurement , untuk menghasilkan confusion matrix

# untuk load data 
from PIL import Image 
from imutils import paths
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt #untuk menampilkan contoh data dalam gambar


# construct the argument parser and parse the arguments
# grab all image paths in the input dataset directory, then initialize
# our list of images and corresponding class labels
print("[INFO] loading images...")
imagePaths = paths.list_images("/content/drive/My Drive/Colab Notebooks/Classify-Image/3scenes")
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, resize it to 32x32 pixels, scale
	# the pixel intensities to the range [0, 1], and then update our
	# images list
	image = Image.open(imagePath)
	image = np.array(image.resize((32, 32))) / 255.0
	data.append(image)
	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	# label = imagePath.title
	labels.append(label)
	# print(label)

# encode the labels, converting them from strings to integers
print(label)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25)

tes_data = testX[90]
plt.imshow(tes_data)
print(trainY)


# define our Convolutional Neural Network architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4))
model.add(Activation("softmax"))

model.summary()

# train the model using the Adam optimizer
print("[INFO] training network...")
opt = Adam(learning_rate=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=10, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

model.save("/content/drive/My Drive/Colab Notebooks/Classify-Image/cnn_classify_image_train.hdf5")
