import numpy as np
import pandas as pd
import scipy.io as sio

import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from tensorflow.keras import layers, models

EPOCHS = 500
# TRAIN_SPLIT = 200
TEST_SPLIT = 4000


def get_labels(annotations, size):
	labels = np.zeros((size, 5))
	for i in range(size):
		for j in range(5):
			labels[i, j] = annotations['annotations'][0][i][j][0][0]
	labels = pd.DataFrame(labels, columns=('bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2', 'class'))
	return labels


def peek_image(path, index, labels):
	path = '../stanford_data/' + path + '/'
	image_names = os.listdir(path)
	name = str(index)
	for i in range(5 - len(name)):
		name = '0' + name
	name = name + '.jpg'
	im = cv2.imread(path + name)
	h_resize = int(128 * 1.5)
	w_resize = 128
	im = cv2.resize(im, (h_resize, w_resize), interpolation=cv2.INTER_LINEAR)
	print('The label for this car is: ', labels['class'][index - 1])
	plt.imshow(im)


def read_image(path, labels, test=False):
	path = '../stanford_data/' + path + '/'
	image_names = os.listdir(path)
	x = []
	y = []

	if test:
		split = TEST_SPLIT  # 4000
	else:
		split = len(image_names)
	for i in tqdm(range(split)):
		file = image_names[i]
		im = cv2.imread(path + file)
		h_resize = int(128 * 1.5)
		w_resize = 128
		im = cv2.resize(im, (h_resize, w_resize), interpolation=cv2.INTER_LINEAR)
		x.append(im)
		index = int(file.split('.')[0])
		y.append(labels['class'][index - 1])
	return np.array(x), np.array(y)


def create_model(input_shape, output_shape):
	model = models.Sequential()
	model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
	model.add(layers.MaxPool2D(pool_size=(2, 2)))
	model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(layers.MaxPool2D(pool_size=(2, 2)))
	model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(output_shape, activation='softmax'))
	return model


def train_model(model, x_train, y_train, x_test, y_test):
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	history = model.fit(x_train, y_train,
						epochs=EPOCHS,
						validation_data=(x_test, y_test))
	return history


def eval_model(model, history, x_test, y_test):
	plt.plot(history.history['acc'], label='accuracy')
	plt.plot(history.history['val_acc'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')

	test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
	print("Test Loss = ", test_loss)
	print("Test Accuracy = ", test_acc)


def gap():
	print()
	print()
	print()
	print()

annotations = sio.loadmat('../stanford_data/devkit/cars_train_annos.mat')
_, train_size = annotations['annotations'].shape

gap()
print("Loading Images...")
labels = get_labels(annotations, train_size)
x_train, y_train = read_image('cars_train', labels)
x_test, y_test = read_image('cars_test', labels, test=True)

gap()
print("Creating Model...")
input_shape = x_train.shape[1:]
output_shape = y_train.shape[0]
model = create_model(input_shape, output_shape)
model.summary()

gap()
print("Training Model...")
history = train_model(model, x_train, y_train, x_test, y_test)

gap()
print("Evaluating Model...")
eval_model(model, history, x_test, y_test)
