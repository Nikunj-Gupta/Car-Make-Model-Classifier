import numpy as np
import pandas as pd
import scipy.io as sio

import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from keras import layers, models

EPOCHS = 5
# TRAIN_SPLIT = 200
TEST_SPLIT = 4000


def get_labels(annotations_temp, size):
	labels_temp = np.zeros((size, 5))
	for i in range(size):
		for j in range(5):
			labels_temp[i, j] = annotations_temp['annotations'][0][i][j][0][0]
	labels_temp = pd.DataFrame(labels_temp, columns=('bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2', 'class'))
	return labels_temp


def peek_image(path, index, labels_temp):
	path = '../stanford_data/' + path + '/'
	name = str(index)
	for i in range(5 - len(name)):
		name = '0' + name
	name = name + '.jpg'
	im = cv2.imread(path + name)
	h_resize = int(128 * 1.5)
	w_resize = 128
	im = cv2.resize(im, (h_resize, w_resize), interpolation=cv2.INTER_LINEAR)
	print('The label for this car is: ', labels_temp['class'][index - 1])
	plt.imshow(im)


def read_image(path, labels_temp, test=False):
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
		y.append(labels_temp['class'][index - 1])
	return np.array(x), np.array(y)


def create_model(input_shape_temp, output_shape_temp):
	model_temp = models.Sequential()
	model_temp.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape_temp))
	model_temp.add(layers.MaxPool2D(pool_size=(2, 2)))
	model_temp.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model_temp.add(layers.MaxPool2D(pool_size=(2, 2)))
	model_temp.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
	model_temp.add(layers.Flatten())
	model_temp.add(layers.Dense(64, activation='relu'))
	model_temp.add(layers.Dense(output_shape_temp, activation='softmax'))
	return model_temp


def train_model(model_temp, x_train_temp, y_train_temp, x_test_temp, y_test_temp):
	model_temp.compile(optimizer='adam',
					   loss='sparse_categorical_crossentropy',
					   metrics=['accuracy'])

	history_temp = model_temp.fit(x_train_temp, y_train_temp,
								  epochs=EPOCHS,
								  validation_data=(x_test_temp, y_test_temp))
	return history_temp


def eval_model(model_temp, history_temp, x_test_temp, y_test_temp):
	plt.plot(history_temp.history['acc'], label='accuracy')
	plt.plot(history_temp.history['val_acc'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')

	test_loss, test_acc = model_temp.evaluate(x_test_temp, y_test_temp, verbose=2)
	print("Test Loss = ", test_loss)
	print("Test Accuracy = ", test_acc)

def resnet_model(inp_shape, out_shape):
	return


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
