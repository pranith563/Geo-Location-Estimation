import cv2
import glob
import numpy as np
import os
import random

image_size = 256
num_images = 6000
num_channels = 3

#def one_hot_encoding(labels):


def load_data(dataset_dir):
	images = []
	labels = []
	dirs = os.listdir(dataset_dir)

	for dir_id,dir_name in enumerate(dirs):
		path = dataset_dir + "/" + dir_name;
		image_paths = os.listdir(path)
		for image_path in image_paths:
			image = cv2.imread(path+"/"+image_path)
			if image is not None:
				image = cv2.resize(image,(256,256))
				images.append(image)
				labels.append(dir_id)
	images = np.array(images)
	labels = np.array(labels)
	#labels = one_hot_encoding(labels)

	return images,labels

#images,labels = load_data("./dataset")

