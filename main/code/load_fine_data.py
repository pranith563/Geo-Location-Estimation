import cv2
import glob
import numpy as np
import os
import random
import scipy.io
import warnings
import re
from numpy import genfromtxt
from sklearn.cluster import MiniBatchKMeans
from sklearn import cluster
from street_data_utils import *

DATA_DIR = './data/images'

COORDINATES_FILENAME = './coordinates/GPS_Compass.csv'

output_cluster = './clusters/clusters.npy'

def load_data(dataset_dir=None,cluster_file=None,coords_file=None):
	images=[]
	labels=[]
	img_paths = os.listdir(dataset_dir)
	
	
	bm_param, km_param = np.load(cluster_file)
	m = Basemap(**bm_param)
	km = MiniBatchKMeans(n_clusters=km_param.shape[0])
	km.cluster_centers_ = km_param
	

	lat_long = genfromtxt(coords_file, delimiter=',')

	for img_path in img_paths:
		img = cv2.imread(dataset_dir+"/"+img_path)
		if img is not None:
			images.append(img)
			image_idx = int(img_path[:-6])
			idx = image_idx-4403
			lat = lat_long[idx][0]
			lon = lat_long[idx][1]
			label = np.int32(km.predict(np.array(m(lon,lat))[None])[0])
			labels.append(label)
	images = np.array(images)
	labels = np.array(labels)

	return images,labels