import mercantile
from numpy import genfromtxt
import numpy as np
import os
import cv2

zoom = 14.59

lat_long = genfromtxt('coordinates/GPS_Compass.csv',delimiter=',')
lat_long = np.array(lat_long)
print(lat_long.shape)


tiles= {}
for i in range(len(lat_long)):
	lat_ = lat_long[i][0]
	long_ = lat_long[i][1]
	tile = mercantile.tile(long_,lat_,zoom)
	if(tile in tiles):
		tiles[tile] += 1
	else:
		tiles[tile] = 1

#print(tiles)

sum = 0
for i,j in enumerate(tiles):
	sum += tiles[j]
	print(tiles[j],end=" ")

print()

print(len(set(tiles)))

zoom = 14.59
'''
def load_data(dataset_dir,zoom):
	images = []
	labels = []
	dirs = os.listdir(dataset_dir)
	labels_ids = {}

	for dir_id,dir_name in enumerate(dirs):
		path = dataset_dir+"/"+dir_name
		image_paths = os.listdir(path)
		for image_path in image_paths:
			image = cv2.imread(dataset_dir+"/"+image_path)
			images.append(image)
			idx = int(image_path[:-6]) - 4403
			lat = lat_long[idx][0]
			longt = lat_long[idx][1]
			tile = mercantile.tile(longt,lat,zoom)
			if tile in labels_ids:
				label = labels_ids[tile]
				labels.append(label)
			else:
				labels_ids[tile] = len(labels_ids)
				label = labels_ids[tile]
				labels.append(label)
				
	return np.array(images),np.array(labels)


images,labels = load_data("/home/sravya/Data/part1",zoom)

import pickle

file = open('2_X_Data.pckl','wb')
pickle.dump(images,file)
file.close()

file = open('2_y_Data.pckl','wb')
pickle.dump(labels,file)
file.close()
'''