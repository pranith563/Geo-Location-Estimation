"""
:description: Utilities for loading in, preprocessing, and organizing data.
"""
import cv2
import glob
import numpy as np
import os
import random
import scipy.io
from mpl_toolkits.basemap import Basemap
import warnings
from numpy import genfromtxt
from sklearn.cluster import MiniBatchKMeans
from sklearn import cluster

output_cluster = '../clusters/clusters.npy'
def cluster(file=None, output=None, n_clusters=None):
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	
	if n_clusters is None: n_clusters = 100
	
	lat_long = genfromtxt(file, delimiter=',')
	c = np.delete(lat_long, 2,1)
	

	# Create the basemap parameters
	bnd = 0
	basemap_params = dict(projection='merc',llcrnrlat=np.min(c[:,0])-bnd,urcrnrlat=np.max(c[:,0])+bnd, llcrnrlon=np.min(c[:,1])-bnd,urcrnrlon=np.max(c[:,1])+bnd)
	
	# Select a subset of the coordinates to cluster
	'''if max_files is None:
		max_files = 10000
	np.random.shuffle(c)
	c = c[:max_files]'''
	
	# Project the coordinates into x, y coordinates
	m = Basemap(**basemap_params)
	x,y = m(c[:,1],c[:,0])

	km = MiniBatchKMeans(n_clusters=n_clusters).fit(np.concatenate((x[:,None],y[:,None]),axis=1))
	#print(km.cluster_centers_)
	np.save(output,(basemap_params,km.cluster_centers_))

def getNCluster(cluster_file):
	bm_param, km_param = np.load(cluster_file)
	return km_param.shape[0]


def coordToCluster(lat,lon,cluster_file):
	import tensorflow as tf
	# Setup the basemap and cluster
	from mpl_toolkits.basemap import Basemap
	from sklearn import cluster
	bm_param, km_param = np.load(cluster_file)
	m = Basemap(**bm_param)
	km = cluster.MiniBatchKMeans(n_clusters=km_param.shape[0])
	km.cluster_centers_ = km_param
	# Cluster function
	#def _cluster(lt,ln):
	return np.int32(km.predict(np.array(m(lon,lat))[None])[0])
	#r = tf.py_func(_cluster, [lat,lon], [tf.int32])[0]
	#r.set_shape(())
	#return r

if __name__ == "__main__":
	#cluster(n_clusters=16)
	print(coordToCluster(40.757292,-72.989946,output_cluster))

