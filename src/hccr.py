import sys
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import h5py

from keras.models import Model, load_model
from pyplotz.pyplotz import PyplotZ
from utils import (
	GlobalWeightedAveragePooling2D, 
	tagcode_to_unicode, 
	preprocess_input
	)


class HCCR(object):
	def __init__(
		self,
		model_filepath='Melnyk-Net.hdf5',
		label2tagcode='label2tagcode.npy',
		show_summary=False,
	):
		if not os.path.exists(model_filepath):
			print("\nError: 'Melnyk-Net.hdf5' not found")
			print('Please, donwload the model and place it in the directory of this file.')
			print('URL: https://drive.google.com/open?id=1s8PQo7CKpOGdo-eXwtYeweY8-yjs7RYp')
			input('\npress Enter to exit')			
			exit()

		# load the model architecture:
		self.model = load_model(model_filepath, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})
		self.label2tagcode = np.load(label2tagcode)
		if show_summary:
			print(self.model.summary())


	def predict(self, data, num_predictions=5, verbose=0, plot_cam=False):
		'''
		Returns the prediction for an input image/images
		'''
		imgs = data[0]
		prepr_imgs = data[1]
		num_segments = len(prepr_imgs)
		
		pY = self.model.predict(x=prepr_imgs, verbose=0)		
		
		labels = pY.argsort(axis=1)[:, -num_predictions:][:, ::-1]
		assert num_segments == len(labels)

		if plot_cam:
			heat_maps = []

			Wgwap = self.model.get_layer('global_weighted_average_pooling2d_1').get_weights()[0]
			activation_layer = self.model.get_layer('activation_14')
			model_tranc = Model(inputs=self.model.input, outputs=activation_layer.output)
		
			# get the feature map weights - 
			# 'GWOAP - FC-3755' weight matrix:
			final_dense = self.model.get_layer('dense_1')
			W = final_dense.get_weights()[0] # 448 x 3755

			# get the feature maps:
			fmaps = model_tranc.predict(x=prepr_imgs) # num_segments x 6 x 6 x 448
			fmaps *= Wgwap

			for i in range(num_segments):
				# get the last weight matrix' column for the predicted class:
				w = W[:, labels[i, 0]] # (448 x 1)
				# get the heat_map - class activation map -
				# calculate the dot-product between fmaps and w:
				heat_map = fmaps[i].dot(w) #  (6 x 6)
				
				# resize and save the heat_map:
				heat_map = sp.misc.imresize(heat_map, size=(96, 96), interp='bilinear', mode='F')
				heat_maps.append(heat_map)
						
		
		chars_candidates = np.chararray((num_predictions, num_segments), unicode=True)
		scores = np.empty((num_predictions, num_segments))

		if verbose != 0:
			print('\nMelnyk-Net response: ')

		# NOTE: in the following piece of code i - columns, j - rows:
		for i in range(num_segments):
			predictions = self.label2tagcode[labels[i]].ravel().astype(np.uint16)
			j = 0
			for prediction, p in zip(predictions, pY[i, labels[i]].ravel()):
				prediction = tagcode_to_unicode(prediction)[0]
				chars_candidates[j, i] = prediction
				scores[j, i] = np.round(p, 3) 
				if verbose != 0:
					if j % num_predictions == 0:
						print()				
					print('Dear User, I am %.3f %s sure it\'s     %s     ' % (p*100.0, '%', prediction))
				j += 1
					
			pltz = PyplotZ()
			pltz.enable_chinese()

			if plot_cam:
				plt.subplot(1, 2, 1)

			plt.imshow(imgs[i], cmap='gray')
			pltz.title('prediction: ' + tagcode_to_unicode(predictions[0])[0], fontsize=25, y=1.03)
			pltz.automate_font_size(scale=1.2)

			if plot_cam:
				plt.subplot(1, 2, 2)
				plt.imshow(prepr_imgs[i].reshape(96, 96), cmap='gray', alpha=0.7)
				plt.imshow(heat_maps[i], cmap='jet', alpha=0.6)
				plt.title('class activation map', size=22, y=1.03)

			plt.show()

		
		return chars_candidates, scores

	
