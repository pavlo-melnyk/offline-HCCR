import os
import numpy as np
import sys
import matplotlib.pyplot as plt 
import h5py

from keras.models import load_model
from pyplotz.pyplotz import PyplotZ
from skimage import io
from utils import tagcode_to_unicode, preprocess_input



class HCCR(object):
	def __init__(self, model_filepath='model.h5', display_img=False):
		# load the model architecture:
		
		if not os.path.exists(model_filepath):
			print("\nError: 'model.h5' not found")
			print('Please, donwload the model and place it in the directory of this file.')
			print('URL: https://drive.google.com/open?id=12UVBrGixJLmg6er1bsLC52rWIlSv00Fs')
			input('\npress Enter to exit')			
			exit()

		self.model = load_model(model_filepath)

		# load the 'dictionary' for prediction:
		self.label2tagcode = np.load('label2tagcode.npy')
	
		


	def predict(self, data, num_predictions=5, verbose=0):
		'''
		Returns the prediction for an input image/ images
		'''
		# print(data.shape)
		# print(type(data))
		imgs = data[0]
		prepr_imgs = data[1]
		num_segments = len(prepr_imgs)
		
		pY = self.model.predict(x=prepr_imgs, verbose=0)		
		
		labels = pY.argsort(axis=1)[:, -num_predictions:][:, ::-1]
		assert num_segments == len(labels)
		# print(labels.shape)

		chars_candidates = np.chararray((num_predictions, num_segments), unicode=True)
		scores = np.empty((num_predictions, num_segments))

		if verbose != 0:
			print('\nMELNYK-Net response: ')

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
							print('\n prediction for image #%d' % (i+1))				
					print('Dear User, I am %.3f %s sure you\'ve written     %s     ' % (p*100.0, '%', prediction))
				j += 1					
			
			pltz = PyplotZ() # matplotlib.pyplot may not display Chinese characters correctly
			pltz.enable_chinese()
			plt.imshow(imgs[i], cmap='gray')
			pltz.title('prediction: ' + tagcode_to_unicode(predictions[0])[0], fontsize=22)
			pltz.automate_font_size(scale=1)
			# plt.title('prediction: ' + tagcode_to_unicode(predictions[0]), fontsize=22)
			plt.show()

		return chars_candidates, scores

	
