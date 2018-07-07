import os
import struct
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.misc import imresize
from skimage.filters import threshold_otsu
from skimage import io
from glob import glob

img_shape = (96, 96)

def normalize_bitmap(bitmap):
	# pad the bitmap to make it squared
	pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
	if bitmap.shape[0] < bitmap.shape[1]:
		pad_dims = ((pad_size, pad_size), (0, 0))
	else:
		pad_dims = ((0, 0), (pad_size, pad_size))
	bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

	# rescale and add empty border
	bitmap = imresize(bitmap, (96 - 4*2, 96 - 4*2))
	bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
	assert bitmap.shape == img_shape

	bitmap = np.expand_dims(bitmap, axis=0)
	assert bitmap.shape == (1, *img_shape)
	return bitmap



def preprocess_bitmap(bitmap):
	# reverse the gray values to ensure the fast computation on the training step:
	bitmap = 255 - bitmap

	# contrast stretching
	p2, p98 = np.percentile(bitmap, (2, 98))
	bitmap = skimage.exposure.rescale_intensity(bitmap, in_range=(p2, p98))
										
	return bitmap


def tagcode_to_unicode(tagcode):
	# print(struct.pack('>H', tagcode).decode('gbk'))
	return struct.pack('>H', tagcode).decode('gbk')


def unicode_to_tagcode(tagcode_unicode):
	return struct.unpack('>H', tagcode_unicode.encode('gbk'))[0]


def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

	
def preprocess_input(img, show_img=False):
		'''
		Preprocesses an input image for futrther prediction
		'''             
				
		if show_img:
			print('\nOriginal image shape:', img.shape)
			plt.imshow(img)
			plt.title('original')
			plt.show()

		gray_img = rgb2gray(img)
		thresh = threshold_otsu(gray_img)
		# binarize to get a 'white' (255) background:
		if np.mean(gray_img > thresh) > np.mean(gray_img < thresh):
			gray_img[gray_img > thresh] = 255
		else: 
			gray_img[gray_img < thresh] = 0
			gray_img = 255 - gray_img 

		# if show_img:
		# 	print('Grayscale image shape:', gray_img.shape)
		# 	plt.imshow(gray_img, cmap='gray')
		# 	plt.title('grayscaled')
		# 	plt.show()

		norm_img = normalize_bitmap(np.array(gray_img, dtype=np.uint8))
		prepr_img = np.array(preprocess_bitmap(norm_img), dtype=np.uint8).reshape(1, *img_shape, 1)
		if show_img:
			# print('Normalized image shape:', norm_img.shape)
			# plt.imshow(norm_img.reshape(*img_shape), cmap='gray')
			# plt.title('normalized')
			# plt.show()
			print('Preprocessed image shape:', prepr_img.shape)
			plt.imshow(prepr_img.reshape(*img_shape), cmap='gray')
			plt.title('preprocessed')
			# plt.savefig('prepr_img.png')
			plt.show()

		return prepr_img


def load_data(PATH_TO_TEST_IMAGES_DIR):
	print('\nLoading the data.....................................')
	files = glob(PATH_TO_TEST_IMAGES_DIR + '/*.png') # for .png
	files += glob(PATH_TO_TEST_IMAGES_DIR + '/*.jp*g')	# append for .jpg and .jped
	n_files = len(files)
	print('Found %d images in the specified directory\n' % n_files)
	# print(files)
	
	prepr_imgs = np.empty((n_files, 96, 96, 1))
	imgs = []
	for i, img_filepath in enumerate(files):
		img = io.imread(img_filepath)
		imgs.append(img)
		prepr_img = preprocess_input(img, show_img=False)
		prepr_imgs[i] = prepr_img
	
	return imgs, prepr_imgs
