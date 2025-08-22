''' 
The functions normalize_bitmap, tagcode_to_unicode and unicode_to_tagcode
are written by Alessandro and Francesco https://github.com/integeruser/CASIA-HWDB1.1-cnn
'''

import os
import struct

import numpy as np
import scipy as sp 
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import h5py

from scipy.io import loadmat 

from skimage.filters import threshold_otsu

from PIL import Image

from glob import glob

from keras.layers import (
	Input,
	Activation,
	Dense,
	Flatten,
	GlobalAveragePooling2D
)
from keras import backend as K
from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import mnist, fashion_mnist, cifar10, cifar100

# compatibility fix for older code:
if not hasattr(np.lib, 'pad'):
    np.lib.pad = np.pad



IMG_SHAPE = (96, 96)


class GlobalWeightedAveragePooling2D(GlobalAveragePooling2D):

	def __init__(self, kernel_initializer='uniform', **kwargs):
		self.kernel_initializer = kernel_initializer
		super(GlobalWeightedAveragePooling2D, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name='W',
								 shape=input_shape[1:],
								 initializer=self.kernel_initializer,
								 trainable=True)
		# print('input_shape:', input_shape)
		super(GlobalWeightedAveragePooling2D, self).build(input_shape)

	def call(self, inputs):
		inputs = inputs*self.W # element-wise multiplication for every entry of input
		if self.data_format == 'channels_last':
			return tf.reduce_sum(inputs, axis=[1, 2]) # K.sum(inputs, axis=[1, 2])
		else:
			return tf.reduce_sum(inputs, axis=[1, 2]) # K.sum(inputs, axis=[2, 3])


class GlobalWeightedOutputAveragePooling2D(GlobalAveragePooling2D):

	def __init__(self, kernel_initializer='uniform', **kwargs):
		self.kernel_initializer = kernel_initializer
		super(GlobalWeightedOutputAveragePooling2D, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.data_format == 'channels_last':
			kernel_shape = [input_shape[-1]]
		else:
			kernel_shape = [input_shape[1]]

		self.W = self.add_weight(name='W',
								 shape=kernel_shape,
								 initializer=self.kernel_initializer,
								 trainable=True)
		# print('input_shape:', input_shape)
		super(GlobalWeightedOutputAveragePooling2D, self).build(input_shape)

	def call(self, inputs):
		inputs = inputs*self.W # element-wise multiplication for every entry of input
		if self.data_format == 'channels_last':
			return tf.reduce_sum(inputs, axis=[1, 2]) # K.sum(inputs, axis=[1, 2])
		else:
			return tf.reduce_sum(inputs, axis=[1, 2]) # K.sum(inputs, axis=[2, 3])


def normalize_bitmap(bitmap):
	# pad the bitmap to make it squared
	pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
	if bitmap.shape[0] < bitmap.shape[1]:
		pad_dims = ((pad_size, pad_size), (0, 0))
	else:
		pad_dims = ((0, 0), (pad_size, pad_size))
	bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

	# rescale and add empty border
	# bitmap = sp.misc.imresize(bitmap, (96 - 4*2, 96 - 4*2))	
	bitmap = imresize(bitmap, (96 - 4*2, 96 - 4*2))	
	bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
	assert bitmap.shape == IMG_SHAPE

	bitmap = np.expand_dims(bitmap, axis=0)
	assert bitmap.shape == (1, *IMG_SHAPE)
	return bitmap


def preprocess_bitmap(bitmap):
	# inverse the gray values:
	bitmap = 255 - bitmap
	return bitmap


def tagcode_to_unicode(tagcode):
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

	
		norm_img = normalize_bitmap(np.array(gray_img, dtype=np.uint8))
		prepr_img = np.array(preprocess_bitmap(norm_img), dtype=np.uint8).reshape(1, *IMG_SHAPE, 1)
		if show_img:
			
			print('Preprocessed image shape:', prepr_img.shape)
			plt.imshow(prepr_img.reshape(*IMG_SHAPE), cmap='gray')
			plt.title('preprocessed')
			plt.show()

		return prepr_img


def load_data(image_path=None):
	print('\nLoading the data.....................................')
	if image_path and image_path[-4:].lower() in ['.png', '.jpg', 'jpeg']:
		img = np.array(image.load_img(image_path))
		prepr_img = preprocess_input(img, show_img=False)
		print()

		return [img], prepr_img
		
	elif image_path:
		files = glob(image_path + '/*.png') # for .png
		files += glob(image_path + '/*.jp*g') # append for .jpg and .jped
		n_files = len(files)
		print('Found %d images in the specified directory\n' % n_files)
		# print(files)
				
		prepr_imgs = np.empty((n_files, 96, 96, 1))
		imgs = []
		for i, img_filepath in enumerate(files):
			img = np.array(image.load_img(img_filepath))
			imgs.append(img)
			prepr_img = preprocess_input(img, show_img=False)
			prepr_imgs[i] = prepr_img	

		return imgs, prepr_imgs


def get_mnist_data(reshape=False, normalize=False):
	# load the MNIST data:
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()	
	Xtrain = np.expand_dims(Xtrain, axis=3)
	Xtest = np.expand_dims(Xtest, axis=3)
	
	if reshape:
		N, K = len(Ytrain), len(set(Ytrain))
		D = Xtrain.shape[1]*Xtrain.shape[2]
		# reshape the data to be (NxD):
		Xtrain, Xtest = Xtrain.reshape(N, D), Xtest.reshape(len(Xtest), D)

	if normalize:
		Xtrain = np.float32(Xtrain / 255.0)
		Xtest = np.float32(Xtest / 255.0)

	return (Xtrain, Ytrain), (Xtest, Ytest)


def get_fashion_mnist_data(reshape=False, normalize=False, add_label_names=False):
	# load the Fashion-MNIST data:
	(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()	
	Xtrain = np.expand_dims(Xtrain, axis=3)
	Xtest = np.expand_dims(Xtest, axis=3)
	
	if reshape:
		N, K = len(Ytrain), len(set(Ytrain))
		D = Xtrain.shape[1]*Xtrain.shape[2]
		# reshape the data to be (NxD):
		Xtrain, Xtest = Xtrain.reshape(N, D), Xtest.reshape(len(Xtest), D)

	if normalize:
		Xtrain = np.float32(Xtrain / 255.0)
		Xtest = np.float32(Xtest / 255.0)

	if add_label_names:
		label_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
		return (Xtrain, Ytrain), (Xtest, Ytest), label_names

	return (Xtrain, Ytrain), (Xtest, Ytest)


def imresize(arr, size, interp='bilinear', mode=None):
    """
    Exact replacement for scipy.misc.imresize with identical behavior.
    
    Parameters:
    -----------
    arr : array_like
        Input array (image)
    size : tuple or float
        Size of output image as (height, width) or scaling factor
    interp : str
        Interpolation method ('bilinear', 'nearest', 'cubic', etc.)
    mode : str
        PIL mode for handling data types ('F' for float, 'L' for grayscale, etc.)
    
    Returns:
    --------
    resized : ndarray
        Resized array with same dtype as input
    """
    
    # handle size parameter
    if np.isscalar(size):
        # if size is a scalar, it's a scaling factor:
        if arr.ndim == 2:
            new_size = (int(arr.shape[0] * size), int(arr.shape[1] * size))
        else:
            new_size = (int(arr.shape[0] * size), int(arr.shape[1] * size))
    else:
        # size is already (height, width) - scipy format:
        new_size = tuple(size)
    
    # Store original dtype
    original_dtype = arr.dtype
    
    # handle interpolation mapping:
    interp_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'cubic': Image.BICUBIC,  # scipy used cubic as alias for bicubic
        'lanczos': Image.LANCZOS
    }
    
    pil_interp = interp_map.get(interp.lower(), Image.BILINEAR)
    
    # auto-detect mode if not specified (scipy behavior):
    if mode is None:
        if arr.dtype in [np.float32, np.float64]:
            mode = 'F'  # float mode
        elif arr.dtype == np.uint8:
            if arr.ndim == 2:
                mode = 'L'  # grayscale
            elif arr.ndim == 3 and arr.shape[2] == 3:
                mode = 'RGB'
            elif arr.ndim == 3 and arr.shape[2] == 4:
                mode = 'RGBA'
            else:
                mode = 'L'
        else:
            # convert to float for other types:
            arr = arr.astype(np.float32)
            mode = 'F'
    
    # handle different array dimensions
    if arr.ndim == 2:
        # 2D array (grayscale)
        if mode == 'F':
            # for float mode, ensure proper range
            img = Image.fromarray(arr.astype(np.float32), mode='F')
        else:
            img = Image.fromarray(arr, mode=mode)
        
        # PIL uses (width, height), scipy used (height, width)
        resized_img = img.resize((new_size[1], new_size[0]), pil_interp)
        result = np.array(resized_img)
        
    elif arr.ndim == 3:
        # 3D array (color or multi-channel):
        if arr.shape[2] <= 4:  # standard RGB/RGBA
            if mode == 'F':
                # for float arrays, handle each channel separately to maintain precision:
                channels = []
                for i in range(arr.shape[2]):
                    channel = arr[:, :, i].astype(np.float32)
                    img = Image.fromarray(channel, mode='F')
                    resized = img.resize((new_size[1], new_size[0]), pil_interp)
                    channels.append(np.array(resized))
                result = np.stack(channels, axis=2)
            else:
                img = Image.fromarray(arr, mode=mode)
                resized_img = img.resize((new_size[1], new_size[0]), pil_interp)
                result = np.array(resized_img)
        else:
            # multi-channel, handle each channel separately:
            channels = []
            for i in range(arr.shape[2]):
                channel = arr[:, :, i]
                if mode == 'F' or arr.dtype in [np.float32, np.float64]:
                    img = Image.fromarray(channel.astype(np.float32), mode='F')
                else:
                    img = Image.fromarray(channel, mode='L')
                resized = img.resize((new_size[1], new_size[0]), pil_interp)
                channels.append(np.array(resized))
            result = np.stack(channels, axis=2)
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    
    # preserve original dtype:
    if original_dtype != result.dtype:
        if original_dtype in [np.uint8, np.int32, np.int64]:
            result = np.round(result).astype(original_dtype)
        else:
            result = result.astype(original_dtype)
    
    return result