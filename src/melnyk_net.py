		import os

import numpy as np

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.initializers import RandomNormal

from utils import GlobalWeightedAveragePooling2D, GlobalWeightedOutputAveragePooling2D


MODEL_FILEPATH = 'Melnyk-Net.hdf5'

def melnyk_net(input_shape=(96, 96, 1), reg=1e-3, global_average_type='GWAP', use_pretrained_weights=False, num_classes=3755, include_top=True):	
	if global_average_type == 'GWAP':
		GlobalAveragePooling = GlobalWeightedAveragePooling2D(kernel_initializer='ones')
	elif global_average_type == 'GWOAP':
		GlobalAveragePooling = GlobalWeightedOutputAveragePooling2D(kernel_initializer='ones')
	else:
		GlobalAveragePooling = GlobalAveragePooling2D()

	if use_pretrained_weights:
		if global_average_type == 'GWAP':
			if not os.path.exists(MODEL_FILEPATH):
				print("\nError: 'Melnyk-Net.hdf5' not found")
				print('Please, donwload the model and place it in the current.')
				print('URL: https://drive.google.com/open?id=1s8PQo7CKpOGdo-eXwtYeweY8-yjs7RYp')
				input('\npress Enter to exit')			
				exit()

			model = load_model(MODEL_FILEPATH, custom_objects={"GlobalWeightedAveragePooling2D": GlobalWeightedAveragePooling2D})

			if include_top:
				return model
			else:
				dropout = model.get_layer('dropout_1')
				model_tranc = Model(inputs=model.input, outputs=dropout.output)

				return model_tranc

		else:
			print('pretrained weights available only for melnyk-net with gwap')
			exit()

	random_normal = RandomNormal(stddev=0.001, seed=1996) # output layer initializer

	input_ = Input(shape=input_shape)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(input_)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
			kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)


	x = GlobalAveragePooling(x)
	x = Dropout(0.5)(x)

	if include_top:
		x = Dense(units=num_classes, kernel_initializer=random_normal, activation='softmax', 
				  kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)

	model = Model(inputs=input_, outputs=x)

	return model



if __name__ == '__main__':
	model = melnyk_net(use_pretrained_weights=True, include_top=True)
	model.summary()