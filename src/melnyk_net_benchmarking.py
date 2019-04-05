import numpy as np
np.random.seed(1996)

import scipy as sp
import h5py
import matplotlib.pyplot as plt 
import pandas as pd 

import keras

from keras import optimizers
from keras.layers import Dense
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

from melnyk_net import melnyk_net
from utils import get_mnist_data, get_fashion_mnist_data


def main(dataset='fashion_mnist'):

	# load the data:
	if dataset == 'fashion_mnist':
		print('\nbenchmarking on fashion_mnist (no preprocessing):')
		print('optimizer: sgd + momentum,  mini-batch size: 64,  initial lr: 0.01,  mu: 0.9,  epochs: 40\n')
		epochs = 40
		(Xtrain, Ytrain), (Xtest, Ytest), label_names = get_fashion_mnist_data(normalize=False, add_label_names=True)
	
	else:
		print('\nbenchmarking on mnist (no preprocessing):')
		print('optimizer: sgd + momentum,  mini-batch size: 64,  initial lr: 0.01,  mu: 0.9,  epochs: 25\n')
		epochs = 25
		(Xtrain, Ytrain), (Xtest, Ytest) = get_mnist_data(normalize=False)

	
	N = Xtrain.shape[0]
	K = len(set(Ytrain.ravel()))

	# plot randomly selected training samples:
	for n in range(3):
		i = np.random.choice(N)
		sample = Xtrain[i]
		plt.imshow(sample[:, :, 0], cmap='gray')
		if dataset == 'fashion_mnist':
			plt.title(label_names[Ytrain[i]])
		else:
			plt.title(Ytrain[i])
		plt.show()
	
	# define our model
	model = melnyk_net(input_shape=Xtrain.shape[1:], num_classes=K)
	
	# model.summary()
	
	# learning rate:
	lr = 0.01

	sgd = optimizers.SGD(lr, momentum=0.9)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	csv_logger = CSVLogger('training_%s.log' % dataset, append=False)
	lr_reducer = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=0, verbose=1, mode='auto', min_delta=0.0001, min_lr=0.000001)

	callbacks_list = [csv_logger, lr_reducer]

	r = model.fit(
			Xtrain, Ytrain, validation_data=(Xtest, Ytest),
			# validation_split=0.2,
			epochs=1,  batch_size=64, shuffle=True, verbose=1, 
			callbacks=callbacks_list,
		)

	score = model.evaluate(Xtest, Ytest, verbose=1)

	model.save('%s-model.h5' % dataset)


	print('Competition loss:', score[0])
	print('Competition accuracy:', score[1])


	# plot and save the losses and accuracies:
	df = pd.read_csv('training_%s.log' % dataset)

	# plot the train and the validation cost:
	plt.plot(df['loss'], label='train_loss')
	plt.plot(df['val_loss'], label='val_loss')
	plt.xlabel('epochs')
	plt.ylabel('cost')
	plt.legend()
	plt.savefig('%s-loss.png' % dataset)
	plt.show()

	plt.gcf().clear() # clear the figure before plotting another

	# plot the train and the validation accuracy:
	plt.plot(df['acc'], label='train_acc')
	plt.plot(df['val_acc'], label='val_acc')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()
	plt.savefig('%s-accuracies.png' % dataset)
	plt.show()




if __name__ == '__main__':
	dataset = input('\nPlease, select a dataset for benchmarking (\'mnist\'/\'fashion_mnist\'):\n').strip()
	main(dataset)

	