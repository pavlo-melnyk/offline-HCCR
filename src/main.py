import numpy as np 
import os 
import matplotlib.pyplot as plt 

from hccr import HCCR
from utils import load_data


def main():
	print('\n---------- Offline Handwritten Chinese Character Recognizer ----------\n')
	print('Based on the convolutional neural network called Melnyk-Net.')
	print('Recognizes 3755 Chinese character classes from GB2312-80 level-1.')
	print('\nCopyright © 2018 Pavlo O. Melnyk\n\n')

	show_summary = False
	prompt = input('Show Melnyk-Net summary? [Y/n]\n')
	if prompt in ['Y', 'y']:
		show_summary = True


	hccr = HCCR('Melnyk-Net.hdf5', show_summary=show_summary)
		
	while True:		
		while True:
			data_filepath = input('\nPlease specify data filepath:\n')
			try:
				data = load_data(data_filepath)
				if len(data[0]) == 0:
					print('Ooops... Image(s) not found :(')
					print('Try again')
					continue 
				break
			except Exception as e:
				print('Ooops... An exception occured:\n', e)
				print('Try again')

		while True:
			try:
				num_predictions = int(input('\nPlease specify number of predictions per input image:\n'))
				break
			except Exception as e:
				print('Ooops... An exception occured:\n', e)
				print('Try again')

		chars_candidates, scores = hccr.predict(data, num_predictions=num_predictions, plot_cam=True, verbose=1)
		
		prompt = input('\nContinue? [Y/n]\n')
		if prompt in ['N', 'n']:
			break

	print('\nThanks for using Melnyk-Net Offline HCCR software!')

	
	
if __name__ == '__main__':
	main()