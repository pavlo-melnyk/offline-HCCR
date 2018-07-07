import numpy as np 
import os 
import matplotlib.pyplot as plt 

from hccr import HCCR
from utils import load_data


def main():
	
	hccr = HCCR()

	print()	

	while True:
		data_filepath = input('Please specify data filepath:\n')
		data = load_data(data_filepath)
		num_predictions = int(input('Please specify number of candidates per segment:\n'))
		chars_candidates, scores = hccr.predict(data, num_predictions=num_predictions, verbose=1)
				
		prompt = input('\nPress "y" to continue...\n')
		if prompt not in ['y', 'Y']:
			break

	print('\nThanks for using MELNYK-Net HCCR software!')


if __name__ == '__main__':
	main()