# Offline Handwritten Chinese Character Classifier
## Description:

- recognizes isolated handwritten Chinese characters among 3755 classes of GB2312-80 level-1 standard;

- based on a CNN called Melnyk-Net [[Melnyk et al, “A High-Performance CNN Method for Offline Handwritten Chinese Character Recognition and Visualization”, 2018]](https://arxiv.org/abs/1812.11489);

- implemented using the amazing Keras library with the TensorFlow backend: 
![Architecture](architecture.png)


- Trained on CASIA-HWDB1.0-1.1 datasets collected by National Laboratory of Pattern Recognition (NLPR), Institute of Automation of Chinese Academy of Sciences (CASIA), written by 420 and 300 persons. The overall training dataset contains 2,678,424 samples.

- Evaluated on the most common benchmark – ICDAR-2013 competition dataset containing 224,419 samples written by 60 persons.

- **Model accuracy** - 97.61%.

- **Total number of parameters** - 6,523,819 (~24.9 MB of storage).

## Dependencies:
Python3, tensorflow 1.7, keras 2.1.6, scipy, numpy, matplotlib, pyplotz, glob, struct, h5py

## Usage:
 - before using the application, please download [Melnyk-Net](https://drive.google.com/open?id=1s8PQo7CKpOGdo-eXwtYeweY8-yjs7RYp) and place it in the directory with the other files (don't rename it);
 - run 'main.py';
 - follow the prompt instructions:
    - specify the isolated character image/images directory (e.g. 'images'; it will upload all the images from this folder);
       supports both b&w and colorful images;
    - specify the number of candidates for recognition, *n*, - the program will show *n* most confident predictions per each character image.
    

## Demo:
![Demo](demo.png)

```
$ main.py

---------- Offline Handwritten Chinese Character Recognizer ----------

Based on the convolutional neural network called Melnyk-Net.
Recognizes 3755 Chinese character classes from GB2312-80 level-1.

Copyright © 2018 Pavlo O. Melnyk


Show Melnyk-Net summary? [Y/n]
n


Please specify data filepath:
images\鲍.png

Loading the data.....................................


Please specify number of predictions per input image:
5

Melnyk-Net response:

Dear User, I am 99.986 % sure it's     鲍
Dear User, I am 0.011 % sure it's     鸵
Dear User, I am 0.001 % sure it's     跑
Dear User, I am 0.001 % sure it's     鳃
Dear User, I am 0.000 % sure it's     鲸

Continue? [Y/n]
n

Thanks for using Melnyk-Net Offline HCCR software!
```
